"""Ticker alias layer — translate between synthetic tickers (what the bot
sees internally) and real Kalshi tickers (what crosses the wire).

Use case: Kalshi tournament-final markets are listed as tournament-winner
contracts (e.g. KXCS2-IEMCOL26-FAL/-FUR), not as per-game series. For
tournament finals this is functionally identical to a series winner since
exactly two teams remain. We give the bot a synthetic series-shaped ticker
(KXCS2GAME-YYMMMDDHHMMTEAMTEAM-{SUFFIX}) and translate at the API boundary
so the rest of the bot — BO5 framework, position store, kpi_tracker — sees
a normal series ticker.

Invariant: internal state = synthetic; Kalshi API wire = real. Mixing the
two within one code path is the foot-gun.

CSV format (ticker_aliases.csv):
    synthetic,real
    KXCS2GAME-26JUN211100FALFUR-FAL,KXCS2-IEMCOL26-FAL
    KXCS2GAME-26JUN211100FALFUR-FUR,KXCS2-IEMCOL26-FUR

Both directions must be unique (1:1 mapping). Duplicates log a warning and
the first row wins.

Thread-safety: the lookup dicts are reassigned atomically on reload (Python
dict assignment is atomic under the GIL). No lock needed for the read path.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Dict, Iterable, Optional, Set

log = logging.getLogger(__name__)

_DEFAULT_PATH = "ticker_aliases.csv"

# Module-level state. Reassigned (not mutated) on reload so concurrent readers
# always see a consistent snapshot.
_synth_to_real: Dict[str, str] = {}
_real_to_synth: Dict[str, str] = {}
_synth_event_bases: Set[str] = set()
_last_mtime: float = 0.0
_loaded_path: str = ""


def _event_base_of(ticker: str) -> str:
    """Extract the event_base prefix from a synthetic ticker.

    KXCS2GAME-26JUN211100FALFUR-FAL → KXCS2GAME-26JUN211100FALFUR
    Returns the input unchanged if it has no `-` (shouldn't happen for
    well-formed synthetic tickers, but defensive)."""
    idx = ticker.rfind("-")
    if idx <= 0:
        return ticker
    return ticker[:idx]


def load_aliases(path: str = _DEFAULT_PATH) -> int:
    """Load the alias CSV from disk, replacing in-memory state.

    Returns the number of mappings loaded. Safe to call repeatedly; each
    call fully replaces the previous state via atomic dict reassignment.
    Missing file → empty aliases, no error (alias system is opt-in).
    """
    global _synth_to_real, _real_to_synth, _synth_event_bases
    global _last_mtime, _loaded_path

    if not os.path.exists(path):
        _synth_to_real = {}
        _real_to_synth = {}
        _synth_event_bases = set()
        _last_mtime = 0.0
        _loaded_path = path
        return 0

    new_s2r: Dict[str, str] = {}
    new_r2s: Dict[str, str] = {}
    new_ebs: Set[str] = set()

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "synthetic" not in reader.fieldnames or "real" not in reader.fieldnames:
            log.warning("[TICKER ALIAS] %s missing required columns 'synthetic,real'; ignoring", path)
            _synth_to_real = {}
            _real_to_synth = {}
            _synth_event_bases = set()
            _last_mtime = os.path.getmtime(path)
            _loaded_path = path
            return 0

        for row in reader:
            synth = (row.get("synthetic") or "").strip()
            real = (row.get("real") or "").strip()
            if not synth or not real:
                continue
            if synth.startswith("#"):
                continue
            if synth == real:
                log.warning("[TICKER ALIAS] synthetic==real (%s); skipping (no-op row)", synth)
                continue
            if synth in new_s2r:
                log.warning("[TICKER ALIAS] duplicate synthetic %r (existing→%r, new→%r); keeping first",
                            synth, new_s2r[synth], real)
                continue
            if real in new_r2s:
                log.warning("[TICKER ALIAS] duplicate real %r (existing←%r, new←%r); keeping first",
                            real, new_r2s[real], synth)
                continue
            new_s2r[synth] = real
            new_r2s[real] = synth
            new_ebs.add(_event_base_of(synth))

    # Atomic swap.
    _synth_to_real = new_s2r
    _real_to_synth = new_r2s
    _synth_event_bases = new_ebs
    _last_mtime = os.path.getmtime(path)
    _loaded_path = path
    log.info("[TICKER ALIAS] loaded %d mapping(s) from %s", len(new_s2r), path)
    return len(new_s2r)


def reload_if_changed(path: str = "") -> bool:
    """Reload only if the file's mtime advanced since the last load.

    Returns True if a reload happened. Safe to call on every request loop;
    the mtime stat is cheap. Use this in any daemon-like consumer that
    needs to pick up CSV edits without restart.

    `path` defaults to whatever path was last passed to load_aliases() —
    so tests that bootstrap via a tmp CSV won't get clobbered by a
    no-arg reload pulling the production CSV instead. Falls back to
    the module default if nothing has been loaded yet.
    """
    global _loaded_path
    if not path:
        path = _loaded_path or _DEFAULT_PATH
    if path != _loaded_path:
        # First call or explicit path change — force a load.
        load_aliases(path)
        return True
    if not os.path.exists(path):
        if _synth_to_real:
            # File was deleted — clear aliases.
            load_aliases(path)
            return True
        return False
    cur_mtime = os.path.getmtime(path)
    if cur_mtime > _last_mtime:
        load_aliases(path)
        return True
    return False


def to_kalshi(synthetic: str) -> Optional[str]:
    """Synthetic → real Kalshi ticker. Returns None if not aliased.

    Hot path: call at every Kalshi API egress (order POST/amend/cancel,
    WS subscribe). `None` means pass through unchanged (default behavior
    for non-aliased tickers)."""
    return _synth_to_real.get(synthetic)


def from_kalshi(real: str) -> Optional[str]:
    """Real Kalshi ticker → synthetic. Returns None if not aliased.

    Hot path: call at every Kalshi API ingress (WS book delta, fill
    stream, /portfolio response). `None` means pass through unchanged."""
    return _real_to_synth.get(real)


def resolve(ticker: str) -> str:
    """Synthetic → real, with pass-through. Always returns a string.

    Equivalent to `to_kalshi(ticker) or ticker`, but as a single call
    that's safe to drop into any API egress point. Non-aliased tickers
    return unchanged.

    Use at: WS subscribe, order create/amend/cancel, REST market reads."""
    return _synth_to_real.get(ticker, ticker)


def reverse(real: str) -> str:
    """Real → synthetic, with pass-through. Always returns a string.

    Equivalent to `from_kalshi(real) or real`. Non-aliased tickers
    return unchanged.

    Use at: WS orderbook_delta payload dispatch, fill stream dispatch,
    `/portfolio/orders` and `/portfolio/fills` response handling."""
    return _real_to_synth.get(real, real)


def is_synthetic(ticker: str) -> bool:
    """True iff this ticker is a synthetic with a real-ticker alias."""
    return ticker in _synth_to_real


def is_real_aliased(real: str) -> bool:
    """True iff this real Kalshi ticker has a synthetic alias."""
    return real in _real_to_synth


def event_base_is_synthetic(event_base: str) -> bool:
    """True iff at least one synthetic ticker exists under this event_base.

    Used by populate_configs to bypass Kalshi-API validation for synthetic
    events (the API won't recognize the synthetic event_base)."""
    return event_base in _synth_event_bases


def all_synthetic_event_bases() -> Set[str]:
    """Snapshot of all synthetic event_bases. Returns a copy so callers
    can mutate safely without affecting module state."""
    return set(_synth_event_bases)


def all_pairs() -> Iterable[tuple]:
    """Iterate (synthetic, real) pairs. Snapshot — safe for concurrent
    reload."""
    return list(_synth_to_real.items())


# Eager-load on import so callers don't need to remember to call load_aliases()
# explicitly. If the file doesn't exist, this is a no-op.
try:
    load_aliases()
except Exception as e:
    log.warning("[TICKER ALIAS] initial load failed: %r; aliases disabled until reload", e)
