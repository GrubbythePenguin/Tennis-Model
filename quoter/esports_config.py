# Per-game NA hour thresholds (Eastern Time).
# Maps use Poly CLOB outside NA hours, Kalshi WS during NA hours.
# Value forms:
#   int N            → NA hours = et_hour >= N (open-ended, original behavior)
#   (start, end)     → NA hours = start <= et_hour < end (windowed)
# To change a cutoff, edit the single value here — it propagates to
# run.py, arber_bot.py, and live_series_model.py.

NA_HOUR_ET = {
    "KXLOL": 13,         # 1pm ET = 10am PST
    "KXDOTA2": (21, 22), # DreamLeague is EU/Asia-driven — only treat 9pm–10pm ET as NA
    "KXCS2": (0, 23),    # Temporary 2026-05-27: Kalshi 0-22 ET, Poly only 23-24 ET (1hr symbolic) — tier-2 CS2 spreads on Kalshi observed tighter than Poly today, tier-1 CS pinned to data_source=poly explicitly so unaffected
    "KXVALORANT": (16, 24),  # 4pm–midnight ET (2026-07-18): VCT Americas evening slate → Kalshi maps
                             # (proven faster+truer than Poly, LOUDKRU 2026-07-17). VCT EU/China/Pacific play
                             # mornings ET → outside window → Poly. Only VCT rows use kalshi_na_poly_eu; all
                             # VCL rows pinned data_source=poly, so VCL (incl. Americas) is unaffected by this window.
}

def is_na_hours(ticker: str) -> bool:
    """Check if a ticker is in NA hours (Kalshi primary for maps)."""
    import datetime, pytz
    et_hour = datetime.datetime.now(pytz.timezone("US/Eastern")).hour
    for prefix, threshold in NA_HOUR_ET.items():
        if prefix in ticker:
            if isinstance(threshold, tuple):
                start, end = threshold
                return start <= et_hour < end
            return et_hour >= threshold
    return et_hour >= 13  # default
