import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Global runtime mapping for thread-safe memory sharing across decoupled python loops!
ORDER_TELEMETRY: Dict[str, Dict[str, Any]] = {}
MAX_TELEMETRY_SIZE = 5000

def put_telemetry(order_id: str, raw_theo: float, adjusted_theo: float, arb_context: str = "", trigger_type: str = "", quoted_size: int = 0, was_size_capped: bool = False, sweep_id: str = "", sweep_total_vol: int = 0, skew_shift_cents: Optional[float] = None):
    """
    Safely binds mathematical algorithmic state arrays directly into the Kalshi Order UUID.

    `quoted_size` and `was_size_capped` are logging-only: used by trade_logger
    and watch_fills to flag full-fill events for PTA. Never read by execution.
    `was_size_capped=True` means our internal sizing config (max_fire_size OR
    the edge-derived `authorized` tier from base_vol × scaling) bound the
    sweep — i.e., raising one of those configs would have given us more fills.
    False means edge / book depth / position room was the binding constraint.

    `sweep_id` + `sweep_total_vol` are set for cross-book sweep orders so
    trade_logger can aggregate cumulative taker fills across both legs of
    the same sweep (the cap binds on combined vol, not per-leg). When
    present, trade_logger uses sweep_id as the cumulative key and
    sweep_total_vol as the qsz threshold instead of per-order values.
    """
    global ORDER_TELEMETRY
    if not order_id:
        return

    ORDER_TELEMETRY[order_id] = {
        "raw_theo": raw_theo,
        "adjusted_theo": adjusted_theo,
        "arb_context": arb_context,
        "trigger_type": trigger_type,
        "quoted_size": quoted_size,
        "was_size_capped": was_size_capped,
        "sweep_id": sweep_id,
        "sweep_total_vol": sweep_total_vol,
        "skew_shift_cents": skew_shift_cents,
    }

    # Write to shared file so watch_fills.py can read edge data
    try:
        import json
        entry = {"id": order_id, "raw": raw_theo, "adj": adjusted_theo}
        if trigger_type:
            entry["trig"] = trigger_type
        if quoted_size:
            entry["qsz"] = quoted_size
        if was_size_capped:
            entry["scap"] = True
        if sweep_id:
            entry["swid"] = sweep_id
            entry["sw_total"] = sweep_total_vol
        # Independent signed skew shift in cents (+ = skew favored the take,
        # - = skew opposed). Surfaced by watch_fills for taker visibility.
        # None when the writer didn't compute it (legacy paths) — watch_fills
        # then falls back to adj-raw delta.
        if skew_shift_cents is not None:
            entry["skew"] = skew_shift_cents
        with open("_telemetry.jsonl", "a") as tf:
            tf.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    
    # Organic limit truncating natively protects RAM bounds from ghost/evicted orders
    if len(ORDER_TELEMETRY) > MAX_TELEMETRY_SIZE:
        keys_to_keep = list(ORDER_TELEMETRY.keys())[-2000:]
        ORDER_TELEMETRY = {k: ORDER_TELEMETRY[k] for k in keys_to_keep}

def get_telemetry(order_id: str) -> Dict[str, Any]:
    """
    Pulls telemetry directly for WebSocket threads resolving fills dynamically. 
    Returns an empty dict dynamically bypassing KeyError natively if the order was placed externally/manually!
    """
    return ORDER_TELEMETRY.get(order_id, {})
