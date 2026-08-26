from dataclasses import dataclass
from enum import Enum

class QuoteSide(Enum):
    BID = "bid"
    OFFER = "offer"

class QuoteStatus(Enum):
    PENDING = "pending"
    LIVE = "live"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class ActiveQuote:
    order_id: str
    client_order_id: str
    side: QuoteSide
    ticker: str
    kalshi_side: str  # typically "yes" or "no"
    limit_cents: int
    size: int
    filled_count: int
    status: QuoteStatus
    # Wall-clock seconds at the moment this order was confirmed posted.
    # Used by reconciliation to decide whether "missing from Kalshi's resting
    # list" means "filled/cancelled" (old enough to be conclusive) or
    # "still propagating" (too fresh to safely remove from local tracking).
    placement_ts: float = 0.0
