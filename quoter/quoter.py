import math
from framework_config import logit_scaled_edge

def calculate_quote_levels(
    bid_theo: float,
    offer_theo: float,
    top_level_bid: float,
    top_level_offer: float,
    min_distance_from_top_level: float,
    min_edge: float,
    min_absolute_edge: float,
    volumes: list[int],
    tick_step: int = 1,
    fav_edge_k: float = 0.0,
    own_bid_prices: frozenset = frozenset(),
    own_offer_prices: frozenset = frozenset(),
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Calculates multiple levels of bids and offers you are willing to quote.
    
    Args:
        volumes: List of integer sizes for each level we want to quote.
        tick_step: How many cents worse each subsequent level should be priced. 
                   Defaults to 1.
                   
    Returns:
        (bids, offers) where each is a list of (price_cents, size) tuples.
        
    Base Level Logic:
      For bids (buying YES): Lower price is better. Cap bid at the minimum
      of theoretical edge and the market's top level minus a distance.
      
      For offers (selling YES): Higher price is better. Bottom offer at the maximum
      of theoretical edge and the market's top level plus a distance.
    """
    # Per-side variance-scaled edge: narrowest requirement at extreme theos,
    # widest at 50c. min_absolute_edge floors it (matters when min_absolute_edge > min_edge).
    #
    # ORIENTATION (fav_edge_k, 2026-08-09): favourite_edge_mult is asymmetric, so
    # it needs the price of the side we go LONG, not the ticker's YES price.
    #   bid   = we BUY YES        -> our-side price is bid_theo
    #   offer = we SELL YES = BUY NO -> our-side price is 100 - offer_theo
    # Posting an 85c offer is buying NO at 15c (an underdog) and correctly gets
    # ~1.0x, not the 1.46x that naively passing 85c would apply. The legacy
    # variance term 4p(1-p) is symmetric, so this distinction is new with fav_edge_k.
    bid_edge = logit_scaled_edge(
        bid_theo, min_edge, min_absolute_edge,
        fav_edge_k=fav_edge_k, our_side_cents=bid_theo,
    )
    offer_edge = logit_scaled_edge(
        offer_theo, min_edge, min_absolute_edge,
        fav_edge_k=fav_edge_k, our_side_cents=100.0 - offer_theo,
    )

    bids = []
    offers = []

    # ILLIQUID MARKET CHECK: If the Kalshi orderbook is completely native-empty on a side (0c bid or 100c offer),
    # physically abort the entire quoting layer for that side so we don't automatically anchor down to 1c / 99c!
    # SELF-JUMP GUARD (2026-09-09, prerequisite for negative min_distance):
    # a top level whose price matches one of OUR OWN resting orders is JOINED,
    # never improved. Without this, any scrub miss (placement->tracking lag,
    # orphans) makes a negative min_distance leapfrog our own order every
    # cycle until min_edge binds — the 24->26->28 ratchet. With it, -1 is
    # stable under every failure mode:
    #   external 26, us at 27          -> top 26 external -> cap 27, stay ✓
    #   scrub misses our 27            -> top 27 == ours  -> cap 27, stay ✓
    #   competitor JOINS our 27        -> top 27 == ours  -> stay (they're
    #                                     behind us in FIFO — improving pays
    #                                     a cent for priority we already own)
    #   competitor improves to 28      -> top 28 external -> cap 29 (bounded
    #                                     by theo - edge, as ever)
    if top_level_bid > 0:
        if int(round(top_level_bid)) in own_bid_prices:
            bid_cap = float(int(round(top_level_bid)))
        else:
            bid_cap = top_level_bid - min_distance_from_top_level
        raw_bid_level = min(bid_theo - bid_edge, bid_cap)
        base_bid_level = int(math.floor(raw_bid_level))
    else:
        base_bid_level = None

    if top_level_offer < 100:
        if int(round(top_level_offer)) in own_offer_prices:
            offer_cap = float(int(round(top_level_offer)))
        else:
            offer_cap = top_level_offer + min_distance_from_top_level
        raw_offer_level = max(offer_theo + offer_edge, offer_cap)
        base_offer_level = int(math.ceil(raw_offer_level))
    else:
        base_offer_level = None
    
    bids = []
    offers = []
    
    for i, vol in enumerate(volumes):
        if vol <= 0:
            continue
            
        # Bound to valid Kalshi cent prices (1 to 99). Drop entirely if mathematics command impossible limits!
        if base_bid_level is not None:
            b_level = base_bid_level - (i * tick_step)
            if 1 <= b_level <= 99:
                bids.append((b_level, vol))
            
        if base_offer_level is not None:
            o_level = base_offer_level + (i * tick_step)
            if 1 <= o_level <= 99:
                offers.append((o_level, vol))
        
    return bids, offers
