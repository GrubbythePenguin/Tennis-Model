import requests
import time
import threading
import logging

log = logging.getLogger(__name__)

# ESPN MLS endpoint
ESPN_MLS_URL = "http://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard"

# Soft-matching Kalshi MLS Team Abbreviations to substring of ESPN team names
KALSHI_TO_MLS = {
    "CHI": "Chicago", "CIN": "Cincinnati", "CLB": "Columbus", "DCU": "D.C.",
    "MIA": "Miami", "MTL": "Montréal", "NE": "New England", "NYC": "NY City",
    "NYRB": "New York", "ORL": "Orlando", "PHI": "Philadelphia", "TOR": "Toronto",
    "ATX": "Austin", "COL": "Colorado", "DAL": "Dallas", "HOU": "Houston",
    "LAG": "LA Galaxy", "LAFC": "Los Angeles FC", "MIN": "Minnesota", "POR": "Portland",
    "RSL": "Salt Lake", "SJE": "San Jose", "SEA": "Seattle", "SKC": "Sporting KC",
    "STL": "St. Louis", "VAN": "Vancouver", "CLT": "Charlotte", "NSH": "Nashville", 
    "ATL": "Atlanta"
}

class SoccerTracker:
    def __init__(self):
        self._cache = {}
        self._running = False
        self._thread = None
        
    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True, name="SoccerTrackerThread")
            self._thread.start()
            log.info("Soccer Live Tracker daemon spun up natively.")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                data = requests.get(ESPN_MLS_URL, timeout=10).json()
                events = data.get('events', [])
                
                fresh_cache = {}
                for event in events:
                    home_team = event['competitions'][0]['competitors'][0]
                    away_team = event['competitions'][0]['competitors'][1]
                    
                    home_name = home_team['team']['displayName']
                    away_name = away_team['team']['displayName']
                    
                    # Convert names to Kalshi tags implicitly
                    home_tag = None
                    away_tag = None
                    for k, v in KALSHI_TO_MLS.items():
                        if v.lower() in home_name.lower(): home_tag = k
                        if v.lower() in away_name.lower(): away_tag = k
                            
                    if not home_tag or not away_tag:
                        continue
                        
                    # Extract Time
                    clock = event['status']['displayClock']
                    minute = 0
                    if 'HT' in clock.upper() or 'HALF' in clock.upper():
                        minute = 45
                    elif 'FT' in clock.upper() or 'FINAL' in clock.upper():
                        minute = 90
                    else:
                        try:
                            m_str = clock.replace('+', '').replace("'", "")
                            minute = int(m_str.split(':')[0]) if ':' in m_str else int(m_str)
                        except:
                            raw = event['status']['clock']
                            minute = int(raw / 60) if raw > 300 else int(raw)
                    
                    # Extract Score
                    home_score = int(home_team.get('score', 0))
                    away_score = int(away_team.get('score', 0))
                    
                    # Extract Red Cards 
                    h_reds, a_reds = 0, 0
                    for stat in home_team.get('statistics', []):
                        if stat.get('name') == 'redCards': h_reds = int(stat.get('displayValue', 0))
                    for stat in away_team.get('statistics', []):
                        if stat.get('name') == 'redCards': a_reds = int(stat.get('displayValue', 0))
                        
                    # Build unique key for this matchup e.g. ATLLAFC, HOUDAL
                    # Usually Kalshi does AWAYHOME
                    match_key_1 = f"{away_tag}{home_tag}"
                    match_key_2 = f"{home_tag}{away_tag}"
                    
                    game_state = {
                        "minute": minute,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_red": h_reds,
                        "away_red": a_reds,
                        "status": event['status']['type']['detail'],
                        "home_name": home_name,
                        "away_name": away_name,
                        "home_tag": home_tag,
                        "away_tag": away_tag
                    }
                    
                    fresh_cache[match_key_1] = game_state
                    fresh_cache[match_key_2] = game_state
                    
                self._cache = fresh_cache
            except Exception as e:
                log.error(f"Soccer Tracker API poll failed: {e}")
                
            time.sleep(30) # Poll every 30 seconds

    def get(self, game_tag: str) -> dict:
        """ Returns the structured dictionary match for the given Kalshi game tag if active. """
        return self._cache.get(game_tag)

_instance = None
def get_instance():
    global _instance
    if _instance is None:
        _instance = SoccerTracker()
        _instance.start()
    return _instance
