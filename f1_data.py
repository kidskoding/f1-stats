"""
F1 Data Fetching and Caching Module.

Fetches driver data from the Jolpica F1 API (Ergast-compatible),
caches responses locally, and builds pandas DataFrames.
"""

import json
import os
import re
import time
from collections import OrderedDict

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASONS = range(2015, 2025)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "f1_cache")
PAGE_SIZE = 100


def _fetch_json(url: str, retries: int = 3) -> dict:
    """Fetch JSON from a URL with rate limiting and retry on 429."""
    for attempt in range(retries):
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(1.0)
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def _get_cached_or_fetch(cache_key: str, url: str) -> dict:
    """Return cached JSON if available, otherwise fetch and cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    data = _fetch_json(url)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def _merge_race_pages(pages: list[list[dict]], result_key: str) -> list[dict]:
    """Merge paginated race data where a single race can split across pages."""
    merged = OrderedDict()
    for races in pages:
        for race in races:
            key = race["round"]
            if key in merged:
                merged[key][result_key].extend(race.get(result_key, []))
            else:
                merged[key] = race
    return list(merged.values())


def _fetch_paginated(base_url: str, table_key: str, result_key: str, cache_key: str) -> list[dict]:
    """Fetch paginated race data, merging across pages."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    first_url = f"{base_url}?limit={PAGE_SIZE}&offset=0"
    first = _fetch_json(first_url)
    total = int(first["MRData"]["total"])
    pages = [first["MRData"][table_key]["Races"]]

    offset = PAGE_SIZE
    while offset < total:
        url = f"{base_url}?limit={PAGE_SIZE}&offset={offset}"
        data = _fetch_json(url)
        pages.append(data["MRData"][table_key]["Races"])
        offset += PAGE_SIZE

    merged = _merge_race_pages(pages, result_key)

    with open(cache_path, "w") as f:
        json.dump(merged, f)
    return merged


def _is_dnf(status: str) -> bool:
    """Determine if a race status indicates a DNF."""
    if status == "Finished":
        return False
    if re.match(r"\+\d+ Lap", status):
        return False
    if status == "Lapped":
        return False
    return True


def fetch_standings(season: int) -> list[dict]:
    """Fetch driver standings for a season."""
    url = f"{BASE_URL}/{season}/driverStandings.json?limit=100"
    data = _get_cached_or_fetch(f"standings_{season}", url)
    standings_lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings_lists:
        return []
    return standings_lists[0]["DriverStandings"]


def fetch_results(season: int) -> list[dict]:
    """Fetch all race results for a season (paginated)."""
    url = f"{BASE_URL}/{season}/results.json"
    return _fetch_paginated(url, "RaceTable", "Results", f"results_{season}")


def fetch_qualifying(season: int) -> list[dict]:
    """Fetch all qualifying results for a season (paginated)."""
    url = f"{BASE_URL}/{season}/qualifying.json"
    return _fetch_paginated(url, "RaceTable", "QualifyingResults", f"qualifying_{season}")


def build_standings_df() -> pd.DataFrame:
    """Build a DataFrame of driver standings across all seasons."""
    rows = []
    for season in SEASONS:
        print(f"  Fetching {season} standings...")
        standings = fetch_standings(season)
        for entry in standings:
            driver = entry["Driver"]
            constructor = entry["Constructors"][0] if entry["Constructors"] else {}
            rows.append({
                "season": season,
                "position": int(entry["position"]) if "position" in entry else None,
                "points": float(entry["points"]),
                "wins": int(entry["wins"]),
                "driver_id": driver["driverId"],
                "driver_code": driver.get("code", ""),
                "driver_name": f"{driver['givenName']} {driver['familyName']}",
                "constructor": constructor.get("name", ""),
                "nationality": driver.get("nationality", ""),
            })
    return pd.DataFrame(rows).sort_values(["season", "position"]).reset_index(drop=True)


def build_results_df() -> pd.DataFrame:
    """Build a DataFrame of race results across all seasons."""
    rows = []
    for season in SEASONS:
        print(f"  Fetching {season} results...")
        races = fetch_results(season)
        for race in races:
            for result in race.get("Results", []):
                driver = result["Driver"]
                constructor = result["Constructor"]
                pos_text = result.get("positionText", "R")
                try:
                    position = int(result["position"])
                except (ValueError, KeyError):
                    position = None

                fastest_lap = result.get("FastestLap", {})
                try:
                    fl_rank = int(fastest_lap.get("rank", 0))
                except (ValueError, TypeError):
                    fl_rank = None

                rows.append({
                    "season": season,
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "date": race.get("date", ""),
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code", ""),
                    "driver_name": f"{driver['givenName']} {driver['familyName']}",
                    "constructor": constructor.get("name", ""),
                    "grid": int(result.get("grid", 0)),
                    "position": position,
                    "position_text": pos_text,
                    "points": float(result.get("points", 0)),
                    "laps": int(result.get("laps", 0)),
                    "status": result.get("status", ""),
                    "is_dnf": _is_dnf(result.get("status", "")),
                    "fastest_lap_rank": fl_rank,
                })
    return pd.DataFrame(rows).sort_values(["season", "round"]).reset_index(drop=True)


def build_qualifying_df() -> pd.DataFrame:
    """Build a DataFrame of qualifying results across all seasons."""
    rows = []
    for season in SEASONS:
        print(f"  Fetching {season} qualifying...")
        races = fetch_qualifying(season)
        for race in races:
            for result in race.get("QualifyingResults", []):
                driver = result["Driver"]
                constructor = result["Constructor"]
                rows.append({
                    "season": season,
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code", ""),
                    "driver_name": f"{driver['givenName']} {driver['familyName']}",
                    "constructor": constructor.get("name", ""),
                    "position": int(result["position"]),
                    "q1": result.get("Q1"),
                    "q2": result.get("Q2"),
                    "q3": result.get("Q3"),
                })
    return pd.DataFrame(rows).sort_values(["season", "round", "position"]).reset_index(drop=True)


def load_all_data() -> dict[str, pd.DataFrame]:
    """Fetch all F1 data and return as DataFrames."""
    print("Loading standings...")
    standings = build_standings_df()
    print("Loading race results...")
    results = build_results_df()
    print("Loading qualifying...")
    qualifying = build_qualifying_df()

    return {
        "standings": standings,
        "results": results,
        "qualifying": qualifying,
    }


if __name__ == "__main__":
    data = load_all_data()
    for name, df in data.items():
        print(f"\n{name}: {df.shape}")
        print(df.head())
