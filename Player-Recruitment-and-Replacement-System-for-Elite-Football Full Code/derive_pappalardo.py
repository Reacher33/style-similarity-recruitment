"""
derive_pappalardo.py
====================
Derives per-player season statistics from the Pappalardo et al. (2019)
public soccer-logs dataset (Wyscout, Big Five leagues, 2017-18).

INPUT (from the figshare collection, DOI 10.6084/m9.figshare.c.4415000):
    raw/matches/matches_<League>.json   (lineups + substitutions -> minutes)
    raw/events/events_<League>.json     (all match events)
    raw/players.json                    (names, roles, birth dates)

OUTPUT:
    data/processed/players_derived.csv  one row per player, season counts
                                        + minutes, ready for the notebook

METRIC DEFINITIONS (Wyscout tag ids per the dataset documentation):
    accurate            tag 1801        key pass        tag 302
    assist              tag 301         goal            tag 101
    interception        tag 1401        duel won        tag 703
    Pitch coordinates are percentages: x 0-100 toward opponent goal,
    y 0-100. Penalty area approximated as x>=84, 19<=y<=81.
    Progressive pass: accurate pass advancing the ball >=15 units (~15.7m).
    Progressive distance: sum of forward x-gain (x1.05 -> metres).

USAGE:
    python derive_pappalardo.py --data-dir raw --out data/processed/players_derived.csv
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

LEAGUES = {
    "England": "Premier League",
    "Spain": "La Liga",
    "Germany": "Bundesliga",
    "Italy": "Serie A",
    "France": "Ligue 1",
}

ACCURATE, KEY_PASS, ASSIST, GOAL, INTERCEPT, WON = 1801, 302, 301, 101, 1401, 703
PEN_X, PEN_Y_LO, PEN_Y_HI = 84, 19, 81
PROG_GAIN = 15          # units of x-gain to count a pass as progressive
METRES_PER_X = 1.05     # 105m pitch / 100 units

COUNT_FIELDS = [
    "minutes", "matches",
    "passes", "passes_accurate", "key_passes", "assists", "smart_passes",
    "passes_into_penalty_area", "progressive_passes", "progressive_pass_distance",
    "crosses", "long_passes",
    "shots", "shots_on_target", "goals",
    "dribbles_won", "dribbles_attempted", "accelerations", "touches_att_pen_area",
    "tackles_won", "tackles_attempted", "interceptions", "clearances", "aerials_won",
]


def tag_ids(e):
    return {t["id"] for t in e.get("tags", [])}


def minutes_from_matches(matches: list) -> dict[int, dict]:
    """playerId -> {'minutes': m, 'matches': n} from lineups + substitutions."""
    acc = defaultdict(lambda: {"minutes": 0.0, "matches": 0})
    for m in matches:
        for team in m.get("teamsData", {}).values():
            f = team.get("formation") or {}
            subs = f.get("substitutions") or []
            if subs == "null":
                subs = []
            out_min = {s["playerOut"]: s["minute"] for s in subs}
            in_min = {s["playerIn"]: s["minute"] for s in subs}
            for p in f.get("lineup") or []:
                pid = p["playerId"]
                mins = min(out_min.get(pid, 90), 90)
                acc[pid]["minutes"] += max(mins, 0)
                acc[pid]["matches"] += 1
            for p in f.get("bench") or []:
                pid = p["playerId"]
                if pid in in_min:
                    mins = max(90 - in_min[pid], 0)
                    if mins > 0:
                        acc[pid]["minutes"] += mins
                        acc[pid]["matches"] += 1
    return acc


def in_pen_area(pos) -> bool:
    return pos["x"] >= PEN_X and PEN_Y_LO <= pos["y"] <= PEN_Y_HI


def derive_league(events: list) -> dict[int, dict]:
    """playerId -> counting stats for one league's events."""
    S = defaultdict(lambda: defaultdict(float))

    for e in events:
        pid = e.get("playerId", 0)
        if not pid:
            continue
        s = S[pid]
        name, sub = e.get("eventName"), e.get("subEventName")
        tags = tag_ids(e)
        pos = e.get("positions") or []
        start = pos[0] if pos else None
        end = pos[1] if len(pos) > 1 else None

        # touches in the attacking penalty area (any on-ball action started there)
        if start and name in ("Pass", "Shot", "Duel", "Others on the ball", "Free Kick") \
           and in_pen_area(start):
            s["touches_att_pen_area"] += 1

        if INTERCEPT in tags:
            s["interceptions"] += 1

        if name == "Pass":
            s["passes"] += 1
            acc = ACCURATE in tags
            if acc:
                s["passes_accurate"] += 1
            if KEY_PASS in tags:
                s["key_passes"] += 1
            if ASSIST in tags:
                s["assists"] += 1
            if sub == "Smart pass" and acc:
                s["smart_passes"] += 1
            if sub == "Cross":
                s["crosses"] += 1
            if sub in ("Launch", "High pass"):
                s["long_passes"] += 1
            if acc and start and end:
                gain = end["x"] - start["x"]
                if end and in_pen_area(end):
                    s["passes_into_penalty_area"] += 1
                if gain >= PROG_GAIN:
                    s["progressive_passes"] += 1
                if gain > 0:
                    s["progressive_pass_distance"] += gain * METRES_PER_X

        elif name == "Shot":
            s["shots"] += 1
            if ACCURATE in tags:
                s["shots_on_target"] += 1
            if GOAL in tags:
                s["goals"] += 1

        elif name == "Free Kick":
            # penalties and direct free kicks can score
            if GOAL in tags:
                s["goals"] += 1
            if sub in ("Free kick shot", "Penalty"):
                s["shots"] += 1
                if ACCURATE in tags:
                    s["shots_on_target"] += 1

        elif name == "Duel":
            if sub == "Ground attacking duel":
                s["dribbles_attempted"] += 1
                if WON in tags:
                    s["dribbles_won"] += 1
            elif sub in ("Ground defending duel",):
                s["tackles_attempted"] += 1
                if WON in tags:
                    s["tackles_won"] += 1
            elif sub == "Air duel" and WON in tags:
                s["aerials_won"] += 1

        elif name == "Others on the ball":
            if sub == "Clearance":
                s["clearances"] += 1
            elif sub == "Acceleration":
                s["accelerations"] += 1

    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="raw",
                    help="folder containing matches/, events/, players.json")
    ap.add_argument("--out", default="data/processed/players_derived.csv")
    a = ap.parse_args()
    root = Path(a.data_dir)

    players_meta = {p["wyId"]: p for p in json.load(open(root / "players.json"))}
    print(f"players.json: {len(players_meta)} players")

    rows: dict[int, dict] = {}
    for key, league in LEAGUES.items():
        print(f"\n=== {league} ===")
        matches = json.load(open(root / "matches" / f"matches_{key}.json"))
        events = json.load(open(root / "events" / f"events_{key}.json"))
        print(f"  {len(matches)} matches, {len(events)} events")

        mins = minutes_from_matches(matches)
        stats = derive_league(events)

        # union of players seen in lineups or events
        for pid in set(mins) | set(stats):
            meta = players_meta.get(pid)
            if meta is None:
                continue
            r = rows.setdefault(pid, {
                "player_id": pid,
                "player": (meta.get("shortName") or "").encode()
                          .decode("unicode-escape"),
                "position": {"MD": "MF"}.get((meta.get("role") or {}).get("code2", ""),
                             (meta.get("role") or {}).get("code2", "")),
                "birth_date": meta.get("birthDate", ""),
                "league": league,
                **{f: 0.0 for f in COUNT_FIELDS},
            })
            r["league"] = league          # last league seen; fine for single-league players
            r["minutes"] += mins.get(pid, {}).get("minutes", 0)
            r["matches"] += mins.get(pid, {}).get("matches", 0)
            for f, v in stats.get(pid, {}).items():
                r[f] += v

    df = pd.DataFrame(rows.values())
    # season 2017-18 ages
    bd = pd.to_datetime(df["birth_date"], errors="coerce")
    df["age"] = (pd.Timestamp("2018-01-01") - bd).dt.days // 365
    df["season"] = "2017-18"
    import numpy as np
    df["passes_pct"] = (100 * df["passes_accurate"] /
                        df["passes"].replace(0, np.nan)).round(1)
    df["dribble_success_pct"] = (100 * df["dribbles_won"] /
                                 df["dribbles_attempted"].replace(0, np.nan)).round(1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\nSaved {len(df)} players -> {a.out}")
    mf = df[(df["position"] == "MF") & (df["minutes"] >= 900)]
    print(f"Midfielders with 900+ minutes: {len(mf)}")
    print(mf.groupby("league")["player"].count().to_string())


if __name__ == "__main__":
    main()
