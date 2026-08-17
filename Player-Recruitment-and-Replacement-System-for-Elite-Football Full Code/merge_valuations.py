"""
merge_valuations.py
===================
Builds data/raw/values.csv (player, market_value_eur_m) for the 2017-18
season by joining the dcaribou/transfermarkt-datasets valuation history
to the derived player list.

GET THE TWO INPUT FILES (browser, ~1 min):
  https://github.com/dcaribou/transfermarkt-datasets
  -> the README links the prepared CSVs; download:
       players.csv            (id -> name)
       player_valuations.csv  (id, date, market_value_in_eur)
  Put both in data/raw/

RUN:
  python merge_valuations.py
  python merge_valuations.py --snapshot 2018-01-15   (default)

WHAT IT DOES:
  1. For each player in the valuation history, takes the value dated
     closest to the snapshot date *within the 2017-07-01..2018-06-30
     season window* (players with no valuation in that window get none —
     no silent use of modern values).
  2. Matches names three ways, in order of trust:
       exact  : normalised full name
       short  : Wyscout form ('kevin de bruyne' -> 'k de bruyne')
       fuzzy  : token-sort ratio >= 90 (strict, to avoid wrong-namesake
                matches like the Jorginho 0.6m trap)
  3. Writes values.csv + a full audit file (values_audit.csv) showing
     the matched TM name, value, date and method for every player —
     REVIEW THE FUZZY ROWS before trusting RQ3 output.
"""
from __future__ import annotations
import re, unicodedata, argparse
from pathlib import Path
import numpy as np
import pandas as pd

SPECIAL = str.maketrans({"Ø":"O","ø":"o","Đ":"D","đ":"d","Ł":"L","ł":"l","ß":"ss",
                         "Æ":"AE","æ":"ae","Œ":"OE","œ":"oe","İ":"I","ı":"i",
                         "Þ":"Th","þ":"th","Ð":"D","ð":"d"})

def norm(n):
    if not isinstance(n, str): return ""
    a = unicodedata.normalize("NFKD", n.translate(SPECIAL)) \
                   .encode("ascii","ignore").decode().lower()
    return re.sub(r"\s+"," ", re.sub(r"[^\w\s-]","",a)).strip()

def short_form(full_norm):
    parts = full_norm.split()
    if len(parts) < 2: return full_norm
    return parts[0][0] + " " + " ".join(parts[1:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="data/raw/players.csv")
    ap.add_argument("--valuations", default="data/raw/player_valuations.csv")
    ap.add_argument("--derived", default="data/processed/players_derived.csv")
    ap.add_argument("--snapshot", default="2018-01-15")
    ap.add_argument("--season-start", default="2017-07-01")
    ap.add_argument("--season-end", default="2018-06-30")
    ap.add_argument("--out", default="data/raw/values.csv")
    a = ap.parse_args()

    for f in (a.players, a.valuations, a.derived):
        if not Path(f).exists():
            raise SystemExit(f"Missing input: {f}\n(see the header of this script for download steps)")

    players = pd.read_csv(a.players, usecols=lambda c: c in
                          ("player_id","id","name","player_name","pretty_name"))
    id_col = "player_id" if "player_id" in players.columns else "id"
    name_col = next(c for c in ("name","player_name","pretty_name") if c in players.columns)
    players = players.rename(columns={id_col:"player_id", name_col:"tm_name"})[["player_id","tm_name"]]

    vals = pd.read_csv(a.valuations)
    vcol = next(c for c in ("market_value_in_eur","market_value") if c in vals.columns)
    dcol = next(c for c in ("date","datetime","dateweek") if c in vals.columns)
    vals = vals.rename(columns={vcol:"mv", dcol:"date"})
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")

    lo, hi = pd.Timestamp(a.season_start), pd.Timestamp(a.season_end)
    snap = pd.Timestamp(a.snapshot)
    vals = vals[(vals["date"] >= lo) & (vals["date"] <= hi)].copy()
    print(f"valuations inside 2017-18 window: {len(vals)} rows, "
          f"{vals['player_id'].nunique()} players")

    vals["dist"] = (vals["date"] - snap).abs()
    season_val = (vals.sort_values("dist").drop_duplicates("player_id")
                      [["player_id","mv","date"]])
    tm = season_val.merge(players, on="player_id", how="left").dropna(subset=["tm_name"])
    tm["mv_eur_m"] = tm["mv"] / 1e6
    tm["_full"] = tm["tm_name"].apply(norm)
    tm["_short"] = tm["_full"].apply(short_form)
    print(f"TM players with a 2017-18 season value: {len(tm)}")

    lookup_full  = tm.drop_duplicates("_full").set_index("_full")
    lookup_short = tm.drop_duplicates("_short", keep=False).set_index("_short")  # unique shorts only

    try:
        from rapidfuzz import process, fuzz
        fuzzy_keys = lookup_full.index.tolist(); FUZZY = True
    except ImportError:
        FUZZY = False
        print("rapidfuzz not installed — skipping fuzzy stage (pip install rapidfuzz)")

    mine = pd.read_csv(a.derived)
    if "position" in mine.columns and "minutes" in mine.columns:
        mine = mine[(mine["position"]=="MF") & (mine["minutes"]>=900)]
    print(f"derived players to match: {len(mine)}")

    rows = []
    for p in mine["player"]:
        n = norm(p)
        hit, how = None, "none"
        if n in lookup_full.index:
            hit, how = lookup_full.loc[n], "exact"
        elif n in lookup_short.index:
            hit, how = lookup_short.loc[n], "short"
        elif FUZZY:
            r = process.extractOne(n, fuzzy_keys, scorer=fuzz.token_sort_ratio,
                                   score_cutoff=90)
            if r:
                hit, how = lookup_full.loc[r[0]], f"fuzzy_{int(r[1])}"
        if hit is not None:
            rows.append((p, round(float(hit["mv_eur_m"]), 2), hit["tm_name"],
                         str(hit["date"].date()), how))
        else:
            rows.append((p, np.nan, "", "", "none"))

    out = pd.DataFrame(rows, columns=["player","market_value_eur_m",
                                      "tm_name","value_date","match_method"])
    ok = out["market_value_eur_m"].notna().sum()
    print(f"\nMATCH RATE: {ok}/{len(out)} ({100*ok/len(out):.1f}%)")
    print(out["match_method"].str.split("_").str[0].value_counts().to_dict())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out[["player","market_value_eur_m"]].to_csv(a.out, index=False)
    out.to_csv("data/raw/values_audit.csv", index=False)
    print(f"\nWrote {a.out} and data/raw/values_audit.csv")
    print("REVIEW the fuzzy rows in the audit file before running RQ3.")

    for check in ["K. De Bruyne","N. Kanté","L. Modrić","Jorginho"]:
        row = out[out["player"]==check]
        if len(row):
            r = row.iloc[0]
            print(f"  {check:16s} -> {r['market_value_eur_m']} EUR m  "
                  f"[{r['match_method']}] ({r['tm_name']}, {r['value_date']})")

if __name__ == "__main__":
    main()
