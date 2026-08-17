"""
A Data-Driven Player Recruitment and Replacement System
--------------------------------------------------------
Interactive demonstrator for the MSc dissertation:
"A Data-Driven Player Recruitment and Replacement System for Elite Football:
 Identifying Playing Style Similarity and Market Value Efficiency Using
 Performance Analytics."

Runs the same pipeline described in the dissertation: cosine similarity over
five engineered style dimensions, an optional archetype constraint, and a
value-efficiency score. Built on the derived 2017-18 Big Five leagues dataset
(590 midfielders, >= 900 minutes) from the public Wyscout event archive
(Pappalardo et al., 2019).

Run with:   streamlit run recruitment_app.py
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
DIMS = ["Creativity", "Distribution", "Ball-carrying", "Defensive work", "Goal threat"]
ARCH_COLOR = {
    "Balanced": "#A5A5A5", "Carrier": "#ED7D31", "Distributor": "#4472C4",
    "Ball-Winner": "#FFC000", "Creator": "#5B9BD5",
}

@st.cache_data
def load_players():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "players.json"), encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    df = pd.DataFrame({
        "name": [r[0] for r in rows],
        "league": [data["leagues"][r[1]] for r in rows],
        "arch": [data["arch"][r[2]] for r in rows],
        "minutes": [r[3] for r in rows],
        "value": [None if r[4] < 0 else r[4] for r in rows],
    })
    Z = np.array([r[5:] for r in rows], dtype=float)      # standardised style vectors
    mean = np.array(data["mean"], dtype=float)
    std = np.array(data["std"], dtype=float)
    return df, Z, mean, std

df, Z, MEAN, STD = load_players()
NORMS = np.linalg.norm(Z, axis=1)


def cosine_to(vec):
    """Cosine similarity of a single style vector against every player."""
    dots = Z @ vec
    denom = NORMS * (np.linalg.norm(vec) + 1e-12)
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0)


def value_efficiency(sim, value):
    v = 1.0 if value is None else max(value, 1.0)
    return sim / (v / 10.0)


def fmt_val(v):
    if v is None:
        return "n/a"
    return f"€{v:.0f}m" if v >= 10 else f"€{v:.1f}m"


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="Player Recruitment System", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background-color: #0E1C19; }
      h1, h2, h3, p, label, .stMarkdown { color: #EFEDE6; }
      .metric-cyan { color: #63D6C6; font-family: monospace; }
      .metric-amber { color: #E9B33C; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='color:#8FA8A0;font-size:12px;letter-spacing:0.15em;"
    "text-transform:uppercase;font-family:monospace'>"
    "Big Five leagues · 2017-18 · 590 midfielders</div>",
    unsafe_allow_html=True,
)
st.title("Player Recruitment & Replacement System")
st.markdown(
    "<p style='color:#8FA8A0;font-size:15px;max-width:720px'>"
    "Finds stylistically equivalent players using cosine similarity over five "
    "engineered style dimensions, then ranks them by how much of that fit each "
    "euro buys.</p>",
    unsafe_allow_html=True,
)

mode = st.radio("Mode", ["Replace a player", "Search by role"], horizontal=True, label_visibility="collapsed")

left, right = st.columns([1, 2], gap="large")

# ----------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------
with left:
    if mode == "Replace a player":
        default_ix = int(df.index[df.name == "K. De Bruyne"][0]) if (df.name == "K. De Bruyne").any() else 0
        names = df.name.tolist()
        target_name = st.selectbox("Departing player", names, index=default_ix)
        t = df.index[df.name == target_name][0]
        probe = Z[t]
        target_arch = df.at[t, "arch"]

        st.markdown(
            f"<div style='background:#16302A;border:1px solid #24463D;border-radius:8px;"
            f"padding:12px;margin:8px 0'>"
            f"<div style='display:flex;justify-content:space-between'>"
            f"<b style='color:#EFEDE6'>{target_name}</b>"
            f"<span class='metric-amber'>{fmt_val(df.at[t,'value'])}</span></div>"
            f"<div style='color:{ARCH_COLOR[target_arch]};font-family:monospace;font-size:12px'>"
            f"{target_arch} · {df.at[t,'league']} · {int(df.at[t,'minutes']):,} min</div></div>",
            unsafe_allow_html=True,
        )
        restrict = st.checkbox("Restrict to same archetype", value=True)
    else:
        st.markdown("**Desired role profile**")
        st.caption("Set the profile you want. The system returns the closest players, ranked by value.")
        role = [st.slider(d, 0.0, 1.0, v, 0.01) for d, v in
                zip(DIMS, [0.55, 0.45, 0.5, 0.3, 0.45])]
        probe = (np.array(role) - MEAN) / STD
        restrict = False
        target_arch = None

    st.divider()
    budget = st.slider("Maximum budget (€m)", 1, 120, 120,
                       help="Slide to 120 for no limit.")
    rank_by = st.radio("Rank by", ["Similarity", "Value efficiency"], horizontal=True)

    st.caption(
        "Value efficiency = similarity ÷ (market value in units of €10m), "
        "with value floored at €1m. Valuations cover 566 of 590 players."
    )

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
sims = cosine_to(probe)
res = df.copy()
res["sim"] = sims

if mode == "Replace a player":
    res = res[res.name != target_name]
if restrict and target_arch is not None:
    res = res[res.arch == target_arch]
if budget < 120:
    res = res[res.value.notna() & (res.value <= budget)]

res["ve"] = [value_efficiency(s, v) for s, v in zip(res.sim, res.value)]

if rank_by == "Value efficiency":
    res = res[res.value.notna()].sort_values("ve", ascending=False)
else:
    res = res.sort_values("sim", ascending=False)

res = res.head(12).reset_index(drop=True)

with right:
    scope = (f"{target_arch} only" if (mode == "Replace a player" and restrict)
             else "all archetypes")
    st.markdown(
        f"<div style='color:#8FA8A0;font-size:12px;letter-spacing:0.12em;"
        f"text-transform:uppercase;font-family:monospace'>"
        f"{len(res)} candidates · {scope}</div>",
        unsafe_allow_html=True,
    )

    if len(res) == 0:
        st.info("No players match these constraints. Raise the budget or turn off the archetype restriction.")
    else:
        table = pd.DataFrame({
            "#": range(1, len(res) + 1),
            "Player": res.name,
            "Archetype": res.arch,
            "League": res.league,
            "Similarity": [f"{s*100:.1f}%" for s in res.sim],
            "Value": [fmt_val(v) for v in res.value],
            "Value eff.": [f"{v:.2f}" if pd.notna(v) else "—" for v in res.ve],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)

        # ---- comparison for a chosen candidate ----
        pick = st.selectbox("Compare a candidate's profile", res.name.tolist())
        p = res.index[res.name == pick][0]
        pz = Z[df.index[df.name == pick][0]]

        comp = pd.DataFrame({
            "Dimension": DIMS,
            pick: pz,
            ("Target" if mode == "Replace a player" else "Requested role"):
                (probe if mode == "Search by role" else Z[t]),
        }).set_index("Dimension")
        st.markdown(f"**{pick} — style profile (z-scores vs population mean)**")
        st.bar_chart(comp, horizontal=True, height=260)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Similarity", f"{res.at[p,'sim']*100:.1f}%")
        c2.metric("Market value", fmt_val(res.at[p, "value"]))
        c3.metric("Value efficiency",
                  f"{res.at[p,'ve']:.2f}" if pd.notna(res.at[p, "ve"]) else "—")
        c4.metric("Minutes", f"{int(df.at[df.index[df.name==pick][0],'minutes']):,}")

st.divider()
st.caption(
    "Demonstrator running on 2017-18 season data derived from the public Wyscout "
    "event archive (Pappalardo et al., 2019). Because the season is historical, every "
    "recommendation can be checked against what the market did next. It is a decision-support "
    "tool: it excludes injury history, contract status, wages, and temperament, all of which "
    "a real signing must weigh."
)
