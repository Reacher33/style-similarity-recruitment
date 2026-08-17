FULL PROJECT CODE — Data-Driven Player Recruitment System
=========================================================
Everything the dissertation reports, in one place. The notebook is
PRE-EXECUTED (zero errors) so every result is inspectable immediately.
An interactive demonstrator is included so the system can be USED, not
just read — provided two ways (browser and Python) so it runs anywhere.

CONTENTS
--------
Football_Recruitment_System.ipynb  the complete pipeline, sections 1-12:
    1-6   load derived data, per-90, five style dimensions, corr+PCA
    7     k-means archetypes (k selection documented) + naming by z-deviation
    8     recruitment engine: Mode 1 (replacement) & Mode 2 (role search)
    9     summary
    10    RF/DT feature validation + confusion/boundary analysis (Chapter 4)
    11    comparative baselines: cosine vs euclidean, constrained vs not (Ch 5)
    12    RQ3 value-efficiency rankings on real 2017-18 values (Ch 5)
derive_pappalardo.py     raw Wyscout events -> per-player season metrics
merge_valuations.py      2017-18 Transfermarkt snapshot + 5-stage name match
rf_dt_analysis.py        standalone script version of section 10
Project_Pseudocode.docx  language-neutral pseudocode of all 8 algorithms
data/processed/players_derived.csv   derivation output (2,568 players)
data/raw/values.csv                  2017-18 values, 566/590 midfielders
data/raw/values_audit.csv            per-player match audit
outputs/                             all generated figures

demo/                                interactive demonstrator (see below)
    index.html            zero-install browser version, fully offline
    recruitment_app.py    Streamlit version, runs alongside the pipeline
    players.json          the 590-player dataset the demo reads

INTERACTIVE DEMONSTRATOR
------------------------
A working front end for the pipeline: pick a departing player and get a
ranked shortlist of stylistically similar candidates with similarity,
market value, and value-efficiency; or set a role profile with five
sliders and get the best-value matches. The archetype constraint, budget
filter, and similarity-vs-value ranking are all exposed, so the Chapter 5
comparative study can be explored interactively. Every figure it shows is
reproduced from the same derived data as the dissertation (e.g. De Bruyne
-> Pastore 98.1%, David Silva 97.8%; Kanté -> N'Zonzi 97.5%).

Provided two ways:

  index.html         Double-click to open in any browser. Self-contained:
                     React is inlined and the JSX is pre-compiled, so it
                     needs NO internet, NO install, and NO build step.
                     This is the version to carry on a laptop or USB stick.
                     To serve it via GitHub Pages, keep the name index.html
                     and point Pages at the demo/ folder (or move it to the
                     repo root).

  recruitment_app.py Streamlit version, for running in the same Python
                     environment as the pipeline:
                         pip install streamlit pandas numpy
                         streamlit run recruitment_app.py
                     Opens in the browser at http://localhost:8501.
                     Keep players.json in the same folder.

The demonstrator is decision-support, not a signing engine: it describes
playing style and market value only, and deliberately excludes injury
history, contract status, wages, and temperament, all of which a real
recruitment decision must weigh. Because 2017-18 is historical, every
recommendation can be checked against the transfers that actually followed.

RUN ORDER (to reproduce from scratch)
-------------------------------------
1. Obtain raw dataset (figshare DOI 10.6084/m9.figshare.c.4415000, or the
   GitHub mirror koenvo/wyscout-soccer-match-event-dataset) -> raw/
2. python derive_pappalardo.py --data-dir raw --out data/processed/players_derived.csv
3. (values.csv already provided; to rebuild: merge_valuations.py, see header)
4. jupyter notebook Football_Recruitment_System.ipynb -> Run All
5. (optional) launch the demonstrator: open demo/index.html in a browser,
   or `streamlit run demo/recruitment_app.py`

All seeds fixed (random_state=42). Note: re-running on different sklearn
versions can shift CV accuracies by ~0.2-0.5pp (e.g. 87.3 vs 87.5); this is
version jitter, not nondeterminism within an environment.