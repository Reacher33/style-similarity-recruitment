"""
rf_dt_analysis.py — Chapter 4 experiments.
Feature-importance extraction (Random Forest, Decision Tree) and archetype
classification on the derived 2017-18 midfielder dataset.

    python rf_dt_analysis.py
Requires data/processed/players_derived.csv (from derive_pappalardo.py).
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

df = pd.read_csv("data/processed/players_derived.csv")
df = df[(df.position == "MF") & (df.minutes >= 900)].reset_index(drop=True)
n90 = df.minutes / 90
COUNT = ["key_passes","smart_passes","assists","passes_into_penalty_area",
         "progressive_passes","progressive_pass_distance","crosses",
         "dribbles_won","accelerations","touches_att_pen_area",
         "tackles_won","interceptions","clearances","aerials_won",
         "shots","shots_on_target","goals"]
for c in COUNT: df[c+"_p90"] = df[c] / n90
RAW = [c+"_p90" for c in COUNT] + ["passes_pct","dribble_success_pct"]

DIMS = {
 "creativity_score":{"key_passes_p90":.30,"smart_passes_p90":.25,"assists_p90":.20,"passes_into_penalty_area_p90":.25},
 "distribution_score":{"passes_pct":.35,"progressive_pass_distance_p90":.35,"progressive_passes_p90":.30},
 "ball_carrying_score":{"dribbles_won_p90":.35,"accelerations_p90":.25,"touches_att_pen_area_p90":.20,"dribble_success_pct":.20},
 "defensive_score":{"tackles_won_p90":.30,"interceptions_p90":.30,"clearances_p90":.20,"aerials_won_p90":.20},
 "goal_threat_score":{"shots_p90":.35,"shots_on_target_p90":.30,"goals_p90":.35},
}
for d, comps in DIMS.items():
    parts = [MinMaxScaler().fit_transform(df[[c]].fillna(0)).flatten()*w for c, w in comps.items()]
    df[d] = np.stack(parts, 1).sum(1)
SS = list(DIMS)
X = StandardScaler().fit_transform(df[SS].values)
df["cluster"] = KMeans(5, random_state=42, n_init=10).fit_predict(X)

Xr, y = df[RAW].fillna(0).values, df["cluster"].values
cv = StratifiedKFold(5, shuffle=True, random_state=42)

print("Archetype sizes:", df.cluster.value_counts().sort_index().to_dict())
print("Majority baseline: %.3f" % (df.cluster.value_counts().max()/len(df)))

rf = RandomForestClassifier(n_estimators=500, random_state=42)
print("RF (19 raw features) 5-fold acc: %.3f" % cross_val_score(rf, Xr, y, cv=cv).mean())
print("RF (5 dimensions)   5-fold acc: %.3f" %
      cross_val_score(RandomForestClassifier(500, random_state=42), df[SS].values, y, cv=cv).mean())
for d in [2,3,4,5,6,None]:
    acc = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), Xr, y, cv=cv).mean()
    print(f"DT depth={str(d):>4} acc: {acc:.3f}")

rf.fit(Xr, y)
dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xr, y)
print("\nRF importances (%):")
print((pd.Series(rf.feature_importances_, index=RAW).sort_values(ascending=False)*100).round(1).head(10).to_string())
print("\nDT importances (%):")
s = pd.Series(dt.feature_importances_, index=RAW)
print((s[s>0].sort_values(ascending=False)*100).round(1).to_string())


# ---- Confusion / boundary analysis (verifies the boundary-error claim) ----
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
yhat = cross_val_predict(RandomForestClassifier(500, random_state=42), Xr, y, cv=cv)
cm = confusion_matrix(y, yhat)
print("\nConfusion matrix (rows=true, cols=pred):\n", pd.DataFrame(cm).to_string())
err = yhat != y
km = KMeans(5, random_state=42, n_init=10).fit(X)
D = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
ds = np.sort(D, axis=1); margin = ds[:, 1] - ds[:, 0]
print("errors: %d/%d (%.1f%%)" % (err.sum(), len(y), 100*err.mean()))
print("median boundary margin  correct: %.3f | errors: %.3f"
      % (np.median(margin[~err]), np.median(margin[err])))
second = np.argsort(D, axis=1)[:, 1]
print("errors predicted as 2nd-nearest archetype: %.1f%%"
      % (100*(yhat[err] == second[err]).mean()))
