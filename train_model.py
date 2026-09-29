"""
HealthWise - training module.

Run once (or whenever the data changes):
    python train_model.py                # uses healthwise.csv next to this file
    python train_model.py path/to/data.csv

Creates:
    healthwise_model.joblib   (Stage-1 classifier, Stage-2 per-tier regressors, global baseline)
    metrics.json              (numbers shown on the dashboard's "Model Performance" tab)
app.py also imports this file and retrains automatically if the saved model can't be loaded.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TIERS = ["Low", "Medium", "High"]
NUM = ["age", "bmi", "children", "exercise_freq"]
CAT = ["sex", "smoker", "region"]
FEATURES = NUM + CAT
TREE_DEPTH = 5
BASE = Path(__file__).parent


# ------------------------------------------------------------------ data
def load_clean(path=None) -> pd.DataFrame:
    """Load healthwise.csv and clean it exactly like the lab notebook."""
    df = pd.read_csv(path or BASE / "healthwise.csv")
    df = df.drop_duplicates().copy()
    for c in ["sex", "smoker", "region", "preferred_contact", "marketing_opt_in"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip().str.lower()
    df["risk_tier"] = df["risk_tier"].astype(str).str.strip().str.title()
    for c in ["bmi", "exercise_freq"]:
        df[c] = df[c].fillna(df[c].median())
    return df.reset_index(drop=True)


def make_pre():
    return ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
    ])


def _clf():
    return Pipeline([("pre", make_pre()),
                     ("model", DecisionTreeClassifier(max_depth=TREE_DEPTH, random_state=RANDOM_STATE))])


def _reg():
    return Pipeline([("pre", make_pre()), ("model", LinearRegression())])


def _fit_stage2(train_df):
    return {t: _reg().fit(train_df.loc[train_df.risk_tier == t, FEATURES],
                          train_df.loc[train_df.risk_tier == t, "annual_charge"]) for t in TIERS}


def _two_stage_predict(tiers, regs, X):
    out = np.zeros(len(X))
    tiers = np.asarray(tiers)
    for t in TIERS:
        m = tiers == t
        if m.any():
            out[m] = regs[t].predict(X.loc[m, FEATURES])
    return np.maximum(out, 0)


# ------------------------------------------------------------------ training
def train_all(df: pd.DataFrame):
    """Returns (bundle, metrics). bundle = final models fitted on ALL data (what the app uses).
    metrics = honest numbers measured on a held-out 20 % test set."""
    # ---------- held-out evaluation ----------
    tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["risk_tier"])
    clf_h = _clf().fit(tr[FEATURES], tr["risk_tier"])
    regs_h = _fit_stage2(tr)
    glob_h = _reg().fit(tr[FEATURES], tr["annual_charge"])

    y = te["annual_charge"].values
    pred_tier = clf_h.predict(te[FEATURES])
    p_glob = np.maximum(glob_h.predict(te[FEATURES]), 0)
    p_true = _two_stage_predict(te["risk_tier"].values, regs_h, te)
    p_pred = _two_stage_predict(pred_tier, regs_h, te)

    def scores(p):
        return {"MAE": float(mean_absolute_error(y, p)),
                "RMSE": float(np.sqrt(mean_squared_error(y, p))),
                "R2": float(r2_score(y, p))}

    model_compare = {"Global linear (flat price)": scores(p_glob),
                     "Two-stage (true tier)": scores(p_true),
                     "Two-stage (predicted tier)": scores(p_pred)}

    per_tier = {}
    for t in TIERS:
        m = te["risk_tier"].values == t
        per_tier[t] = {"n_test": int(m.sum()),
                       "MAE": float(mean_absolute_error(y[m], p_true[m])),
                       "R2": float(r2_score(y[m], p_true[m]))}

    cm = confusion_matrix(te["risk_tier"], pred_tier, labels=TIERS)
    rep = classification_report(te["risk_tier"], pred_tier, labels=TIERS, output_dict=True)

    # classifier comparison
    lr = Pipeline([("pre", make_pre()), ("sc", StandardScaler(with_mean=False)),
                   ("model", LogisticRegression(max_iter=3000))]).fit(tr[FEATURES], tr["risk_tier"])
    rf = Pipeline([("pre", make_pre()),
                   ("model", RandomForestClassifier(200, random_state=RANDOM_STATE))]).fit(tr[FEATURES], tr["risk_tier"])
    clf_compare = {
        f"Decision Tree (depth {TREE_DEPTH})": float(accuracy_score(te["risk_tier"], pred_tier)),
        "Logistic Regression": float(accuracy_score(te["risk_tier"], lr.predict(te[FEATURES]))),
        "Random Forest": float(accuracy_score(te["risk_tier"], rf.predict(te[FEATURES]))),
    }

    # bias-variance style depth sweep (5-fold CV)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    sweep = []
    for d in range(1, 13):
        cv = cross_validate(Pipeline([("pre", make_pre()),
                                      ("model", DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE))]),
                            df[FEATURES], df["risk_tier"], cv=skf, scoring="accuracy", return_train_score=True)
        sweep.append({"depth": d, "train_acc": float(cv["train_score"].mean()),
                      "cv_acc": float(cv["test_score"].mean())})

    # ---------- final models on ALL data (deployed) ----------
    clf = _clf().fit(df[FEATURES], df["risk_tier"])
    regs = _fit_stage2(df)
    glob = _reg().fit(df[FEATURES], df["annual_charge"])

    names = list(clf.named_steps["pre"].get_feature_names_out())
    importances = dict(zip(names, map(float, clf.named_steps["model"].feature_importances_)))
    coefs = {}
    for t in TIERS:
        rn = list(regs[t].named_steps["pre"].get_feature_names_out())
        coefs[t] = dict(zip(rn, map(float, regs[t].named_steps["model"].coef_)))

    metrics = {
        "n_rows": int(len(df)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "accuracy": float(accuracy_score(te["risk_tier"], pred_tier)),
        "confusion_matrix": cm.tolist(), "labels": TIERS,
        "classification_report": {k: {m: float(v) for m, v in rep[k].items()} for k in TIERS},
        "model_compare": model_compare, "per_tier": per_tier,
        "clf_compare": clf_compare, "depth_sweep": sweep,
        "feature_importance": {k: v for k, v in sorted(importances.items(), key=lambda kv: -kv[1])},
        "tier_coefficients": coefs,
        "mae_improvement_pct": float(100 * (model_compare["Global linear (flat price)"]["MAE"]
                                             - model_compare["Two-stage (predicted tier)"]["MAE"])
                                     / model_compare["Global linear (flat price)"]["MAE"]),
    }
    bundle = {"clf": clf, "regs": regs, "global": glob, "features": FEATURES, "num": NUM, "cat": CAT,
              "med": {c: float(df[c].median()) for c in NUM}}
    return bundle, metrics


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "healthwise.csv"
    df = load_clean(csv)
    bundle, metrics = train_all(df)
    joblib.dump(bundle, BASE / "healthwise_model.joblib")
    with open(BASE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mc = metrics["model_compare"]
    print(f"Rows used            : {metrics['n_rows']}")
    print(f"Stage-1 accuracy     : {metrics['accuracy']:.3f}")
    print(f"Global MAE           : ${mc['Global linear (flat price)']['MAE']:,.0f}")
    print(f"Two-stage MAE (pred) : ${mc['Two-stage (predicted tier)']['MAE']:,.0f}"
          f"  ({metrics['mae_improvement_pct']:.0f}% better)")
    print("Saved healthwise_model.joblib and metrics.json")


if __name__ == "__main__":
    main()
