"""
HealthWise Insurance - Smart Premium Engine (Streamlit dashboard)

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import train_model as tm

BASE = Path(__file__).parent
TIER_COLORS = {"Low": "#2e9e6b", "Medium": "#e8a317", "High": "#d64545"}
TIERS = tm.TIERS
FEATURES, NUM, CAT = tm.FEATURES, tm.NUM, tm.CAT

st.set_page_config(page_title="HealthWise Smart Premium Engine", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.tier-badge {display:inline-block; padding:.35rem 1rem; border-radius:999px; color:white;
             font-weight:700; font-size:1.3rem; letter-spacing:.5px;}
.big-number {font-size:2.6rem; font-weight:800; line-height:1.1;}
.muted {color:#8a8f98; font-size:.9rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: .8rem 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading models...")
def load_all():
    csv = BASE / "healthwise.csv"
    df = tm.load_clean(csv) if csv.exists() else None
    bundle = metrics = None
    try:                                              # use the saved model if it loads
        bundle = joblib.load(BASE / "healthwise_model.joblib")
        with open(BASE / "metrics.json") as f:
            metrics = json.load(f)
    except Exception:
        bundle = metrics = None
    if (bundle is None or metrics is None) and df is not None:   # otherwise retrain automatically
        bundle, metrics = tm.train_all(df)
    return bundle, metrics, df


bundle, metrics, data = load_all()
if bundle is None:
    st.error("Could not load `healthwise_model.joblib` and `healthwise.csv` was not found to retrain. "
             "Put both files next to app.py (or run `python train_model.py`).")
    st.stop()

clf, regs, glob, MED = bundle["clf"], bundle["regs"], bundle.get("global"), bundle["med"]


# ------------------------------------------------------------------ prediction helpers
def predict_df(frame: pd.DataFrame) -> pd.DataFrame:
    """Two-stage prediction for many customers at once."""
    X = frame.copy()
    for c in NUM:
        if c not in X:
            X[c] = MED[c]
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(MED[c])
    for c in ["sex", "smoker", "region"]:
        X[c] = X[c].astype(str).str.strip().str.lower()
    tiers = clf.predict(X[FEATURES])
    proba = pd.DataFrame(clf.predict_proba(X[FEATURES]), columns=clf.classes_, index=X.index)
    prem = np.zeros(len(X))
    for t in TIERS:
        m = tiers == t
        if m.any():
            prem[m] = regs[t].predict(X.loc[m, FEATURES])
    out = pd.DataFrame({"predicted_tier": tiers, "predicted_premium": np.round(np.maximum(prem, 0), 2)},
                       index=X.index)
    for t in TIERS:
        out[f"P({t})"] = proba[t].round(3) if t in proba else 0.0
    if glob is not None:
        out["flat_model_premium"] = np.round(np.maximum(glob.predict(X[FEATURES]), 0), 2)
    return out


def predict_one(c: dict):
    r = predict_df(pd.DataFrame([c])).iloc[0]
    return r


def fmt(x):
    return f"${x:,.0f}"


# ------------------------------------------------------------------ sidebar: customer profile
st.sidebar.title("🏥 HealthWise")
st.sidebar.caption("Customer profile - everything updates live")
age = st.sidebar.slider("Age", 18, 64, 40)
bmi = st.sidebar.slider("BMI", 16.0, 45.0, 27.0, 0.1)
children = st.sidebar.slider("Children", 0, 4, 1)
exercise = st.sidebar.slider("Exercise (workouts / week)", 0, 7, 3)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["north", "south", "east", "west"])
customer = dict(age=age, bmi=bmi, children=children, exercise_freq=exercise,
                sex=sex, smoker=smoker, region=region)
res = predict_one(customer)
tier, premium = res["predicted_tier"], float(res["predicted_premium"])

st.title("Smart Premium Engine")
st.caption("Two-stage pricing: **Stage 1** classifies the risk tier, **Stage 2** prices the customer "
           "with a regression trained only on that tier.")

tab_calc, tab_what, tab_batch, tab_data, tab_perf, tab_about = st.tabs(
    ["💰 Premium Calculator", "🔀 What-if Analysis", "📂 Batch Scoring",
     "📊 Data Explorer", "🎯 Model Performance", "ℹ️ About"])

# ================================================================== TAB 1: calculator
with tab_calc:
    c1, c2, c3 = st.columns([1.1, 1.4, 1.3])
    with c1:
        st.markdown("##### Predicted risk tier")
        st.markdown(f'<span class="tier-badge" style="background:{TIER_COLORS[tier]}">{tier.upper()}</span>',
                    unsafe_allow_html=True)
        st.markdown("&nbsp;")
        st.markdown("##### Estimated annual premium")
        st.markdown(f'<div class="big-number">{fmt(premium)}</div>'
                    f'<div class="muted">≈ {fmt(premium/12)} per month</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("##### Stage-1 confidence")
        pr = pd.DataFrame({"tier": TIERS, "probability": [float(res[f"P({t})"]) for t in TIERS]})
        fig = px.bar(pr, x="probability", y="tier", orientation="h", color="tier",
                     color_discrete_map=TIER_COLORS, range_x=[0, 1], text=pr["probability"].map("{:.0%}".format))
        fig.update_layout(showlegend=False, height=210, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(categoryorder="array", categoryarray=TIERS[::-1]))
        st.plotly_chart(fig)
        if pr["probability"].max() < 0.7:
            st.warning("Borderline customer - the classifier is not very sure. Consider manual underwriting.")
    with c3:
        st.markdown("##### Compared with…")
        if data is not None:
            avg_t = data.loc[data.risk_tier == tier, "annual_charge"].mean()
            lo, hi = data.loc[data.risk_tier == tier, "annual_charge"].agg(["min", "max"])
            st.metric(f"Average {tier}-tier customer", fmt(avg_t), f"{fmt(premium - avg_t)} vs this customer",
                      delta_color="off")
            st.caption(f"Historical {tier}-tier range: {fmt(lo)} - {fmt(hi)}")
        if "flat_model_premium" in res:
            flat = float(res["flat_model_premium"])
            st.metric("Old flat-pricing model", fmt(flat), f"{fmt(premium - flat)} two-stage vs flat",
                      delta_color="off")

    st.divider()
    st.markdown(f"##### Why this price? Difference from the average **{tier}** customer")
    if data is not None:
        pipe = regs[tier]
        names = list(pipe.named_steps["pre"].get_feature_names_out())
        coef = pipe.named_steps["model"].coef_
        Xt = pipe.named_steps["pre"].transform(pd.DataFrame([customer])[FEATURES])
        ref = pipe.named_steps["pre"].transform(data.loc[data.risk_tier == tier, FEATURES]).mean(axis=0)
        contrib = coef * (np.asarray(Xt).ravel() - np.asarray(ref).ravel())
        label = {"num__age": "Age", "num__bmi": "BMI", "num__children": "Children",
                 "num__exercise_freq": "Exercise", "cat__sex_male": "Sex (male)",
                 "cat__smoker_yes": "Smoker", "cat__region_north": "Region: north",
                 "cat__region_south": "Region: south", "cat__region_west": "Region: west"}
        cdf = pd.DataFrame({"factor": [label.get(n, n) for n in names], "effect": contrib})
        cdf = cdf[cdf["effect"].abs() > 0.5].sort_values("effect")
        if len(cdf):
            fig = px.bar(cdf, x="effect", y="factor", orientation="h",
                         color=cdf["effect"] > 0, color_discrete_map={True: "#d64545", False: "#2e9e6b"},
                         labels={"effect": "$ vs average customer in this tier", "factor": ""})
            fig.update_layout(showlegend=False, height=max(220, 40 * len(cdf)), margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig)
        else:
            st.info("This customer looks like the average customer in their tier.")

# ================================================================== TAB 2: what-if
with tab_what:
    st.markdown("Change the profile in the sidebar, then see how the premium reacts.")
    st.markdown("##### Quick scenarios")
    scen = {
        "Current customer": customer,
        "Switch smoker status": {**customer, "smoker": "no" if smoker == "yes" else "yes"},
        "+2 children": {**customer, "children": min(customer["children"] + 2, 4)},
        "BMI −3": {**customer, "bmi": max(16.0, bmi - 3)},
        "BMI +3": {**customer, "bmi": min(45.0, bmi + 3)},
        "Exercise +2 / week": {**customer, "exercise_freq": min(7, exercise + 2)},
        "10 years older": {**customer, "age": min(64, age + 10)},
    }
    sdf = pd.DataFrame(scen).T
    out = predict_df(sdf)
    tbl = pd.DataFrame({"Scenario": sdf.index, "Tier": out["predicted_tier"].values,
                        "Premium": out["predicted_premium"].values})
    tbl["Change vs current"] = tbl["Premium"] - premium
    st.dataframe(tbl.style.format({"Premium": "${:,.0f}", "Change vs current": "{:+,.0f}"}),
                 hide_index=True)
    fig = px.bar(tbl[tbl.Scenario != "Current customer"], x="Scenario", y="Change vs current", color="Tier",
                 color_discrete_map=TIER_COLORS, title="Premium change vs current customer ($)")
    fig.update_layout(height=340, margin=dict(t=40, b=0))
    st.plotly_chart(fig)

    st.markdown("##### Sweep one variable")
    var = st.selectbox("Variable to sweep", ["age", "bmi", "children", "exercise_freq"])
    rng = {"age": np.arange(18, 65), "bmi": np.round(np.arange(16, 45.1, 0.5), 1),
           "children": np.arange(0, 5), "exercise_freq": np.arange(0, 8)}[var]
    sw = pd.DataFrame([{**customer, var: v} for v in rng])
    so = predict_df(sw)
    sw_res = pd.DataFrame({var: rng, "premium": so["predicted_premium"].values, "tier": so["predicted_tier"].values})
    fig = px.scatter(sw_res, x=var, y="premium", color="tier", color_discrete_map=TIER_COLORS,
                     category_orders={"tier": TIERS})
    fig.add_vline(x=customer[var], line_dash="dash", line_color="gray")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(height=380, margin=dict(t=10), yaxis_title="Estimated annual premium ($)")
    st.plotly_chart(fig)
    st.caption("Jumps in the line are tier changes: the customer moves to a different tier's regression model.")

# ================================================================== TAB 3: batch
with tab_batch:
    st.markdown("Upload a CSV of customers to score them all at once.")
    st.markdown("**Required columns:** `age, bmi, children, exercise_freq, sex, smoker, region`")
    template = pd.DataFrame([customer, {**customer, "smoker": "yes", "age": 55, "bmi": 33.0}])
    st.download_button("⬇️ Download template CSV", template.to_csv(index=False), "template_customers.csv", "text/csv")
    up = st.file_uploader("Customers CSV", type="csv")
    if up is not None:
        new = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in new.columns and c not in NUM]
        if missing:
            st.error(f"Missing required column(s): {missing}")
        else:
            scored = pd.concat([new, predict_df(new)], axis=1)
            k1, k2, k3 = st.columns(3)
            k1.metric("Customers scored", len(scored))
            k2.metric("Average premium", fmt(scored["predicted_premium"].mean()))
            k3.metric("High-risk share", f"{(scored['predicted_tier'] == 'High').mean():.0%}")
            st.dataframe(scored)
            cA, cB = st.columns(2)
            cA.plotly_chart(px.histogram(scored, x="predicted_premium", color="predicted_tier",
                                         color_discrete_map=TIER_COLORS, title="Premium distribution"))
            cnt = scored["predicted_tier"].value_counts().reindex(TIERS).fillna(0).reset_index()
            cnt.columns = ["tier", "customers"]
            cB.plotly_chart(px.bar(cnt, x="tier", y="customers", color="tier", color_discrete_map=TIER_COLORS,
                                   title="Customers per tier"))
            st.download_button("⬇️ Download predictions", scored.to_csv(index=False), "predictions.csv", "text/csv")

# ================================================================== TAB 4: data explorer
with tab_data:
    if data is None:
        st.info("Place `healthwise.csv` next to app.py to enable the Data Explorer.")
    else:
        f1, f2, f3 = st.columns(3)
        ft = f1.multiselect("Risk tier", TIERS, default=TIERS)
        fs = f2.multiselect("Smoker", ["yes", "no"], default=["yes", "no"])
        fr = f3.multiselect("Region", ["north", "south", "east", "west"], default=["north", "south", "east", "west"])
        d = data[data.risk_tier.isin(ft) & data.smoker.isin(fs) & data.region.isin(fr)]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers", f"{len(d):,}")
        k2.metric("Avg annual charge", fmt(d.annual_charge.mean()) if len(d) else "-")
        k3.metric("Smokers", f"{(d.smoker == 'yes').mean():.0%}" if len(d) else "-")
        k4.metric("Avg age", f"{d.age.mean():.1f}" if len(d) else "-")

        a, b = st.columns(2)
        a.plotly_chart(px.histogram(d, x="annual_charge", nbins=40, color="risk_tier",
                                    color_discrete_map=TIER_COLORS, title="Annual charge distribution "
                                    f"(skew = {data.annual_charge.skew():.2f})"))
        b.plotly_chart(px.box(d, x="smoker", y="annual_charge", color="smoker",
                              title="Charge by smoker status"))
        a, b = st.columns(2)
        a.plotly_chart(px.scatter(d, x="age", y="annual_charge", color="smoker", opacity=.6,
                                  title="Age vs charge"))
        b.plotly_chart(px.scatter(d, x="bmi", y="annual_charge", color="smoker", opacity=.6,
                                  title="BMI vs charge"))

        st.markdown("##### Feature audit - which columns actually matter?")
        raw = pd.read_csv(BASE / "healthwise.csv").drop_duplicates()
        num_c = ["age", "bmi", "weight_kg", "children", "exercise_freq", "lucky_number"]
        corr = raw[num_c + ["annual_charge"]].corr()["annual_charge"].drop("annual_charge")
        cat_rows = []
        for c in ["smoker", "sex", "region", "favorite_color", "zodiac_sign", "preferred_contact", "marketing_opt_in"]:
            g = raw.groupby(c)["annual_charge"].mean()
            cat_rows.append((c, (g.max() - g.min()) / raw.annual_charge.mean() * 100))
        a, b = st.columns(2)
        cf = corr.reset_index(); cf.columns = ["feature", "correlation"]
        a.plotly_chart(px.bar(cf, x="correlation", y="feature", orientation="h",
                              title="Correlation with annual charge"))
        cg = pd.DataFrame(cat_rows, columns=["feature", "spread %"])
        b.plotly_chart(px.bar(cg, x="spread %", y="feature", orientation="h",
                              title="Group-mean spread of charge (%)"))
        st.caption("`lucky_number`, `favorite_color`, `zodiac_sign`, `preferred_contact`, `marketing_opt_in` and "
                   "`customer_id` were dropped as irrelevant; `weight_kg` (≈ BMI) and `date_of_birth` (= age) as "
                   "redundant. Zodiac's spread comes from 12 small groups - chance, not signal.")
        with st.expander("Show cleaned data"):
            st.dataframe(d)

# ================================================================== TAB 5: performance
with tab_perf:
    mc = metrics["model_compare"]
    g = mc["Global linear (flat price)"]; ts = mc["Two-stage (predicted tier)"]
    st.caption(f"Measured on a held-out test set of {metrics['n_test']} customers "
               f"(train {metrics['n_train']}). Deployed models are refit on all {metrics['n_rows']} customers.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stage-1 accuracy", f"{metrics['accuracy']:.1%}")
    k2.metric("Flat model MAE", fmt(g["MAE"]))
    k3.metric("Two-stage MAE", fmt(ts["MAE"]), f"-{metrics['mae_improvement_pct']:.0f}%", delta_color="inverse")
    k4.metric("Two-stage R²", f"{ts['R2']:.3f}", f"vs {g['R2']:.3f} flat", delta_color="off")

    st.markdown("##### Board question 1 - does two-stage beat one flat model?")
    cmp_df = pd.DataFrame(mc).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(cmp_df.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.3f}"}),
                 hide_index=True)
    st.info("The *true-tier* row is optimistic (the tier already reveals the charge range). "
            "The *predicted-tier* row is the realistic deployment number.")

    a, b = st.columns(2)
    with a:
        cm = np.array(metrics["confusion_matrix"])
        fig = px.imshow(cm, x=TIERS, y=TIERS, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True"), title="Stage-1 confusion matrix")
        fig.update_layout(height=380)
        st.plotly_chart(fig)
    with b:
        rep = pd.DataFrame(metrics["classification_report"]).T.loc[TIERS, ["precision", "recall", "f1-score", "support"]]
        st.markdown("**Per-tier classification quality**")
        st.dataframe(rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))
        pt = pd.DataFrame(metrics["per_tier"]).T.reset_index().rename(columns={"index": "Tier"})
        st.markdown("**Stage-2 regression quality per tier**")
        st.dataframe(pt.style.format({"MAE": "${:,.0f}", "R2": "{:.3f}", "n_test": "{:.0f}"}),
                     hide_index=True)

    st.markdown("##### Bias vs variance - choosing the tree depth")
    sw = pd.DataFrame(metrics["depth_sweep"])
    fig = go.Figure()
    fig.add_scatter(x=sw.depth, y=sw.train_acc, mode="lines+markers", name="Train accuracy")
    fig.add_scatter(x=sw.depth, y=sw.cv_acc, mode="lines+markers", name="5-fold CV accuracy")
    fig.add_vline(x=tm.TREE_DEPTH, line_dash="dash", line_color="gray", annotation_text=f"deployed depth {tm.TREE_DEPTH}")
    fig.update_layout(height=350, xaxis_title="Tree depth (complexity)", yaxis_title="Accuracy", margin=dict(t=10))
    st.plotly_chart(fig)
    st.caption("Low depth = high bias (underfit). High depth = train ≫ CV = high variance (overfit).")

    a, b = st.columns(2)
    with a:
        tsel = st.selectbox("Tier for coefficient view", TIERS, index=2)
        cc = pd.Series(metrics["tier_coefficients"][tsel]).rename_axis("feature").reset_index(name="$ per unit")
        st.plotly_chart(px.bar(cc.sort_values("$ per unit"), x="$ per unit", y="feature", orientation="h",
                               title=f"What drives cost in the {tsel} tier ($ per unit)"))
    with b:
        fi = pd.Series(metrics["feature_importance"]).rename_axis("feature").reset_index(name="importance")
        st.plotly_chart(px.bar(fi.sort_values("importance"), x="importance", y="feature", orientation="h",
                               title="Stage-1 classifier: feature importance"))

    cc_df = pd.DataFrame({"Classifier": list(metrics["clf_compare"]), "Test accuracy": list(metrics["clf_compare"].values())})
    st.markdown("##### Classifier comparison")
    st.dataframe(cc_df.style.format({"Test accuracy": "{:.1%}"}), hide_index=True)

# ================================================================== TAB 6: about
with tab_about:
    st.markdown(f"""
### How the engine works
1. **Stage 1 - Classify.** A decision tree (depth {tm.TREE_DEPTH}) places the customer in a **Low / Medium / High** risk tier.
2. **Stage 2 - Predict.** A linear regression trained **only on that tier's customers** estimates the annual charge.
3. Change the smoker flag and the customer may jump to another tier → a different regression → a different premium.

### Data preparation
- Removed 3 duplicate rows; median-imputed 18 missing `bmi` and 18 missing `exercise_freq` values.
- Dropped irrelevant columns (`customer_id`, `favorite_color`, `zodiac_sign`, `lucky_number`, `preferred_contact`,
  `marketing_opt_in`) and redundant ones (`date_of_birth` = age, `weight_kg` ≈ BMI, VIF ≈ 38).

### Top cost drivers
**Smoking status**, **BMI** and **age**; children and sex matter a little; region and exercise add little.

### Limitations
- Risk tiers in the data line up with charge ranges, so real-world accuracy should be monitored and the models retrained regularly.
- A truly High-risk customer classified as Low would be under-priced - send borderline cases (low confidence) to manual review.
- Trained on ~1,200 customers; predictions are estimates, not quotes.
""")
