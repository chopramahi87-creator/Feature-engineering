# 🏥 HealthWise — Smart Premium Engine (Streamlit dashboard)

Two-stage pricing: **Stage 1** classifies a customer into a Low / Medium / High risk tier, **Stage 2** predicts the annual premium with a regression trained only on that tier.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard (6 tabs) |
| `train_model.py` | Cleans the data, trains the models, writes `healthwise_model.joblib` + `metrics.json` |
| `healthwise_model.joblib` | Saved Stage-1 classifier, three Stage-2 regressors, flat-price baseline |
| `metrics.json` | Numbers shown in the *Model Performance* tab |
| `healthwise.csv` | Your dataset (used by the *Data Explorer* tab and for retraining) |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Theme |

## Run locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py             # optional - re-creates the model files
streamlit run app.py
```
Open http://localhost:8501.

> If `healthwise_model.joblib` was saved with a different scikit-learn version and fails to load, the app **retrains automatically** from `healthwise.csv`. You can also just run `python train_model.py` again.

## Dashboard tabs

1. **Premium Calculator** — change the sidebar inputs; tier, premium, confidence, comparison with the flat model, and a "why this price" breakdown update instantly.
2. **What-if Analysis** — scenario table (switch smoker, +2 children, BMI ±3, …) and single-variable sweep curves that show tier jumps.
3. **Batch Scoring** — upload a CSV (columns: `age, bmi, children, exercise_freq, sex, smoker, region`), get tier + premium for every row, download the results. A template CSV can be downloaded in the tab.
4. **Data Explorer** — filters, distributions, smoker/age/BMI relationships, and the feature audit that justifies the dropped columns.
5. **Model Performance** — accuracy, confusion matrix, flat vs two-stage MAE/R², per-tier fit, bias-vs-variance depth curve, coefficients, feature importance.
6. **About** — method, data preparation, limitations.

## Deploy free on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → *New app* → pick the repo and `app.py`.
3. Deploy. (`requirements.txt` is installed automatically.)

## Run from Google Colab (optional)
```python
!pip install -q streamlit pyngrok plotly
# upload the whole folder, then:
from pyngrok import ngrok
!streamlit run app.py &>/dev/null &
print(ngrok.connect(8501))
```
(needs a free ngrok auth token).

## Key results (held-out test set)
- Stage-1 accuracy ≈ 93.8 %
- Flat model MAE ≈ $6.1k → two-stage MAE ≈ $1.7k (≈ 72 % lower, using the *predicted* tier)
- Top cost drivers: smoking, BMI, age
