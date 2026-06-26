import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import pickle
import os

PROCESSED = r"C:\Users\Harsha Koshti\OneDrive\Desktop\NeuralRetail\data\processed"
ROOT = r"C:\Users\Harsha Koshti\OneDrive\Desktop\NeuralRetail"

print("="*50)
print("STEP 1 - Loading Clean Data (fast)")
print("="*50)

# Use already cleaned CSV - much faster than xlsx!
df = pd.read_csv(os.path.join(PROCESSED, "clean_retail.csv"))
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print(f"Loaded: {df.shape} ✅")

print("\n" + "="*50)
print("STEP 2 - Feature Engineering")
print("="*50)

snapshot_date = df["InvoiceDate"].max()
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()
rfm.to_csv(os.path.join(PROCESSED, "rfm.csv"), index=False)

daily = df.groupby(df["InvoiceDate"].dt.date)["Revenue"].sum().reset_index()
daily.columns = ["ds", "y"]
daily["ds"] = pd.to_datetime(daily["ds"])
daily.to_csv(os.path.join(PROCESSED, "daily_sales.csv"), index=False)
print(f"RFM: {rfm.shape} ✅")
print(f"Daily: {daily.shape} ✅")

print("\n" + "="*50)
print("STEP 3 - Demand Forecasting")
print("="*50)

train = daily[:-30]
test = daily[-30:]
model_prophet = Prophet(
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05
)
model_prophet.fit(train)
future = model_prophet.make_future_dataframe(periods=30)
forecast = model_prophet.predict(future)
mape = mean_absolute_percentage_error(
    test["y"].values,
    forecast[-30:]["yhat"].values
)
print(f"MAPE: {mape*100:.2f}% ✅")
fig = model_prophet.plot(forecast)
plt.title("30-Day Forecast")
plt.savefig(os.path.join(PROCESSED, "forecast_plot.png"), bbox_inches="tight")
plt.close()
forecast.to_csv(os.path.join(PROCESSED, "forecast_output.csv"), index=False)
print("Forecast saved ✅")

print("\n" + "="*50)
print("STEP 4 - Churn Prediction")
print("="*50)

rfm["Churn"] = (rfm["Recency"] > 90).astype(int)
X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42,
    verbosity=0
)
model_xgb.fit(X_train, y_train)
auc = roc_auc_score(y_test, model_xgb.predict_proba(X_test)[:,1])
print(f"AUC-ROC: {auc:.4f} ✅")

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(PROCESSED, "shap_summary.png"), bbox_inches="tight")
plt.close()

rfm["Churn_Probability"] = model_xgb.predict_proba(X)[:,1]
rfm.to_csv(os.path.join(PROCESSED, "rfm_churn_scores.csv"), index=False)
pickle.dump(model_xgb, open(os.path.join(ROOT, "churn_model.pkl"), "wb"))
print("Churn saved ✅")

print("\n" + "="*50)
print("STEP 5 - Segmentation")
print("="*50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])
best_score = 0
best_k = 3
for k in range(3, 7):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, Score={score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm["Segment"] = final_km.fit_predict(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(rfm["Recency"], rfm["Monetary"],
            c=rfm["Segment"], cmap="tab10", alpha=0.5)
plt.title(f"Segments k={best_k}")
plt.savefig(os.path.join(PROCESSED, "segments_plot.png"), bbox_inches="tight")
plt.close()
rfm.to_csv(os.path.join(PROCESSED, "rfm_segments.csv"), index=False)
print(f"Best k={best_k} ✅")

print("\n" + "="*50)
print("ALL DONE! 🎉")
print("="*50)
print(f"MAPE: {mape*100:.2f}%")
print(f"AUC-ROC: {auc:.4f}")
print(f"Segments: {best_k}")