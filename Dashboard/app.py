import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(page_title="NeuralRetail", layout="wide", page_icon="🛒")

# Download data directly from UCI repository
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    st.info("Loading data... please wait 30 seconds")
    df = pd.read_excel(url, sheet_name="Year 2009-2010", engine="openpyxl")
    df = df.dropna(subset=["Customer ID"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df

@st.cache_data
def get_rfm(df):
    snapshot_date = df["InvoiceDate"].max()
    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    return rfm

@st.cache_data
def get_daily(df):
    daily = df.groupby(df["InvoiceDate"].dt.date)["Revenue"].sum().reset_index()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    return daily

# Sidebar
st.sidebar.title("🛒 NeuralRetail")
st.sidebar.markdown("AI Sales Intelligence")
page = st.sidebar.selectbox("Navigate", [
    "📊 Executive KPI",
    "🔮 Demand Forecast",
    "⚠️ Churn Analysis",
    "👥 Customer Segments",
    "📦 Inventory Health"
])

# Load data
df = load_data()
rfm = get_rfm(df)
daily = get_daily(df)

# ── Page 1 ──────────────────────
if page == "📊 Executive KPI":
    st.title("📊 Executive KPI Dashboard")
    st.markdown("### NeuralRetail – AI Sales Intelligence")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"£{df['Revenue'].sum():,.0f}")
    col2.metric("👥 Customers", f"{df['Customer ID'].nunique():,}")
    col3.metric("📦 Orders", f"{df['Invoice'].nunique():,}")
    col4.metric("💳 Avg Order", f"£{df.groupby('Invoice')['Revenue'].sum().mean():,.0f}")

    st.markdown("---")
    fig = px.line(daily, x="ds", y="y",
                  title="Daily Revenue Trend",
                  color_discrete_sequence=["#E84E1B"])
    fig.update_layout(xaxis_title="Date", yaxis_title="Revenue (£)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_countries = df.groupby("Country")["Revenue"].sum().nlargest(10).reset_index()
        fig2 = px.bar(top_countries, x="Revenue", y="Country",
                      orientation="h",
                      title="Top 10 Countries by Revenue",
                      color_discrete_sequence=["#F7941D"])
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        monthly = df.groupby(
            df["InvoiceDate"].dt.to_period("M").astype(str)
        )["Revenue"].sum().reset_index()
        fig3 = px.bar(monthly, x="InvoiceDate", y="Revenue",
                      title="Monthly Revenue",
                      color_discrete_sequence=["#FBBA13"])
        st.plotly_chart(fig3, use_container_width=True)

# ── Page 2 ──────────────────────
elif page == "🔮 Demand Forecast":
    st.title("🔮 Demand Forecast")
    st.markdown("### 30-Day Revenue Forecast using Prophet")

    try:
        from prophet import Prophet
        from sklearn.metrics import mean_absolute_percentage_error

        train = daily[:-30]
        test = daily[-30:]

        with st.spinner("Training forecast model..."):
            model = Prophet(seasonality_mode="multiplicative",
                           changepoint_prior_scale=0.05)
            model.fit(train)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

        mape = mean_absolute_percentage_error(
            test["y"].values,
            forecast[-30:]["yhat"].values
        )
        st.metric("MAPE", f"{mape*100:.2f}%")

        fig = px.line(forecast, x="ds", y=["yhat","yhat_lower","yhat_upper"],
                      title="30-Day Forecast",
                      color_discrete_sequence=["#E84E1B","#F7941D","#FBBA13"])
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Forecast error: {e}")
        fig = px.line(daily, x="ds", y="y", title="Historical Revenue")
        st.plotly_chart(fig, use_container_width=True)

# ── Page 3 ──────────────────────
elif page == "⚠️ Churn Analysis":
    st.title("⚠️ Churn Risk Analysis")

    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        X = rfm[["Recency","Frequency","Monetary"]]
        y = rfm["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        with st.spinner("Training churn model..."):
            model = xgb.XGBClassifier(n_estimators=100, verbosity=0)
            model.fit(X_train, y_train)
            rfm["Churn_Prob"] = model.predict_proba(X)[:,1]

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{auc:.4f}")
        col2.metric("🔴 High Risk", len(rfm[rfm["Churn_Prob"]>0.7]))
        col3.metric("🟡 Medium Risk", len(rfm[(rfm["Churn_Prob"]>0.4)&(rfm["Churn_Prob"]<=0.7)]))
        col4.metric("🟢 Low Risk", len(rfm[rfm["Churn_Prob"]<=0.4]))

        fig = px.histogram(rfm, x="Churn_Prob",
                           title="Churn Probability Distribution",
                           color_discrete_sequence=["#E84E1B"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### High Risk Customers")
        st.dataframe(rfm[rfm["Churn_Prob"]>0.7].sort_values(
            "Churn_Prob", ascending=False).head(20))

    except Exception as e:
        st.error(f"Churn model error: {e}")

# ── Page 4 ──────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    with st.spinner("Finding best segments..."):
        best_score = 0
        best_k = 3
        for k in range(3, 7):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k

        final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        rfm["Segment"] = final.fit_predict(X_scaled)

    st.metric("Best Segments", best_k)
    st.metric("Silhouette Score", f"{best_score:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(rfm, x="Recency", y="Monetary",
                         color="Segment",
                         title=f"Customer Segments (k={best_k})")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        fig2 = px.pie(seg_counts, values="count", names="Segment",
                      title="Customers per Segment")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Segment Summary")
    st.dataframe(rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(2))

# ── Page 5 ──────────────────────
elif page == "📦 Inventory Health":
    st.title("📦 Inventory Health")

    top_products = df.groupby("Description")["Quantity"].sum().nlargest(10).reset_index()
    fig = px.bar(top_products, x="Quantity", y="Description",
                 orientation="h",
                 title="Top 10 Products by Volume",
                 color_discrete_sequence=["#E84E1B"])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_rev = df.groupby("Description")["Revenue"].sum().nlargest(10).reset_index()
        fig2 = px.bar(top_rev, x="Revenue", y="Description",
                      orientation="h",
                      title="Top 10 Products by Revenue",
                      color_discrete_sequence=["#F7941D"])
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        monthly_qty = df.groupby(
            df["InvoiceDate"].dt.to_period("M").astype(str)
        )["Quantity"].sum().reset_index()
        fig3 = px.line(monthly_qty, x="InvoiceDate", y="Quantity",
                       title="Monthly Quantity Trend",
                       color_discrete_sequence=["#FBBA13"])
        st.plotly_chart(fig3, use_container_width=True)
