import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="NeuralRetail", layout="wide", page_icon="🛒")

PROCESSED = r"C:\Users\Harsha Koshti\OneDrive\Desktop\NeuralRetail\data\processed"

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED + r"\clean_retail.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

@st.cache_data
def load_rfm():
    return pd.read_csv(PROCESSED + r"\rfm_segments.csv")

@st.cache_data
def load_forecast():
    return pd.read_csv(PROCESSED + r"\forecast_output.csv")

@st.cache_data
def load_churn():
    return pd.read_csv(PROCESSED + r"\rfm_churn_scores.csv")

# Sidebar
st.sidebar.title("🛒 NeuralRetail")
page = st.sidebar.selectbox("Navigate", [
    "📊 Executive KPI",
    "🔮 Demand Forecast",
    "⚠️ Churn Analysis",
    "👥 Customer Segments",
    "📦 Inventory Health"
])

# ── Page 1 ──────────────────────────────────────────
if page == "📊 Executive KPI":
    st.title("📊 Executive KPI Dashboard")
    st.markdown("### NeuralRetail – AI Sales Intelligence")

    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue",
                f"£{df['Revenue'].sum():,.0f}")
    col2.metric("👥 Total Customers",
                f"{df['Customer ID'].nunique():,}")
    col3.metric("📦 Total Orders",
                f"{df['Invoice'].nunique():,}")
    col4.metric("🛍️ Avg Order Value",
                f"£{df.groupby('Invoice')['Revenue'].sum().mean():,.0f}")

    st.markdown("---")

    daily = df.groupby(df["InvoiceDate"].dt.date)["Revenue"].sum().reset_index()
    fig = px.line(daily, x="InvoiceDate", y="Revenue",
                  title="Daily Revenue Trend",
                  color_discrete_sequence=["#E84E1B"])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_countries = df.groupby("Country")["Revenue"].sum().nlargest(10).reset_index()
        fig2 = px.bar(top_countries, x="Revenue", y="Country",
                      orientation="h", title="Top 10 Countries by Revenue",
                      color_discrete_sequence=["#F7941D"])
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        monthly = df.groupby(df["InvoiceDate"].dt.to_period("M").astype(str))["Revenue"].sum().reset_index()
        fig3 = px.bar(monthly, x="InvoiceDate", y="Revenue",
                      title="Monthly Revenue",
                      color_discrete_sequence=["#FBBA13"])
        st.plotly_chart(fig3, use_container_width=True)

# ── Page 2 ──────────────────────────────────────────
elif page == "🔮 Demand Forecast":
    st.title("🔮 Demand Forecast")

    forecast = load_forecast()
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    fig = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
                  title="30-Day Revenue Forecast",
                  color_discrete_sequence=["#E84E1B", "#F7941D", "#FBBA13"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Forecast Data")
    st.dataframe(forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(30))

    st.image(PROCESSED + r"\forecast_plot.png", caption="Prophet Forecast Plot")

# ── Page 3 ──────────────────────────────────────────
elif page == "⚠️ Churn Analysis":
    st.title("⚠️ Churn Risk Analysis")

    churn = load_churn()

    col1, col2, col3 = st.columns(3)
    high_risk = len(churn[churn["Churn_Probability"] > 0.7])
    med_risk = len(churn[(churn["Churn_Probability"] > 0.4) & (churn["Churn_Probability"] <= 0.7)])
    low_risk = len(churn[churn["Churn_Probability"] <= 0.4])

    col1.metric("🔴 High Risk", high_risk)
    col2.metric("🟡 Medium Risk", med_risk)
    col3.metric("🟢 Low Risk", low_risk)

    st.markdown("---")

    fig = px.histogram(churn, x="Churn_Probability",
                       title="Churn Probability Distribution",
                       color_discrete_sequence=["#E84E1B"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SHAP Feature Importance")
    st.image(PROCESSED + r"\shap_summary.png",
             caption="SHAP Summary Plot")

    st.markdown("### High Risk Customers")
    high = churn[churn["Churn_Probability"] > 0.7].sort_values(
        "Churn_Probability", ascending=False
    ).head(20)
    st.dataframe(high)

# ── Page 4 ──────────────────────────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")

    rfm = load_rfm()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(rfm, x="Recency", y="Monetary",
                         color="Segment",
                         title="Segments – Recency vs Monetary",
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig2 = px.pie(seg_counts, values="Count", names="Segment",
                      title="Customers per Segment")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Segment Summary")
    summary = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(2)
    st.dataframe(summary)

# ── Page 5 ──────────────────────────────────────────
elif page == "📦 Inventory Health":
    st.title("📦 Inventory Health")

    df = load_data()

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