import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="Sustainable Supplier Selection",
    page_icon="🌱",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        # fallback file if exists
        if os.path.exists("supplier_data.csv"):
            return pd.read_csv("supplier_data.csv")
        else:
            return pd.DataFrame()

# Calculate scores
def calculate_scores(df, weights):
    if df.empty:
        return df

    df = df.copy()

    # Normalize numerical criteria (if present)
    for col in ["carbon_footprint", "recycling_rate", "energy_efficiency", "water_usage", "waste_production"]:
        if col in df.columns:
            if col in ["carbon_footprint", "water_usage", "waste_production"]:
                df[col + "_norm"] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                df[col + "_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Certification score
    cert_cols = [c for c in ["ISO_14001", "Fair_Trade", "Organic", "B_Corp", "Rainforest_Alliance"] if c in df.columns]
    if cert_cols:
        df["certification_score"] = df[cert_cols].sum(axis=1) / len(cert_cols)
    else:
        df["certification_score"] = 0

    # Weighted score
    df["sustainability_score"] = 0
    if "carbon_footprint_norm" in df.columns:
        df["sustainability_score"] += weights["Carbon Footprint"] * df["carbon_footprint_norm"]
    if "recycling_rate_norm" in df.columns:
        df["sustainability_score"] += weights["Recycling Rate"] * df["recycling_rate_norm"]
    if "energy_efficiency_norm" in df.columns:
        df["sustainability_score"] += weights["Energy Efficiency"] * df["energy_efficiency_norm"]
    if "water_usage_norm" in df.columns:
        df["sustainability_score"] += weights["Water Usage"] * df["water_usage_norm"]
    if "waste_production_norm" in df.columns:
        df["sustainability_score"] += weights["Waste Production"] * df["waste_production_norm"]
    if "certification_score" in df.columns:
        df["sustainability_score"] += weights["Certifications"] * df["certification_score"]

    return df

# Main app
def main():
    st.title("🌱 Sustainable Supplier Selection Tool")

    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload Supplier CSV", type="csv")
    df = load_data(uploaded_file)

    # Debug: Show available columns
    st.sidebar.write("📂 Columns in dataset:", df.columns.tolist())

    if df.empty:
        st.warning("No data available. Please upload a dataset.")
        return

    # Weights
    st.sidebar.header("⚖️ Criteria Weights")
    weights = {
        "Carbon Footprint": st.sidebar.slider("Carbon Footprint", 0.0, 1.0, 0.2),
        "Recycling Rate": st.sidebar.slider("Recycling Rate", 0.0, 1.0, 0.2),
        "Energy Efficiency": st.sidebar.slider("Energy Efficiency", 0.0, 1.0, 0.2),
        "Water Usage": st.sidebar.slider("Water Usage", 0.0, 1.0, 0.1),
        "Waste Production": st.sidebar.slider("Waste Production", 0.0, 1.0, 0.1),
        "Certifications": st.sidebar.slider("Certifications", 0.0, 1.0, 0.2),
    }

    if abs(sum(weights.values()) - 1.0) > 0.01:
        st.sidebar.warning("⚠️ The weights should sum to 1 for proper scoring.")

    # Filters
    st.sidebar.header("🔍 Filters")

    if "industry" in df.columns:
        industry_filter = st.sidebar.multiselect(
            "Select Industry",
            options=df["industry"].dropna().unique(),
            default=df["industry"].dropna().unique()
        )
        df = df[df["industry"].isin(industry_filter)]
    else:
        st.sidebar.warning("⚠️ No 'industry' column found in dataset")

    if "location" in df.columns:
        location_filter = st.sidebar.multiselect(
            "Select Location",
            options=df["location"].dropna().unique(),
            default=df["location"].dropna().unique()
        )
        df = df[df["location"].isin(location_filter)]
    else:
        st.sidebar.warning("⚠️ No 'location' column found in dataset")

    # Calculate scores
    df = calculate_scores(df, weights)

    if df.empty:
        st.warning("⚠️ No suppliers match the selected filters.")
        return

    # Ranking
    st.subheader("🏆 Supplier Ranking")
    st.dataframe(df.sort_values("sustainability_score", ascending=False)[
        ["supplier_id", "name", "sustainability_score"]
    ].reset_index(drop=True))

    # Visualization
    st.subheader("📊 Score Distribution")
    fig = px.histogram(df, x="sustainability_score", nbins=10, title="Distribution of Sustainability Scores")
    st.plotly_chart(fig, use_container_width=True)

    if "industry" in df.columns:
        st.subheader("📌 Industry Comparison")
        fig2 = px.box(df, x="industry", y="sustainability_score", title="Score Distribution by Industry")
        st.plotly_chart(fig2, use_container_width=True)

    if "location" in df.columns:
        st.subheader("🌍 Location Comparison")
        fig3 = px.box(df, x="location", y="sustainability_score", title="Score Distribution by Location")
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()

    main()
