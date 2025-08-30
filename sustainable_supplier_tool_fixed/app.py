import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Sustainable Supplier Selection",
    page_icon="🌱",
    layout="wide"
)

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        try:
            return pd.read_csv('supplier_data.csv')
        except FileNotFoundError:
            st.warning("No CSV file found. Generating sample data...")
            return generate_sample_data()

def generate_sample_data(num_suppliers=50):
    import numpy as np
    np.random.seed(42)
    data = {
        'supplier_id': range(1, num_suppliers+1),
        'name': [f'Supplier {i}' for i in range(1, num_suppliers+1)],
        'carbon_footprint': np.random.uniform(100, 1000, num_suppliers),
        'recycling_rate': np.random.uniform(20, 95, num_suppliers),
        'energy_efficiency': np.random.uniform(50, 95, num_suppliers),
        'water_usage': np.random.uniform(100, 10000, num_suppliers),
        'waste_production': np.random.uniform(10, 500, num_suppliers),
    }
    certifications = ['ISO_14001', 'Fair_Trade', 'Organic', 'B_Corp', 'Rainforest_Alliance']
    for cert in certifications:
        data[cert] = np.random.choice([0, 1], size=num_suppliers, p=[0.6, 0.4])
    locations = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
    industries = ['Electronics', 'Textiles', 'Food', 'Chemicals', 'Manufacturing']
    data['location'] = np.random.choice(locations, num_suppliers)
    data['industry'] = np.random.choice(industries, num_suppliers)
    df = pd.DataFrame(data)
    try:
        df.to_csv('supplier_data.csv', index=False)
    except:
        pass
    return df

def calculate_sustainability_score(row, weights):
    normalized_carbon = 1 - (row['carbon_footprint'] / 1000)
    normalized_recycling = row['recycling_rate'] / 100
    normalized_energy = row['energy_efficiency'] / 100
    normalized_water = 1 - (row['water_usage'] / 10000)
    normalized_waste = 1 - (row['waste_production'] / 500)
    cert_cols = ['ISO_14001', 'Fair_Trade', 'Organic', 'B_Corp', 'Rainforest_Alliance']
    cert_score = sum(row[cert] for cert in cert_cols) / len(cert_cols)
    score = (
        weights['carbon'] * normalized_carbon +
        weights['recycling'] * normalized_recycling +
        weights['energy'] * normalized_energy +
        weights['water'] * normalized_water +
        weights['waste'] * normalized_waste +
        weights['certifications'] * cert_score
    )
    return round(score * 100, 2)

def calculate_scores(df, weights):
    df['sustainability_score'] = df.apply(
        lambda row: calculate_sustainability_score(row, weights),
        axis=1
    )
    return df.sort_values('sustainability_score', ascending=False)

def main():
    st.title("Sustainable Supplier Selection Tool (Debug Mode)")

    # Upload
    uploaded_file = st.sidebar.file_uploader("Upload Supplier Data (CSV)", type=["csv"])
    df = load_data(uploaded_file)

    # Debug: Show uploaded file preview
    st.sidebar.markdown("### Uploaded Data Preview")
    st.sidebar.dataframe(df.head())

    st.write("### Debug: Full Data (first 10 rows)")
    st.dataframe(df.head(10))

    # Filters
    st.sidebar.header("Filters")
    cert_cols = ['ISO_14001', 'Fair_Trade', 'Organic', 'B_Corp', 'Rainforest_Alliance']
    cert_filters = {cert: st.sidebar.checkbox(cert.replace('_', ' '), value=True) for cert in cert_cols}
    industries = st.sidebar.multiselect("Industry", options=df['industry'].unique(), default=df['industry'].unique())
    locations = st.sidebar.multiselect("Location", options=df['location'].unique(), default=df['location'].unique())

    # Weights
    st.sidebar.subheader("Scoring Weights")
    weights = {
        'carbon': st.sidebar.slider("Carbon Footprint", 0.0, 0.3, 0.25, 0.05),
        'recycling': st.sidebar.slider("Recycling Rate", 0.0, 0.3, 0.15, 0.05),
        'energy': st.sidebar.slider("Energy Efficiency", 0.0, 0.3, 0.15, 0.05),
        'water': st.sidebar.slider("Water Usage", 0.0, 0.3, 0.15, 0.05),
        'waste': st.sidebar.slider("Waste Production", 0.0, 0.3, 0.15, 0.05),
        'certifications': st.sidebar.slider("Certifications", 0.0, 0.3, 0.15, 0.05),
    }

    # Apply filters
    filtered_df = df.copy()
    for cert, include in cert_filters.items():
        if not include:
            filtered_df = filtered_df[filtered_df[cert] == 0]
    filtered_df = filtered_df[filtered_df['industry'].isin(industries)]
    filtered_df = filtered_df[filtered_df['location'].isin(locations)]

    st.write(f"### Debug: Suppliers after filters → {len(filtered_df)} rows")
    if filtered_df.empty:
        st.error("⚠️ No suppliers left after applying filters!")
        return

    # Calculate scores
    scored_df = calculate_scores(filtered_df, weights)
    st.write("### Debug: Scored Data (first 10 rows)")
    st.dataframe(scored_df.head(10))

    # Continue with your normal charts / details / scenario comparison below…
    # (keep your existing logic here)

if __name__ == "__main__":
    main()

