i need the exact code...but i want want to see supplier_data.csv not found............modify that part only.......import streamlit as st
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

# Load data with error handling
@st.cache_data
def load_data():
    # Try to load the data file
    return pd.read_csv('supplier_data.csv')

# Scoring functions
def calculate_sustainability_score(row, weights):
    """
    Calculate overall sustainability score based on weighted factors
    """
    # Normalize quantitative metrics (higher is better for most)
    normalized_carbon = 1 - (row['carbon_footprint'] / 1000)
    normalized_recycling = row['recycling_rate'] / 100
    normalized_energy = row['energy_efficiency'] / 100
    normalized_water = 1 - (row['water_usage'] / 10000)
    normalized_waste = 1 - (row['waste_production'] / 500)
    
    # Certification points (qualitative factors)
    cert_cols = ['ISO_14001', 'Fair_Trade', 'Organic', 'B_Corp', 'Rainforest_Alliance']
    cert_score = sum(row[cert] for cert in cert_cols) / len(cert_cols)
    
    # Calculate weighted score
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
    """
    Calculate sustainability scores for all suppliers
    """
    df['sustainability_score'] = df.apply(
        lambda row: calculate_sustainability_score(row, weights), 
        axis=1
    )
    return df.sort_values('sustainability_score', ascending=False)

# Default weights
default_weights = {
    'carbon': 0.25,
    'recycling': 0.15,
    'energy': 0.15,
    'water': 0.15,
    'waste': 0.15,
    'certifications': 0.15
}

def main():
    st.title("Sustainable Supplier Selection Tool")
    st.markdown("Identify and evaluate suppliers based on sustainability metrics")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Certification filters
    st.sidebar.subheader("Certifications")
    cert_cols = ['ISO_14001', 'Fair_Trade', 'Organic', 'B_Corp', 'Rainforest_Alliance']
    cert_filters = {}
    for cert in cert_cols:
        cert_filters[cert] = st.sidebar.checkbox(cert.replace('_', ' '), value=True)
    
    # Industry filter
    industries = st.sidebar.multiselect(
        "Industry",
        options=df['industry'].unique(),
        default=df['industry'].unique()
    )
    
    # Location filter
    locations = st.sidebar.multiselect(
        "Location",
        options=df['location'].unique(),
        default=df['location'].unique()
    )
    
    # Score weights adjustment
    st.sidebar.subheader("Scoring Weights")
    weights = {}
    weights['carbon'] = st.sidebar.slider("Carbon Footprint", 0.0, 0.3, 0.25, 0.05)
    weights['recycling'] = st.sidebar.slider("Recycling Rate", 0.0, 0.3, 0.15, 0.05)
    weights['energy'] = st.sidebar.slider("Energy Efficiency", 0.0, 0.3, 0.15, 0.05)
    weights['water'] = st.sidebar.slider("Water Usage", 0.0, 0.3, 0.15, 0.05)
    weights['waste'] = st.sidebar.slider("Waste Production", 0.0, 0.3, 0.15, 0.05)
    weights['certifications'] = st.sidebar.slider("Certifications", 0.0, 0.3, 0.15, 0.05)
    
    # Check if weights sum to 1
    total_weight = sum(weights.values())
    if total_weight != 1.0:
        st.sidebar.warning(f"Weights sum to {total_weight:.2f} (should sum to 1.0)")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply certification filters
    for cert, include in cert_filters.items():
        if not include:
            filtered_df = filtered_df[filtered_df[cert] == 0]
    
    # Apply industry and location filters
    filtered_df = filtered_df[filtered_df['industry'].isin(industries)]
    filtered_df = filtered_df[filtered_df['location'].isin(locations)]
    
    # Calculate scores
    scored_df = calculate_scores(filtered_df, weights)
    
    # Display top suppliers
    st.header("Top Sustainable Suppliers")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.dataframe(
            scored_df[['name', 'industry', 'location', 'sustainability_score']].head(10),
            use_container_width=True
        )
    
    with col2:
        # Sustainability score distribution
        fig1 = px.histogram(scored_df, x='sustainability_score', 
                           title="Score Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col3:
        # Average scores by industry
        industry_scores = scored_df.groupby('industry')['sustainability_score'].mean().reset_index()
        fig2 = px.bar(industry_scores, x='industry', y='sustainability_score',
                     title="Average Score by Industry")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed supplier view
    st.header("Supplier Details")
    selected_supplier = st.selectbox(
        "Select a supplier to view details",
        options=scored_df['name'].tolist()
    )
    
    if selected_supplier:
        supplier_data = scored_df[scored_df['name'] == selected_supplier].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sustainability Metrics")
            metrics = {
                'Carbon Footprint': supplier_data['carbon_footprint'],
                'Recycling Rate': supplier_data['recycling_rate'],
                'Energy Efficiency': supplier_data['energy_efficiency'],
                'Water Usage': supplier_data['water_usage'],
                'Waste Production': supplier_data['waste_production']
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2f}")
        
        with col2:
            st.subheader("Certifications")
            certs = []
            for cert in cert_cols:
                if supplier_data[cert] == 1:
                    certs.append(cert.replace('_', ' '))
            
            if certs:
                for cert in certs:
                    st.success(f"{cert}")
            else:
                st.warning("No certifications")
    
    # Scenario simulation
    st.header("Scenario Simulation")
    st.subheader("Compare environmental impact of switching suppliers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_supplier = st.selectbox(
            "Current Supplier",
            options=scored_df['name'].tolist(),
            key="current"
        )
    
    with col2:
        alternative_supplier = st.selectbox(
            "Alternative Supplier",
            options=scored_df['name'].tolist(),
            key="alternative"
        )
    
    if current_supplier and alternative_supplier and current_supplier != alternative_supplier:
        current = scored_df[scored_df['name'] == current_supplier].iloc[0]
        alternative = scored_df[scored_df['name'] == alternative_supplier].iloc[0]
        
        # Calculate impact differences
        impact_diff = {
            'Carbon Footprint': alternative['carbon_footprint'] - current['carbon_footprint'],
            'Water Usage': alternative['water_usage'] - current['water_usage'],
            'Waste Production': alternative['waste_production'] - current['waste_production']
        }
        
        # Display comparison
        st.subheader("Environmental Impact Comparison")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=list(impact_diff.keys()),
            y=[current['carbon_footprint'], current['water_usage'], current['waste_production']],
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='Alternative',
            x=list(impact_diff.keys()),
            y=[alternative['carbon_footprint'], alternative['water_usage'], alternative['waste_production']],
            marker_color='green'
        ))
        
        fig.update_layout(barmode='group', title_text="Environmental Impact Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary of changes
        improvements = []
        declines = []
        
        for metric, diff in impact_diff.items():
            if diff < 0:
                improvements.append(f"{metric}: {abs(diff):.2f} reduction")
            else:
                declines.append(f"{metric}: {diff:.2f} increase")
        
        if improvements:
            st.success("Improvements: " + ", ".join(improvements))
        if declines:
            st.error("Declines: " + ", ".join(declines))
        
        # Overall recommendation
        if len(improvements) > len(declines):
            st.success("Recommendation: Consider switching to this supplier")
        elif len(improvements) < len(declines):
            st.error("Recommendation: Not recommended to switch to this supplier")
        else:
            st.warning("Recommendation: Mixed impact - consider other factors")
    elif current_supplier == alternative_supplier:
        st.warning("Please select different suppliers for comparison")

if __name__ == "__main__":
    main()
