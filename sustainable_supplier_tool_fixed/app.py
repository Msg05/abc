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

# Load data
@st.cache_data
def load_data(file_path=None):
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

# Scoring function
def calculate_sustainability_score(row):
    try:
        carbon_score = max(0, 100 - (row['carbon_footprint'] / 10))
        recycling_score = row['recycling_rate']
        energy_score = row['energy_efficiency']
        water_score = max(0, 100 - (row['water_usage'] / 100))
        waste_score = max(0, 100 - (row['waste_production'] / 5))
        certifications = sum([
            row.get('ISO_14001', 0),
            row.get('Fair_Trade', 0),
            row.get('Organic', 0),
            row.get('B_Corp', 0),
            row.get('Rainforest_Alliance', 0)
        ]) * 5
        return (carbon_score + recycling_score + energy_score +
                water_score + waste_score + certifications) / 6
    except Exception:
        return 0

# Main app
def main():
    st.title("🌱 Sustainable Supplier Selection Tool")
    st.markdown("An AI-powered tool for evaluating and comparing suppliers based on sustainability metrics.")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your supplier data (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = df.columns.str.strip().str.lower()

        # Rename back to expected
        df.rename(columns={
            'supplier id': 'supplier_id',
            'supplier name': 'name'
        }, inplace=True)

        # Fill missing certifications with 0
        for col in ['iso_14001','fair_trade','organic','b_corp','rainforest_alliance']:
            if col not in df.columns:
                df[col] = 0

        # Calculate sustainability score
        df['sustainability_score'] = df.apply(calculate_sustainability_score, axis=1)

        # Sidebar filters
        st.sidebar.header("Filters")
        industry_filter = st.sidebar.multiselect(
            "Select Industry",
            options=df['industry'].unique(),
            default=df['industry'].unique()
        )
        location_filter = st.sidebar.multiselect(
            "Select Location",
            options=df['location'].unique(),
            default=df['location'].unique()
        )

        filtered_df = df[
            (df['industry'].isin(industry_filter)) &
            (df['location'].isin(location_filter))
        ]

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard", "🏆 Rankings", "📈 Trends", "🔮 Scenario Simulation"
        ])

        # Dashboard
        with tab1:
            st.header("Supplier Sustainability Dashboard")

            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(filtered_df, x="sustainability_score", nbins=20, title="Distribution of Sustainability Scores")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.box(filtered_df, x="industry", y="sustainability_score", title="Scores by Industry")
                st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.scatter(filtered_df, x="carbon_footprint", y="energy_efficiency",
                              size="sustainability_score", color="industry",
                              hover_name="name", title="Carbon vs Energy Efficiency")
            st.plotly_chart(fig3, use_container_width=True)

        # Rankings
        with tab2:
            st.header("Supplier Rankings")
            top_n = st.slider("Select number of top suppliers to display", 5, 20, 10)
            top_suppliers = filtered_df.nlargest(top_n, 'sustainability_score')
            st.dataframe(top_suppliers[['supplier_id','name','industry','location','sustainability_score']])
            fig4 = px.bar(top_suppliers, x="name", y="sustainability_score", color="industry",
                          title="Top Suppliers by Sustainability Score")
            st.plotly_chart(fig4, use_container_width=True)

        # Trends
        with tab3:
            st.header("Sustainability Trends")
            industry_trends = filtered_df.groupby("industry")["sustainability_score"].mean().reset_index()
            fig5 = px.bar(industry_trends, x="industry", y="sustainability_score",
                          title="Average Sustainability Score by Industry")
            st.plotly_chart(fig5, use_container_width=True)

            location_trends = filtered_df.groupby("location")["sustainability_score"].mean().reset_index()
            fig6 = px.bar(location_trends, x="location", y="sustainability_score",
                          title="Average Sustainability Score by Location")
            st.plotly_chart(fig6, use_container_width=True)

        # Scenario Simulation
        with tab4:
            st.header("Scenario Simulation")
            st.markdown("Compare different supplier choices and their environmental impacts.")

            suppliers = filtered_df['name'].unique()
            col1, col2 = st.columns(2)
            with col1:
                current_supplier = st.selectbox("Current Supplier", suppliers, key="current_supplier")
            with col2:
                alternative_supplier = st.selectbox("Alternative Supplier", suppliers, key="alt_supplier")

            if current_supplier and alternative_supplier:
                current = filtered_df[filtered_df['name'] == current_supplier].iloc[0]
                alternative = filtered_df[filtered_df['name'] == alternative_supplier].iloc[0]

                # Calculate impact differences
                impact_diff = {
                    'Carbon Footprint': alternative['carbon_footprint'] - current['carbon_footprint'],
                    'Water Usage': alternative['water_usage'] - current['water_usage'],
                    'Waste Production': alternative['waste_production'] - current['waste_production'],
                    'Recycling Rate': alternative['recycling_rate'] - current['recycling_rate'],
                    'Energy Efficiency': alternative['energy_efficiency'] - current['energy_efficiency']
                }

                # Display comparison
                st.subheader("Environmental Impact Comparison")
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    name='Current',
                    x=list(impact_diff.keys()),
                    y=[
                        current['carbon_footprint'],
                        current['water_usage'],
                        current['waste_production'],
                        current['recycling_rate'],
                        current['energy_efficiency']
                    ],
                    marker_color='blue'
                ))

                fig.add_trace(go.Bar(
                    name='Alternative',
                    x=list(impact_diff.keys()),
                    y=[
                        alternative['carbon_footprint'],
                        alternative['water_usage'],
                        alternative['waste_production'],
                        alternative['recycling_rate'],
                        alternative['energy_efficiency']
                    ],
                    marker_color='green'
                ))

                fig.update_layout(barmode='group', title_text="Environmental Impact Comparison")
                st.plotly_chart(fig, use_container_width=True)

                # Improvements / declines
                improvements = []
                declines = []

                for metric, diff in impact_diff.items():
                    if metric in ['Recycling Rate', 'Energy Efficiency']:
                        if diff > 0:
                            improvements.append(f"{metric}: {diff:.2f} increase")
                        else:
                            declines.append(f"{metric}: {abs(diff):.2f} decrease")
                    else:
                        if diff < 0:
                            improvements.append(f"{metric}: {abs(diff):.2f} reduction")
                        else:
                            declines.append(f"{metric}: {diff:.2f} increase")

                col1, col2 = st.columns(2)
                with col1:
                    st.success("Improvements")
                    for imp in improvements:
                        st.write("✅ " + imp)
                with col2:
                    st.error("Potential Declines")
                    for dec in declines:
                        st.write("⚠️ " + dec)

    else:
        st.info("👆 Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()

