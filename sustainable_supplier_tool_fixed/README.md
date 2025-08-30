# Sustainable Supplier Selection Tool

A Streamlit application that helps organizations identify and evaluate suppliers based on sustainability metrics.

## Features

- **Supplier Filtering**: Filter by certifications (ISO 14001, Fair Trade, Organic, B Corp, Rainforest Alliance), industry, and location
- **Customizable Scoring**: Adjust weights for different sustainability factors
- **Supplier Ranking**: View suppliers ranked by sustainability score
- **Detailed Supplier Views**: See detailed metrics and certifications for each supplier
- **Scenario Simulation**: Compare environmental impact of switching suppliers
- **Interactive Visualizations**: Charts and graphs using Plotly

## How to Use

1. Use the sidebar filters to narrow down suppliers
2. Adjust scoring weights to match your sustainability priorities
3. View the top suppliers in the main dashboard
4. Click on individual suppliers to see detailed information
5. Use the scenario simulation to compare suppliers

## Deployment

This app is deployed on Streamlit Community Cloud:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app/)

## Local Development

To run locally:

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Data

The app uses sample data with the following sustainability metrics:
- Carbon Footprint
- Recycling Rate
- Energy Efficiency
- Water Usage
- Waste Production
- Sustainability Certifications
