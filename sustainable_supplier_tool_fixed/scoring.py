
import pandas as pd

# Default weights
default_weights = {
    'carbon': 0.25,
    'recycling': 0.15,
    'energy': 0.15,
    'water': 0.15,
    'waste': 0.15,
    'certifications': 0.15
}

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
