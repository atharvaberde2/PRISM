"""
Generate synthetic US Census-style data for real estate investment analysis.
Problem: Identify best zip codes for single-family homes targeting households aged 25-45
with incomes over $100,000, projecting growth through 2030.

Streamlined to ~15 meaningful features that work well with PRISM's 5-stage pipeline.
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

N = 500  # number of zip code records

# --- AREA TYPE ---
area_types = np.random.choice(['Urban', 'Suburban', 'Rural'], size=N, p=[0.30, 0.50, 0.20])

# --- REGION ---
regions = np.random.choice(['South', 'West', 'Midwest', 'Northeast'], size=N, p=[0.35, 0.30, 0.20, 0.15])

# --- DEMOGRAPHIC FEATURES ---
# Total population per zip
total_population = np.random.lognormal(mean=9.5, sigma=0.7, size=N).astype(int)
total_population = np.clip(total_population, 1000, 200000)

# Pct of population aged 25-45 (our target demographic)
pct_age_25_45 = np.random.beta(5, 7, size=N) * 0.5 + 0.15  # range ~15-65%, centered ~35%

# Median age (correlated with pct_age_25_45)
median_age = 50 - (pct_age_25_45 * 30) + np.random.normal(0, 3, size=N)
median_age = np.clip(median_age, 22, 65).round(1)

# --- INCOME FEATURES ---
# Median household income
median_hh_income = np.random.lognormal(mean=11.2, sigma=0.35, size=N).astype(int)
median_hh_income = np.clip(median_hh_income, 30000, 350000)

# Pct households earning > $100k (correlated with median income)
base_pct_100k = (median_hh_income - 30000) / 320000  # normalize
pct_income_over_100k = np.clip(base_pct_100k + np.random.normal(0, 0.08, size=N), 0.02, 0.85)

# --- HOUSING FEATURES ---
# Median home value
median_home_value = (median_hh_income * np.random.uniform(3.0, 7.0, size=N)).astype(int)

# Homeownership rate
homeownership_rate = np.random.beta(6, 4, size=N)
homeownership_rate = np.clip(homeownership_rate, 0.20, 0.95)

# Vacancy rate
vacancy_rate = np.random.beta(2, 15, size=N)
vacancy_rate = np.clip(vacancy_rate, 0.01, 0.30)

# Pct single family homes
pct_single_family = np.where(
    np.array(area_types) == 'Rural', np.random.beta(7, 2, size=N),
    np.where(np.array(area_types) == 'Suburban', np.random.beta(6, 3, size=N),
             np.random.beta(3, 5, size=N))
)

# --- GROWTH / TIME-SERIES TRENDS ---
# Population growth rate (annual, 2020-2024)
pop_growth_rate = np.random.normal(0.012, 0.02, size=N)
pop_growth_rate = np.clip(pop_growth_rate, -0.05, 0.08)

# Home value appreciation (annual, 2020-2024)
home_value_appreciation = np.random.normal(0.05, 0.03, size=N)
home_value_appreciation = np.clip(home_value_appreciation, -0.05, 0.20)

# New building permits (per 1000 housing units, 2024)
new_permits_per_1000 = np.random.exponential(15, size=N)
new_permits_per_1000 = np.clip(new_permits_per_1000, 0, 80).round(1)

# --- ECONOMIC INDICATORS ---
# Unemployment rate
unemployment_rate = np.random.beta(2, 20, size=N)
unemployment_rate = np.clip(unemployment_rate, 0.01, 0.15)

# Education: pct with bachelor's degree or higher
pct_bachelors_plus = np.clip(
    (median_hh_income - 30000) / 400000 + np.random.normal(0, 0.10, size=N),
    0.05, 0.80
)

# Avg commute time
avg_commute_time = np.random.normal(28, 10, size=N)
avg_commute_time = np.clip(avg_commute_time, 5, 75).round(1)

# --- TARGET VARIABLE: high_potential (investment recommendation) ---
# Score based on multiple factors
score = (
    0.25 * (pct_age_25_45 - 0.25) / 0.30 +          # high % of target age group
    0.20 * (pct_income_over_100k - 0.20) / 0.50 +    # high % earning >100k
    0.15 * np.clip(pop_growth_rate / 0.05, -1, 1) +   # growing population
    0.10 * np.clip(home_value_appreciation / 0.10, -1, 1) + # growing home values
    0.10 * (1 - vacancy_rate) +                        # low vacancy
    0.10 * pct_single_family +                          # single family friendly
    0.05 * (1 - unemployment_rate / 0.15) +            # low unemployment
    0.05 * pct_bachelors_plus                           # educated population
)

# Add noise
score += np.random.normal(0, 0.08, size=N)

# Binary classification: top ~40% are "high potential"
threshold = np.percentile(score, 58)
high_potential = np.where(score >= threshold, 'Yes', 'No')

# --- BUILD DATAFRAME ---
df = pd.DataFrame({
    'region': regions,
    'area_type': area_types,
    'total_population': total_population,
    'pct_age_25_45': (pct_age_25_45 * 100).round(2),
    'median_age': median_age,
    'median_hh_income': median_hh_income,
    'pct_income_over_100k': (pct_income_over_100k * 100).round(2),
    'median_home_value': median_home_value,
    'homeownership_rate': (homeownership_rate * 100).round(2),
    'vacancy_rate': (vacancy_rate * 100).round(2),
    'pct_single_family': (pct_single_family * 100).round(2),
    'pop_growth_rate': (pop_growth_rate * 100).round(3),
    'home_value_appreciation': (home_value_appreciation * 100).round(3),
    'new_permits_per_1000': new_permits_per_1000,
    'unemployment_rate': (unemployment_rate * 100).round(2),
    'pct_bachelors_plus': (pct_bachelors_plus * 100).round(2),
    'avg_commute_time': avg_commute_time,
    'high_potential': high_potential,
})

# --- INTRODUCE MISSING VALUES (realistic patterns, ~3-7% per column) ---
missing_cols = [
    'pct_age_25_45', 'median_hh_income', 'pct_income_over_100k',
    'median_home_value', 'vacancy_rate', 'pop_growth_rate',
    'unemployment_rate', 'avg_commute_time', 'pct_bachelors_plus',
]

for col in missing_cols:
    mask = np.random.random(N) < np.random.uniform(0.03, 0.07)
    df.loc[mask, col] = np.nan

# Save
output_path = 'real_estate_census_data.csv'
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} records with {len(df.columns)} columns")
print(f"\nTarget distribution (high_potential):")
print(df['high_potential'].value_counts())
print(f"\nImbalance ratio: {df['high_potential'].value_counts().max() / df['high_potential'].value_counts().min():.2f}")
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample rows:")
print(df.head(3).to_string())
print(f"\nRows after dropna: {len(df.dropna())} (need >= 20 for Stage 4)")
print(f"\nSaved to: {output_path}")
