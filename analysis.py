import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Global visual formatting
sns.set_theme(style="whitegrid", font_scale=1.1)

# 1. LOAD DATA
df = pd.read_csv('data/life_expectancy.csv')

# 2. DATA CLEANING & PREPROCESSING (LO2 & LO5)
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Impute missing values with group medians (by Status) to avoid skewing
impute_cols = ['Life expectancy', 'Adult Mortality', 'Alcohol', 'GDP', 'Schooling', 'Income composition of resources']
for col in impute_cols:
    if col in df.columns:
        df[col] = df.groupby('Status')[col].transform(lambda x: x.fillna(x.median()))

# Drop any remaining unhandled NaNs in target metrics
df = df.dropna(subset=['Life expectancy', 'Adult Mortality', 'Schooling'])
df.to_csv('data/cleaned_life_expectancy.csv', index=False)
print("Data cleaned successfully! Exported to data/cleaned_life_expectancy.csv")

# -------------------------------------------------------------
# PLOT 1: Group 1 (Histogram - Distribution)
# -------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['Life expectancy'], kde=True, color='#1f77b4', bins=25, edgecolor='black', alpha=0.6)
mean_val = df['Life expectancy'].mean()
median_val = df['Life expectancy'].median()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f} yrs')
plt.axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f} yrs')

plt.title('Figure 1: Global Distribution of Life Expectancy (2000–2015)', fontsize=13, weight='bold', pad=12)
plt.xlabel('Life Expectancy (Years)', fontsize=11)
plt.ylabel('Frequency (Country-Year Observations)', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('figures/fig1_distribution.png', dpi=300)
plt.close()
print("Figure 1 generated: figures/fig1_distribution.png")

# -------------------------------------------------------------
# PLOT 2: Group 2 (Line Chart - Trends over Time)
# -------------------------------------------------------------
trend_df = df.groupby(['Year', 'Status'])['Life expectancy'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.lineplot(data=trend_df, x='Year', y='Life expectancy', hue='Status', style='Status', markers=True, dashes=False, palette={'Developed': '#2ca02c', 'Developing': '#d62728'}, linewidth=2.5)

plt.title('Figure 2: Life Expectancy Trajectory Over Time (2000–2015)', fontsize=13, weight='bold', pad=12)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Mean Life Expectancy (Years)', fontsize=11)
plt.xticks(np.arange(df['Year'].min(), df['Year'].max() + 1, 2))
plt.legend(title='Development Status', frameon=True)
plt.tight_layout()
plt.savefig('figures/fig2_trend.png', dpi=300)
plt.close()
print("Figure 2 generated: figures/fig2_trend.png")

# -------------------------------------------------------------
# PLOT 3: Group 3 (Heatmap - Multivariate Relationships)
# -------------------------------------------------------------
selected_features = ['Life expectancy', 'Adult Mortality', 'HIV/AIDS', 'Schooling', 'Income composition of resources', 'GDP', 'Alcohol']
corr = df[[col for col in selected_features if col in df.columns]].corr()

plt.figure(figsize=(8.5, 6.5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='vlag', vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation (r)'}, linewidths=0.5)

plt.title('Figure 3: Correlation Matrix of Key Health & Socioeconomic Factors', fontsize=13, weight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/fig3_heatmap.png', dpi=300)
plt.close()
print("Figure 3 generated: figures/fig3_heatmap.png")

# -------------------------------------------------------------
# EXACT METRICS TO QUOTE IN REPORT
# -------------------------------------------------------------
print("\n" + "="*50)
print("--- SUMMARY METRICS FOR YOUR REPORT ---")
print("="*50)
print(f"Overall Mean: {mean_val:.2f} years")
print(f"Overall Median: {median_val:.2f} years")
print(f"Standard Deviation: {df['Life expectancy'].std():.2f} years")
print(f"Developed Mean: {df[df['Status'] == 'Developed']['Life expectancy'].mean():.2f} years")
print(f"Developing Mean: {df[df['Status'] == 'Developing']['Life expectancy'].mean():.2f} years")
print(f"Correlation (Life Expectancy vs Schooling): {df['Life expectancy'].corr(df['Schooling']):.2f}")
print(f"Correlation (Life Expectancy vs Adult Mortality): {df['Life expectancy'].corr(df['Adult Mortality']):.2f}")
print(f"Correlation (Life Expectancy vs HIV/AIDS): {df['Life expectancy'].corr(df['HIV/AIDS']):.2f}")
print("="*50)
