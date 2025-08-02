import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Working directory:", os.getcwd())

sns.set(style="whitegrid")
plt.switch_backend('Agg')

df = pd.read_csv('startup_failure_prediction.csv')
print("Columns:", df.columns.tolist())

df['Startup_Status'] = df['Startup_Status'].astype(str).str.strip().str.lower()
print("Unique statuses:", df['Startup_Status'].unique())

df_failed = df[df['Startup_Status'] == '1']
print(f"Found {len(df_failed)} failed startups.")

if 'Industry' in df_failed.columns:
    industry_counts = df_failed['Industry'].dropna().value_counts().head(10)
else:
    industry_counts = pd.Series(dtype='int')

if not industry_counts.empty:
    print("\nTop 10 industries where startups failed:\n")
    print(industry_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=industry_counts.values, y=industry_counts.index, color='orange')
    plt.title('Top 10 Industries Where Startups Failed')
    plt.xlabel('Number of Failed Startups')
    plt.ylabel('Industry')
    plt.tight_layout()
    plt.savefig('failure_industries_chart.png', dpi=300)
    print("Saved: failure_industries_chart.png")
else:
    print("No failed startups with industry data found to plot.")

if {'Funding_Amount', 'Burn_Rate'}.issubset(df_failed.columns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_failed, x='Funding_Amount', y='Burn_Rate', hue='Industry', legend=False)
    plt.title("Funding vs Burn Rate")
    plt.xlabel("Funding Amount")
    plt.ylabel("Burn Rate")
    plt.tight_layout()
    plt.savefig("funding_vs_burnrate.png", dpi=300)
    print("Saved: funding_vs_burnrate.png")

if {'Revenue', 'Customer_Retention_Rate'}.issubset(df_failed.columns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_failed, x='Customer_Retention_Rate', y='Revenue', hue='Industry', legend=False)
    plt.title("Customer Retention vs Revenue")
    plt.xlabel("Customer Retention Rate")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig("retention_vs_revenue.png", dpi=300)
    print("Saved: retention_vs_revenue.png")

numeric_cols = df_failed.select_dtypes(include='number')
if not numeric_cols.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap (Failed Startups)")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)
    print("Saved: correlation_heatmap.png")

df_failed.to_csv("failed_startups_only.csv", index=False)
print("Saved: failed_startups_only.csv")
