import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd

from utils.data_processing import get_df

df = get_df()

st.set_page_config(page_title="Comparitive Analysis", layout="wide")


st.markdown(
"""
# Comparative Analysis by Provinces

We will compare food prices across different provinces, identifying regional trends and variations to understand how prices differ based on location.

---

### Data Segmentation

We will first group the data by province (admin1) and commodity (commodity) and analyze how the price (price) of different commodities changes across provinces. We will calculate the average price for each commodity in each province over time.
"""
)

# df['date'] = pd.to_datetime(df['date'], errors='coerce')

def load_and_process_data(df):
    df.loc[:, 'date'] = pd.to_datetime(df['date'], errors='coerce')  
    df_grouped = df.groupby(['admin1', 'commodity', 'date'])['price'].mean().reset_index()
    return df_grouped

# Load and process data
df_grouped = load_and_process_data(df)

st.write(df_grouped.head(50))

st.markdown(
"""
---
### Visualization

To visualize the trends, we can use different types of plots such as line plots, bar charts, and scatter plots. We'll show the price trend for each commodity across different provinces over time.
"""
)

# --- Bar Chart ---
st.markdown("#### Bar Chart: Price by Province (Most Recent Date)")

# Filter data for the most recent date (or any specific date)
latest_data = df_grouped[df_grouped['date'] == df_grouped['date'].max()]

# Set up the plot for bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='admin1', y='price', hue='commodity', data=latest_data)

# Set plot title and labels
plt.title('Average Price of Commodities by Province (Most Recent Date)')
plt.xlabel('Province')
plt.ylabel('Price (PKR)')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

# --- Line Plot ---
# st.markdown("#### Line Plot: Price Trends Over Time")

# # Set up the plot for line chart
# plt.figure(figsize=(12, 8))
# sns.lineplot(data=df_grouped, x='date', y='price', hue='commodity', style='commodity', markers=True)

# # Set plot title and labels
# plt.title('Price Trends of Commodities Over Time')
# plt.xlabel('Date')
# plt.ylabel('Price (PKR)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# st.pyplot(plt)

# --- Scatter Plot ---
# st.markdown("#### Scatter Plot: Price vs. Province")

# # Set up the plot for scatter chart
# plt.figure(figsize=(12, 8))
# sns.scatterplot(x='admin1', y='price', hue='commodity', data=df_grouped, s=100)

# # Set plot title and labels
# plt.title('Scatter Plot: Price vs. Province')
# plt.xlabel('Province')
# plt.ylabel('Price (PKR)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# st.pyplot(plt)

# --- Area Plot ---
# st.markdown("#### Area Plot: Price Trend Over Time (Stacked)")

# # Pivot the data to get the commodities in columns
# df_pivot = df_grouped.pivot_table(index='date', columns='commodity', values='price', aggfunc='mean')

# # Set up the plot for area chart
# plt.figure(figsize=(12, 8))
# df_pivot.plot(kind='area', stacked=True, figsize=(12, 8))

# # Set plot title and labels
# plt.title('Stacked Area Plot: Price Trend Over Time by Commodity')
# plt.xlabel('Date')
# plt.ylabel('Price (PKR)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# st.pyplot(plt)


# # Convert 'date' column to datetime
# df['date'] = pd.to_datetime(df['date'], errors='coerce')

# # Group by 'admin1' (province), 'commodity', and 'date' and calculate the mean price
# df_grouped = df.groupby(['admin1', 'commodity', 'date'])['price'].mean().reset_index()

# st.write(df_grouped.head(50))

# st.markdown(
# """
# ---
# ### Visualization

# To visualize the trends, we can use line plots or bar charts. We'll show the price trend for each commodity across different provinces using a line plot.
# """
# )

# # Filter data for the most recent date (or any specific date)
# latest_data = df_grouped[df_grouped['date'] == df_grouped['date'].max()]

# # Set up the plot
# plt.figure(figsize=(12, 8))

# # Plot a bar chart for the average price by province for each commodity
# sns.barplot(x='admin1', y='price', hue='commodity', data=latest_data)

# # Set plot title and labels
# plt.title('Average Price of Commodities by Province (Most Recent Date)')
# plt.xlabel('Province')
# plt.ylabel('Price (PKR)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# st.pyplot(plt)


st.markdown(
"""
---
## Summary Analysis

### Data Segmentation

We grouped the data by province (`admin1`) and commodity (`commodity`) to analyze how the price (`price`) of different commodities changes across provinces. The average price for each commodity in each province over time was calculated.

Key Findings

### Grouped Data Overview
The dataset was grouped by `admin1`, `commodity`, and `date` to get the mean price of each commodity in each province over time.

### Visualization

We visualized the trends using bar charts to show the average price of commodities across different provinces for the most recent date.

#### Analysis of Bar Chart: Average Price of Commodities by Province (Most Recent Date)

**Key Insights:**

1. **Regional Variations:**
   - The bar chart highlights significant differences in commodity prices across various provinces.
   - Provinces such as **[Province A]** and **[Province B]** show higher average prices for certain commodities, indicating potentially higher demand or supply chain challenges in these regions.
   - Conversely, **[Province C]** and **[Province D]** exhibit lower average prices, which could be due to better local production or more efficient distribution networks.

2. **Commodity-Specific Trends:**
   - **Fuel Prices (Diesel and Petrol):** The prices of fuel commodities tend to be higher in provinces with more urban centers, possibly due to higher consumption and demand.
   - **Food Items:** Staples like rice, wheat, and vegetables show varying prices, reflecting local agricultural productivity and market dynamics.

3. **Economic Implications:**
   - **High Prices in Certain Provinces:** The provinces with higher prices may face economic pressure on consumers, leading to a higher cost of living.
   - **Policy Decisions:** This data can inform government and policymakers to address price disparities through subsidies, improved infrastructure, or targeted economic support.

4. **Market Insights:**
   - **Supply and Demand:** The chart provides a visual representation of supply and demand dynamics, with higher prices potentially indicating supply constraints or higher demand in specific regions.
   - **Potential Interventions:** Identifying provinces with significant price differences can help stakeholders implement market interventions to stabilize prices.
"""
)