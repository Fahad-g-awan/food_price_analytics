from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import io

from utils.data_processing import get_df

df = get_df()


st.set_page_config(page_title="Data Analysis And Preprocessing", layout="wide")

st.markdown(
"""
# Data Analysis and Preprocessing

We will clean, normalize, and transform the data to ensure it is accurate and ready for analysis. This step is crucial for preparing the dataset for further exploration and modeling.

---
### Loading the dataset and perform initial exploration
"""
)

st.markdown(
"""
##### Raw Data
"""
)
st.write(df.head())

st.markdown(
"""
---
#### Data Information
"""
)
st.text(f"Total records: {df.shape[0]} rows and {df.shape[1]} columns")
st.markdown(
"""
##### Data types
"""
)
st.write(df.dtypes)

# Checking for missing values
st.markdown(
"""
---
### Missing Values
"""
)
missing_values = df.isnull().sum()
st.text("Checking for missing values.")
st.text(missing_values)

st.markdown(
"""
---
### Data Cleaning

Cleaning steps includes handling missing values and ensuring numerical data types are correct
"""
)

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.markdown(
"""
---
### Data Normalization

Normalization helps in scaling data so that features have a similar range, which is particularly useful for models that rely on distance metrics
"""
)

scaler = MinMaxScaler()
df[['price', 'usdprice']] = scaler.fit_transform(df[['price', 'usdprice']])

st.markdown(
"""
##### Normalized Data (first 5 rows)
"""
)
st.write(df.head())

st.markdown(
"""
---
### Data Transformation

- For categorical columns, we need to convert them into a suitable format for analysis
- For date, converting date to date_time and sorting
"""
)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Sort the data by date to make it time-series ready
df.sort_values(by='date', inplace=True)

st.markdown(
"""
### Visual Exploration

Initial data trends to gain more insights
"""
)

# Plot prices over time
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['price'], label='Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Trend Over Time')
plt.grid()
plt.legend()
st.pyplot(plt)

st.markdown(
"""
---
## Summary of Analysis

The dataset was initially cleaned to ensure the accuracy of numerical data types and to handle any missing values. The cleaned data was then normalized to bring the price and usdprice columns to a similar range, facilitating better analysis.

Through data transformation, we converted the date column into a proper datetime format and sorted it to prepare for time-series analysis. Visual exploration of the data revealed notable trends in prices over time, with significant fluctuations particularly evident from 2012 onwards. This trend highlights the periods of volatility and overall growth in prices, providing valuable insights into market dynamics.
"""
)