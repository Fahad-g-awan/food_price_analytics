import streamlit as st
import pandas as pd

df = pd.read_csv('wfp_food_prices.csv')

# Data cleanup
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['usdprice'] = pd.to_numeric(df['usdprice'], errors='coerce')
df.dropna(subset=['price', 'usdprice'], inplace=True)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Sort the data by date to make it time-series ready
df.sort_values(by='date', inplace=True)

def get_df():
    if df.empty:
        st.error("The dataset is empty. Please check the data source.")
        st.subheader("The dataset is empty. Please check the data source.")
        return
    return df

