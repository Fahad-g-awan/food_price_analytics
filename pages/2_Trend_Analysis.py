from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from utils.data_processing import get_df

df = get_df()

st.set_page_config(page_title="Trend Analysis", layout="wide")

st.markdown(
"""
# Trend Analysis

This section involves identifying general trends in food prices over time using visual and statistical methods, including time series decomposition and ARIMA modeling.

---
### Visualize Food Price Trends

To get an initial idea of how food prices have behaved over time, we can plot the prices
"""
)

# Set the plot style for better aesthetics
sns.set(style="whitegrid")

# Plot the food prices over time
plt.figure(figsize=(14, 7))
sns.lineplot(x='date', y='price', data=df, label='Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Food Price Trend Over Time')
plt.xticks(rotation=45)
plt.grid(visible=True)
plt.legend()
plt.tight_layout()
st.pyplot(plt)

st.markdown(
"""
---
### Seasonal Analysis

To analyze seasonality and trends more formally, we can use time-series decomposition. This will allow us to break down the data into its trend, seasonality, and residual components
"""
)

seasonal_df = df.set_index('date')

# Apply time-series decomposition (e.g., using 'price' as the target variable)
result = seasonal_decompose(seasonal_df['price'], model='additive', period=365)

# Plot the decomposition
result.plot()
plt.suptitle('Seasonal Decomposition of Food Prices')
plt.tight_layout()
st.pyplot(plt)

st.text("\n")
st.markdown(
"""
##### Monthly Price Distribution
"""
)

seasonal_df['month'] = seasonal_df.index.month

# Boxplot for monthly price trends
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='price', data=seasonal_df)
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Monthly Price Distribution')
st.pyplot(plt)


st.markdown(
"""
---
### Time Series Analysis (for Trends)

Since the project mentions finding general trends over time, time series analysis models could help in understanding price fluctuations:

ARIMA (AutoRegressive Integrated Moving Average) is a strong model for time series forecasting and analyzing the trend and seasonality in data.
"""
)

model = ARIMA(df['price'], order=(5, 1, 0))
model_fit = model.fit()
st.text("Model Summary by ARIMA:")
st.text(model_fit.summary())

st.markdown(
"""
---
## In-Depth Analysis Summary

### 1. Trend Analysis:

Using the ARIMA (AutoRegressive Integrated Moving Average) model, we analyzed the time-series data for food prices to identify trends and patterns. The ARIMA model is particularly effective for forecasting and understanding the components of the data, including trends and seasonality.

### 2. ARIMA Model Insights:

- Model Order: The ARIMA model used is of order (5, 1, 0), which indicates that the model uses 5 lag observations (AR), one differencing operation (I), and no moving average component (MA).
- Coefficients: The coefficients of the autoregressive terms are significant, suggesting that past - values have a substantial impact on the current price. For instance, ar.L1 has a coefficient of -0.7168, indicating a strong negative relationship with the immediate past value.
- Significance: All the AR terms (from L1 to L5) have p-values close to 0, indicating their significance in the model.
- Residuals: The residuals (difference between observed and predicted values) are captured by sigma2, which is approximately 31,800, reflecting the variance in the data.

### 3. Seasonal Decomposition:

The seasonal decomposition of the food prices breaks down the data into:
- Trend: The long-term progression of the series.
- Seasonal: The repeating short-term cycle in the series.
- Residual: The random variation after removing the trend and seasonal components.

The decomposition provides clear visual insights into how prices fluctuate over time due to seasonal factors.

### 4. Monthly Price Distribution:

The boxplot of monthly price distribution helps in understanding how prices vary across different months. This visualization reveals seasonal patterns and variations in prices, indicating periods of higher or lower prices.

### 5. Statistical Significance:

- Ljung-Box Test: The Ljung-Box test result (Q) is 2.67 with a p-value of 0.10, suggesting that the residuals are not significantly different from white noise, indicating a good fit of the model.
- Jarque-Bera Test: The Jarque-Bera test shows a significant skew (2.06) and kurtosis (11.83), indicating that the data is not normally distributed, which is common in real-world economic data.

### 6. Model Evaluation:

The ARIMA model provides a comprehensive understanding of the price dynamics, allowing for forecasting future prices and analyzing historical trends. The results highlight periods of volatility and stability in food prices, offering valuable insights for further economic analysis and decision-making.
"""
)
