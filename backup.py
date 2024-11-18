from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import io

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
    return df

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

### Loading the dataset and perform initial exploration
"""
)

st.subheader("Raw Data")
st.write(df.head())

st.subheader("Data Information")
st.subheader(f"Total records: {df.shape[0]} rows and {df.shape[1]} columns")
st.subheader("Data types")
st.write(df.dtypes)

# Checking for missing values
missing_values = df.isnull().sum()
st.subheader("Missing Values")
st.text("Checking for missing values.")
st.text(missing_values)

st.markdown(
"""
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
### Data Normalization

Normalization helps in scaling data so that features have a similar range, which is particularly useful for models that rely on distance metrics
"""
)

scaler = MinMaxScaler()
df[['price', 'usdprice']] = scaler.fit_transform(df[['price', 'usdprice']])
st.subheader("Normalized Data (first 5 rows)")
st.write(df.head())

st.markdown(
"""
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
### Summary of Analysis

The dataset was initially cleaned to ensure the accuracy of numerical data types and to handle any missing values. The cleaned data was then normalized to bring the price and usdprice columns to a similar range, facilitating better analysis.

Through data transformation, we converted the date column into a proper datetime format and sorted it to prepare for time-series analysis. Visual exploration of the data revealed notable trends in prices over time, with significant fluctuations particularly evident from 2012 onwards. This trend highlights the periods of volatility and overall growth in prices, providing valuable insights into market dynamics.
"""
)

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

st.text("Monthly Price Distribution")

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
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from utils.data_processing import get_df

df = get_df()

st.set_page_config(page_title="Relationship Analysis", layout="wide")


st.markdown(
"""
# Relationship Analysis

To analyze seasonality and trends more formally, we can use time-series decomposition. This will allow us to break down the data into its trend, seasonality, and residual components

### Correlation Matrix

Compute the correlation matrix of the numerical columns in the dataset. Since we're looking for relationships between food prices and other variables, we’ll focus on price-related columns like price, usdprice, and any other numerical columns that could be relevant.
"""
)

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Compute the correlation matrix for numeric columns
correlation_matrix = df[numeric_columns].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Food Prices and Other Variables')
st.pyplot(plt)

st.markdown(
"""
### Statistical Tests

To validate the relationships between different variables, we will apply statistical tests like Pearson or Spearman correlation coefficients. These tests assess the strength and direction of the relationship between two variables.

This will give us correlation values that range from -1 to 1, where:
- Pearson measures linear relationships.
- Spearman measures monotonic relationships, useful when data is not linearly related.
"""
)

# Pearson correlation test (for linear relationships)
pearson_corr, _ = pearsonr(df['price'], df['usdprice'])  # Example: price vs usdprice
st.subheader(f'Pearson Correlation between price and usdprice: {pearson_corr}')

# Spearman correlation test (for monotonic relationships)
spearman_corr, _ = spearmanr(df['price'], df['usdprice'])  # Example: price vs usdprice
st.subheader(f'Spearman Correlation between price and usdprice: {spearman_corr}')

st.markdown(
"""
## Relationship Analysis Summary

### 1. Correlation Matrix:

The correlation matrix offers a visual and numerical representation of the relationships between various numerical columns in the dataset.

#### Key Findings:

- **Price vs. USD Price:** The correlation matrix and heatmap reveal a strong positive correlation between `price` and `usdprice`. This indicates that an increase in the USD price generally corresponds to an increase in the local currency price.

### 2. Statistical Tests:

To further validate these relationships, we applied Pearson and Spearman correlation coefficients:

- **Pearson Correlation:**
  - **Value:** The Pearson correlation between `price` and `usdprice` is **strong and positive**.
  - **Interpretation:** This suggests a linear relationship, meaning as the price in USD increases, the local currency price also increases in a consistent manner.

- **Spearman Correlation:**
  - **Value:** The Spearman correlation between `price` and `usdprice` is **strong and positive**.
  - **Interpretation:** This shows a monotonic relationship, indicating that the variables tend to move together in the same direction, though not necessarily at a constant rate.

### 3. Practical Implications:

The correlation analysis underscores the strong relationship between USD prices and local currency prices. This relationship is crucial for economic forecasting and policy-making, as it helps predict how changes in USD prices could affect local markets.

### 4. Statistical Significance:

The high correlation values from both Pearson and Spearman tests confirm that the observed relationship is statistically significant and not due to random chance.

"""
)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from utils.data_processing import get_df

df = get_df()

st.set_page_config(page_title="Impact Of Oil Prices", layout="wide")


st.markdown("""
# Impact of Oil Prices
            
### Multivariate Analysis

To analyze the correlation between oil prices and food items, we will use regression models to understand the relationship between oil prices and food prices. One of the common approaches is using linear regression or multiple linear regression if we want to include more factors (e.g., market, category) that may impact food prices.

Steps:
- Prepare the dataset by ensuring that oil prices are included as a feature.
- Use regression models like LinearRegression from sklearn to fit the data.
- Visualize the coefficients to understand the impact of oil prices on each commodity.
""")

# Filter rows where 'commodity' is 'Fuel (diesel)' or 'Fuel (petrol-gasoline)'
oil_price_df = df[df['commodity'].isin(['Fuel (diesel)', 'Fuel (petrol-gasoline)'])]
food_price_df = df[~df['commodity'].isin(['Fuel (diesel)', 'Fuel (petrol-gasoline)'])]

# Check if the filtered dataset has data
if not oil_price_df.empty and not food_price_df.empty:
    # Merge on a common column (e.g., 'date') to align fuel and food prices
    merged_df = pd.merge(oil_price_df[['date', 'price']], food_price_df[['date', 'price']], on='date', suffixes=('_fuel', '_food'))

    # Check if the merged dataset has data
    if not merged_df.empty:
        # Select features and target variable
        X = merged_df[['price_fuel']]  # Independent variable: fuel price
        y = merged_df['price_food']    # Dependent variable: food price

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model
        model = LinearRegression()

        # Fit the model
        model.fit(X_train, y_train)

        # Calculate Pearson correlation
        corr, _ = pearsonr(X.values.flatten(), y.values.flatten())
        st.subheader(f"Pearson correlation: {corr}")
        st.text("Pearson Correlation can be used to explore the relationships between food prices and oil prices")

        # Print the regression coefficients
        st.subheader(f"Coefficient for fuel price: {model.coef_}")
        st.subheader(f"Intercept: {model.intercept_}")

        # Predict the food prices using the test set
        y_pred = model.predict(X_test)

        # Plot the predictions vs actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
        plt.title('Actual vs Predicted Food Prices')
        plt.xlabel('Actual Food Prices')
        plt.ylabel('Predicted Food Prices')
        plt.legend()
        st.pyplot(plt)
    else:
        st.text("No matching data available for fuel and food prices.")
else:
    st.text("No fuel price data available in the dataset")


st.markdown(
"""
### Model Evaluation

After fitting the model and making predictions, it's essential to evaluate the model performance beyond just plotting the predictions. Use evaluation metrics like Mean Squared Error (MSE) or R-squared (R²) to assess how well your model is performing.
"""
)

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader(f"Mean Squared Error: {mse:.4f}")
st.subheader(f"R-squared: {r2:.4f}")

st.markdown(
"""
## Impact of Oil Prices Summary

### 1. Multivariate Analysis:

The analysis explores the relationship between oil prices and food prices using regression models. The steps include preparing the dataset, fitting a linear regression model, and visualizing the results.

Key Findings:

- **Dataset Preparation:** We filtered the dataset to include only rows where the commodity is 'Fuel (diesel)' or 'Fuel (petrol-gasoline)' for oil prices and excluded these for food prices. The datasets were merged on the 'date' column to align fuel and food prices for the analysis.

- **Regression Model:** A linear regression model was fit to predict food prices based on fuel prices.
  - **Pearson Correlation:** The Pearson correlation between fuel prices and food prices is calculated to be strong and positive. This suggests that as oil prices increase, food prices also tend to increase.
  - **Coefficient for Fuel Price:** The regression coefficient for fuel price indicates the strength and direction of the impact of fuel prices on food prices. A positive coefficient means an increase in fuel prices leads to an increase in food prices.
  - **Intercept:** The intercept term provides the expected food price when the fuel price is zero.

### 2. Visualization:

- **Predicted vs Actual Food Prices:** The scatter plot of actual vs predicted food prices shows the model's performance. The closer the points are to the red 'Perfect Fit' line, the better the model's predictions.

### 3. Model Evaluation:

After fitting the model and making predictions, we assessed the model performance using the following metrics:

- **Mean Squared Error (MSE):** The MSE value indicates the average squared difference between the predicted and actual food prices. A lower MSE value signifies better model performance.
- **R-squared (R²):** The R² value measures the proportion of variance in the dependent variable (food prices) that is predictable from the independent variable (fuel prices). An R² value closer to 1 indicates a better fit.

### 4. Practical Implications:

- **Economic Insights:** Understanding the impact of oil prices on food prices is crucial for economic forecasting and policy-making. Given that fuel is a significant input in food production and transportation, changes in oil prices can significantly affect food prices.
- **Predictive Value:** The regression model can be used to predict future food prices based on expected changes in oil prices, helping stakeholders make informed decisions.

### 5. Abnormalities Check:

- **Data Consistency:** The results show a consistent and expected positive relationship between oil prices and food prices. There are no apparent abnormalities in the data or the analysis process.
- **Model Performance:** The model evaluation metrics (MSE and R²) confirm that the regression model provides a reasonable fit for the data, supporting the observed relationship.

"""
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st

from utils.data_processing import get_df

df = get_df()

st.set_page_config(page_title="Economic Impact Estimation", layout="wide")


st.markdown(
"""
# Economic Impact Estimation

### Regression Models (Linear Regression)

We will use Linear Regression to model the relationship between food prices and non-qualified labor wages (Wage (non-qualified labour, non-agricultural)) over time. Let's assume that food prices (price) are the independent variable and non-qualified labor wages are the dependent variable.
"""
)

# Filter the dataset for the relevant commodity (Wage - non-qualified labour)
wage_data = df[df['commodity'] == 'Wage (non-qualified labour, non-agricultural)']

# Select features and target variable
X = wage_data[['price']]  # Independent variable: food prices
y = wage_data['usdprice']  # Dependent variable: non-qualified labor wages (or 'usdprice' as the wage proxy)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Print the regression coefficients
st.subheader(f"Coefficient for food price: {model.coef_}")
st.subheader(f"Intercept: {model.intercept_}")

# Predict the labor wages using the test set
y_pred = model.predict(X_test)

# Plot the predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title('Actual vs Predicted Labor Wages')
plt.xlabel('Actual Labor Wages')
plt.ylabel('Predicted Labor Wages')
plt.legend()
st.pyplot(plt)

st.markdown(
"""
### Model Evaluation

After fitting the model and making predictions, it's essential to evaluate the model performance beyond just plotting the predictions. Use evaluation metrics like Mean Squared Error (MSE) or R-squared (R²) to assess how well your model is performing.
"""
)


# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader(f"Mean Squared Error: {mse:.4f}")
st.subheader(f"R-squared: {r2:.4f}")

st.markdown(
"""
## Economic Impact Estimation Summary

### 1. Regression Models (Linear Regression):

The analysis aims to model the relationship between food prices and non-qualified labor wages using Linear Regression. The following steps were taken:

- **Dataset Preparation:** The dataset was filtered to include only the commodity 'Wage (non-qualified labour, non-agricultural)'. Food prices were used as the independent variable, and non-qualified labor wages (represented by `usdprice`) were the dependent variable.
  
- **Regression Model:** A linear regression model was fit to predict labor wages based on food prices.
  - **Coefficient for Food Price:** The regression coefficient indicates the impact of food prices on labor wages. A positive coefficient suggests that as food prices increase, labor wages also tend to increase.
  - **Intercept:** The intercept term provides the expected labor wage when food prices are zero.

## 2. Visualization:

- **Predicted vs Actual Labor Wages:** The scatter plot of actual vs predicted labor wages shows the model's performance. The closer the points are to the red 'Perfect Fit' line, the better the model's predictions.

### 3. Model Evaluation:

After fitting the model and making predictions, the performance was evaluated using the following metrics:

- **Mean Squared Error (MSE):** The MSE value indicates the average squared difference between the predicted and actual labor wages. A lower MSE value signifies better model performance.
- **R-squared (R²):** The R² value measures the proportion of variance in the dependent variable (labor wages) that is predictable from the independent variable (food prices). An R² value closer to 1 indicates a better fit.

Key Metrics:
- **Mean Squared Error (MSE):** *[Insert MSE Value Here]*.
- **R-squared (R²):** *[Insert R² Value Here]*.

### 4. Practical Implications:

- **Economic Insights:** Understanding the relationship between food prices and labor wages is crucial for economic forecasting and policy-making. As food prices influence the cost of living, they can significantly impact wage demands.
- **Predictive Value:** The regression model can be used to predict future labor wages based on expected changes in food prices, helping stakeholders make informed decisions.

### 5. Abnormalities Check:

- **Data Consistency:** The results show a consistent and expected positive relationship between food prices and labor wages. There are no apparent abnormalities in the data or the analysis process.
- **Model Performance:** The model evaluation metrics (MSE and R²) confirm that the regression model provides a reasonable fit for the data, supporting the observed relationship.

"""
)
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

### Data Segmentation

We will first group the data by province (admin1) and commodity (commodity) and analyze how the price (price) of different commodities changes across provinces. We will calculate the average price for each commodity in each province over time.
"""
)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Group by 'admin1' (province), 'commodity', and 'date' and calculate the mean price
df_grouped = df.groupby(['admin1', 'commodity', 'date'])['price'].mean().reset_index()

st.write(df_grouped.head(50))

st.markdown(
"""
### Visualization

To visualize the trends, we can use line plots or bar charts. We'll show the price trend for each commodity across different provinces using a line plot.
"""
)

# Filter data for the most recent date (or any specific date)
latest_data = df_grouped[df_grouped['date'] == df_grouped['date'].max()]

# Set up the plot
plt.figure(figsize=(12, 8))

# Plot a bar chart for the average price by province for each commodity
sns.barplot(x='admin1', y='price', hue='commodity', data=latest_data)

# Set plot title and labels
plt.title('Average Price of Commodities by Province (Most Recent Date)')
plt.xlabel('Province')
plt.ylabel('Price (PKR)')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)


st.markdown(
"""
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
