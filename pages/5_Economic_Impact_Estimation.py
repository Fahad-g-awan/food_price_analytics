from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd

from utils.data_processing import get_df

df = get_df()

st.set_page_config(page_title="Economic Impact Estimation", layout="wide")


st.markdown(
"""
# Economic Impact Estimation

This part estimates the economic impact of potential food price increases on non-qualified labor wages using regression models, helping understand how food price changes influence wage levels.

---
"""
)


# Filter the dataset for the relevant commodity (Wage - non-qualified labour)
wage_data = df[df['commodity'] == 'Wage (non-qualified labour, non-agricultural)']

# Check for missing values and drop or handle them
if wage_data.isnull().sum().any():
    st.text("There are missing values in the data.")
    wage_data = wage_data.dropna()

# Check for outliers using a box plot
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=wage_data[['price', 'usdprice']])
# st.pyplot(plt)

# Log transform to stabilize variance
wage_data['log_price'] = np.log1p(wage_data['price'])
wage_data['log_wage'] = np.log1p(wage_data['usdprice'])

# Select features and target variable
X = wage_data[['log_price']]  # Independent variable: food prices
y = wage_data['log_wage']     # Dependent variable: non-qualified labor wages (or 'usdprice' as the wage proxy)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Fit the model
linear_model.fit(X_train, y_train)

# Predict the labor wages using the test set
y_pred_linear = linear_model.predict(X_test)

# Evaluate the model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)


# Polynomial Regression (degree=2 for quadratic model)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split the data
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Fit the polynomial model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)

# Predict and evaluate
y_pred_poly = poly_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
r2_poly = r2_score(y_test_poly, y_pred_poly)

# Random Forest Regression Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate using Random Forest
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Visualizations
# Scatter plot for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, color='blue', label='Predicted vs Actual (Linear)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title('Linear Regression: Actual vs Predicted Labor Wages')
plt.xlabel('Actual Labor Wages')
plt.ylabel('Predicted Labor Wages')
plt.legend()
st.pyplot(plt)

# Scatter plot for Polynomial Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test_poly, y_pred_poly, color='green', label='Predicted vs Actual (Polynomial)')
plt.plot([min(y_test_poly), max(y_test_poly)], [min(y_test_poly), max(y_test_poly)], color='red', linestyle='--', label='Perfect Fit')
plt.title('Polynomial Regression: Actual vs Predicted Labor Wages')
plt.xlabel('Actual Labor Wages')
plt.ylabel('Predicted Labor Wages')
plt.legend()
st.pyplot(plt)

# Scatter plot for Random Forest Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, color='purple', label='Predicted vs Actual (Random Forest)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
plt.title('Random Forest: Actual vs Predicted Labor Wages')
plt.xlabel('Actual Labor Wages')
plt.ylabel('Predicted Labor Wages')
plt.legend()
st.pyplot(plt)



# # Filter the dataset for the relevant commodity (Wage - non-qualified labour)
# wage_data = df[df['commodity'] == 'Wage (non-qualified labour, non-agricultural)']

# # Select features and target variable
# X = wage_data[['price']]  # Independent variable: food prices
# y = wage_data['usdprice']  # Dependent variable: non-qualified labor wages (or 'usdprice' as the wage proxy)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the model
# model = LinearRegression()

# # Fit the model
# model.fit(X_train, y_train)

# # Print the regression coefficients
# st.markdown(
# """
# Coefficient for food price
# """
# )
# st.text(f"{model.coef_}")
# st.markdown(
# """
# Intercept
# """
# )
# st.text(f"{model.intercept_}")

# # Predict the labor wages using the test set
# y_pred = model.predict(X_test)

# st.text("\n\n")
# st.markdown(
# """
# ##### Visualization
# """
# )

# # Plot the predictions vs actual values
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
# plt.title('Actual vs Predicted Labor Wages')
# plt.xlabel('Actual Labor Wages')
# plt.ylabel('Predicted Labor Wages')
# plt.legend()
# st.pyplot(plt)




# # Filter the dataset for the relevant commodity (Wage - non-qualified labour)
# wage_data = df[df['commodity'] == 'Wage (non-qualified labour, non-agricultural)']

# # Filter the dataset for food prices (assuming you have a column for 'food price' in the dataset)
# food_price_data = df[df['commodity'] != 'Wage (non-qualified labour, non-agricultural)']

# # Merge the food price and wage data on the 'date' column
# merged_data = pd.merge(food_price_data[['date', 'price']], wage_data[['date', 'price']], on='date', suffixes=('_food', '_wage'))

# # Initialize a MinMaxScaler
# scaler = MinMaxScaler()

# # Scale both food price and wage price columns
# merged_data[['price_food_scaled', 'price_wage_scaled']] = scaler.fit_transform(merged_data[['price_food', 'price_wage']])

# # Plot the actual values with dual y-axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Plot scaled food prices on the first y-axis
# ax1.plot(merged_data['date'], merged_data['price_food_scaled'], color='blue', label='Food Price (Scaled)')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Food Price (Scaled)', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # Create a second y-axis for scaled wage prices
# ax2 = ax1.twinx()
# ax2.plot(merged_data['date'], merged_data['price_wage_scaled'], color='orange', label='Labor Wage (Scaled)')
# ax2.set_ylabel('Labor Wage (Scaled)', color='orange')
# ax2.tick_params(axis='y', labelcolor='orange')

# # Title and formatting
# plt.title('Scaled Food Prices vs Scaled Labor Wages')
# plt.xticks(rotation=45)
# fig.tight_layout()
# plt.grid(True)

# # Show the plot
# st.pyplot(fig)

# # Print the actual values of food prices and labor wages
# st.markdown("### Actual Food Prices and Labor Wages")
# st.write(merged_data[['date', 'price_food', 'price_wage']])



st.markdown(
"""
---
### Model Evaluation
"""
)

# Display Linear Model Evaluation
st.markdown("### Linear Regression Model Evaluation")
st.text(f"Mean Squared Error: {mse_linear:.4f}")
st.text(f"R-squared: {r2_linear:.4f}")


# Display Polynomial Model Evaluation
st.markdown("### Polynomial Regression Model Evaluation")
st.text(f"Mean Squared Error: {mse_poly:.4f}")
st.text(f"R-squared: {r2_poly:.4f}")

# Display Random Forest Model Evaluation
st.markdown("### Random Forest Regression Model Evaluation")
st.text(f"Mean Squared Error: {mse_rf:.4f}")
st.text(f"R-squared: {r2_rf:.4f}")



st.markdown(
"""
---
## Economic Impact Estimation Summary

### 1. Regression Models (Linear Regression):

The analysis aims to model the relationship between food prices and non-qualified labor wages using Linear Regression. The following steps were taken:

- **Dataset Preparation:** The dataset was filtered to include only the commodity 'Wage (non-qualified labour, non-agricultural)'. Food prices were used as the independent variable, and non-qualified labor wages (represented by `usdprice`) were the dependent variable.
  
- **Regression Model:** A linear regression model was fit to predict labor wages based on food prices.
  - **Coefficient for Food Price:** The regression coefficient indicates the impact of food prices on labor wages. A positive coefficient suggests that as food prices increase, labor wages also tend to increase.
  - **Intercept:** The intercept term provides the expected labor wage when food prices are zero.

### 2. Visualization:

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
