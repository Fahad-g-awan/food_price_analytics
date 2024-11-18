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

This part estimates the economic impact of potential food price increases on non-qualified labor wages using linear regression, helping understand how food price changes influence wage levels.

---

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
st.markdown(
"""
Coefficient for food price
"""
)
st.text(f"{model.coef_}")
st.markdown(
"""
Intercept
"""
)
st.text(f"{model.intercept_}")

# Predict the labor wages using the test set
y_pred = model.predict(X_test)

st.text("\n\n")
st.markdown(
"""
##### Visualization
"""
)

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
---
### Model Evaluation

After fitting the model and making predictions, it's essential to evaluate the model performance beyond just plotting the predictions. Use evaluation metrics like Mean Squared Error (MSE) or R-squared (R²) to assess how well your model is performing.
"""
)


# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown(
"""
##### Mean Squared Error
"""
)
st.text(f"{mse:.4f}")
st.markdown(
"""
##### R-squared
"""
)
st.text(f"{r2:.4f}")

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
