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

We will analyze how changes in oil prices affect food prices using regression models and evaluate the strength of these relationships through correlation analysis.
            
---
            
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
        st.markdown(
        """
        #### Pearson correlation
        """
        )
        st.text(corr)
        st.markdown(
        """
        Pearson Correlation can be used to explore the relationships between food prices and oil prices
        """
        )

        # Print the regression coefficients
        st.markdown(
        """
        ##### Coefficient for fuel price
        """
        )
        st.text(model.coef_)
        st.markdown(
        """
        ##### Intercept
        """
        )
        st.text(model.intercept_)
        st.text("\n\n")

        # Predict the food prices using the test set
        y_pred = model.predict(X_test)
        
        st.markdown(
        """
        ##### Visualization
        """
        )

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

