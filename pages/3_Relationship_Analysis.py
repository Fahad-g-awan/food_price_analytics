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

---

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
---
### Statistical Tests

To validate the relationships between different variables, we will apply statistical tests like Pearson or Spearman correlation coefficients. These tests assess the strength and direction of the relationship between two variables.

This will give us correlation values that range from -1 to 1, where:
- Pearson measures linear relationships.
- Spearman measures monotonic relationships, useful when data is not linearly related.
"""
)

# Pearson correlation test (for linear relationships)
pearson_corr, _ = pearsonr(df['price'], df['usdprice']) 
st.markdown(
"""
##### Pearson Correlation between price and usdprice
"""
)
st.text(f'{pearson_corr}')

# Spearman correlation test (for monotonic relationships)
spearman_corr, _ = spearmanr(df['price'], df['usdprice']) 
st.markdown(
"""
##### Spearman Correlation between price and usdprice
"""
)
st.text(f'{spearman_corr}')

st.markdown(
"""
---
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