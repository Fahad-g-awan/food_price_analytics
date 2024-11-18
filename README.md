# Food Prices Analysis in Pakistan

## Project Overview

The World Food Programme (WFP) is the world's largest humanitarian agency fighting hunger worldwide, delivering food assistance in emergencies and working with communities to improve nutrition and build resilience. In Pakistan, while the country has become a major producer of wheat and other food items, the National Nutrition Survey 2018 showed that 36.9 percent of the Pakistani population faces food insecurity. This is primarily due to limited economic access by the poorest and most vulnerable group of the population.

This analysis covers a comprehensive range of years as provided in the dataset, starting from 2004 to 2019. The dataset includes food prices for various commodities over multiple years, allowing us to analyze trends, relationships, and impacts across time.

The primary goal of this analysis is to identify trends, seasonality, and relationships between food prices and key factors such as oil prices. The project also explores the impact of external economic factors like oil prices on food costs and aims to provide insights that can help mitigate the economic impact of rising food prices on lower-income populations.

## Objectives

- **Data Analysis and Preprocessing:** Clean, normalize, and transform the data to ensure it is ready for analysis.
- **Trend Analysis:** Identify general trends in food prices over time using visual and statistical methods.
- **Impact of Oil Prices:** Analyze the effect of changes in oil prices on the prices of food items.
- **Economic Impact Estimation:** Estimate the economic impact of potential food price increases on non-qualified labor wages.
- **Comparative Analysis by Provinces:** Compare food prices across different provinces and identify regional trends.
- **Policy Recommendations:** Develop policies to mitigate the impact of potential food price increases according to food groups.

## Methodologies

The following models and techniques are used in this project:

- **Linear Regression**: A regression model used to analyze the relationship between oil prices and food prices. It helps in estimating how fluctuations in oil prices affect food prices.
- **ARIMA (AutoRegressive Integrated Moving Average)**: A time-series forecasting model applied to predict food price trends based on historical data. ARIMA is particularly useful for identifying patterns in data that vary over time, such as seasonal trends and long-term shifts in prices.
- **Statistical Tests**: Pearson and Spearman correlation tests are used to evaluate the strength and direction of relationships between numerical variables, helping to quantify the connection between food prices and other economic factors.

  ## Steps Involved

1. **Data Preprocessing**: The dataset is cleaned by handling missing values, converting data types, and normalizing numerical features.
2. **Trend and Seasonality Analysis**: Visualizing trends and performing seasonal decomposition to identify long-term patterns and seasonal fluctuations in food prices.
3. **Model Building and Evaluation**: Implementing regression models and ARIMA for forecasting, along with evaluating model performance using metrics like Mean Squared Error (MSE) and R-squared.

## Tools and Technologies

- **Python:** Primary programming language used for data analysis and modeling.
- **Streamlit:** Framework used to create interactive and user-friendly web applications.
- **Pandas, NumPy, Scikit-learn:** Libraries for data manipulation, analysis, and machine learning.
- **Matplotlib, Seaborn:** Libraries for creating static, animated, and interactive visualizations.

## Dataset

The dataset used for this project is the World Food Programme (WFP) Food Prices data for Pakistan, which includes prices for various foods such as maize, rice, wheat, beans, fish, and sugar, as well as non-food items like petrol/diesel prices and wages of non-qualified labor.

You can download the dataset from [WFP Food Prices Dataset](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan).

## Conclusion

Through comprehensive analysis and modeling, this project aims to provide valuable insights into food price trends in Pakistan and inform policy decisions to enhance food security and economic stability in the country.

---
