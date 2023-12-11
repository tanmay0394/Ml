import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Read the dataset
df = pd.read_csv("/content/iris.csv")

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.drop(columns=['variety'])

# Fill missing values with mean or other strategies
df_filled = df.fillna(df.mean())

# Check for duplicates
print(df.duplicated().sum())

# Remove duplicates
df_no_duplicates = df.drop_duplicates()

# Check data types
print(df_cleaned.dtypes)

# Data Analysis
print(df_cleaned.describe())

# Calculate correlation matrix
correlation_matrix = df_cleaned.corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, cmap='viridis')
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()

# Perform t-test or other statistical tests
# Note: Replace 'group1' and 'group2' with your actual column names
t_stat, p_value = stats.ttest_ind(df_cleaned['sepal.length'], df_cleaned['sepal.width'])
print(f'T-statistic: {t_stat}, p-value: {p_value}')

# Plot histogram
plt.hist(df_cleaned['sepal.length'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of sepal.length')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.scatter(df['petal.length'], df['petal.width'], color='red', alpha=0.5)
plt.title('Scatter Plot between petal.length and petal.width')
plt.xlabel('petal.length')
plt.ylabel('petal.width')
plt.show()


#Theory :-


# Certainly! Data cleaning is a crucial step in the data preparation process that involves identifying and correcting errors, inconsistencies, and inaccuracies in a dataset. The quality of the data greatly influences the results of any analysis or machine learning model, so it's essential to ensure that the data is accurate, reliable, and ready for further processing.

# Here's a more detailed theoretical overview of the key aspects of data cleaning:

# ### 1. **Data Quality Issues:**
#    - **Missing Values:** Data may have missing values, which can affect the quality of analysis. Strategies for handling missing values include removal, imputation, or using advanced techniques based on the nature of the data.
   
#    - **Duplicates:** Duplicate records can distort analysis results. Identifying and removing duplicates ensures that each observation is unique.

#    - **Inconsistent Data:** Inconsistencies in data formats, units, or representations can lead to errors. Standardizing data formats ensures consistency.

#    - **Outliers:** Outliers, or extreme values, can skew statistical analyses. Detecting and handling outliers is crucial for maintaining data integrity.

# ### 2. **Data Cleaning Techniques:**
#    - **Handling Missing Values:**
#      - Imputation: Fill missing values using statistical measures like mean, median, or mode.
#      - Removal: Eliminate rows or columns with missing values.

#    - **Dealing with Duplicates:**
#      - Identify and remove duplicate records based on unique identifiers.

#    - **Correcting Data Types:**
#      - Ensure that data types match the nature of the information. For example, dates should be in datetime format, and numerical values should have appropriate types.

#    - **Outliers Detection and Handling:**
#      - Use statistical methods like z-scores or interquartile range (IQR) to identify outliers.
#      - Decide whether to remove outliers or transform them based on the context.

#    - **Text Cleaning:**
#      - Standardize text data by converting to lowercase.
#      - Remove unnecessary whitespaces, special characters, or symbols.

#    - **Handling Inconsistent Data:**
#      - Standardize data formats and units to ensure consistency.

# ### 3. **Tools for Data Cleaning:**
#    - **Pandas:** A powerful library for data manipulation and analysis in Python. It provides functions for handling missing data, duplicates, and more.

#    - **NumPy:** A library for numerical operations in Python, useful for mathematical operations and handling numerical data.

#    - **Scikit-learn:** Offers tools for machine learning, including preprocessing techniques that can be useful for outlier detection and imputation.

#    - **Matplotlib and Seaborn:** Useful for data visualization, aiding in the identification of outliers and patterns.

# ### 4. **Best Practices:**
#    - **Explore Data:** Understand the characteristics of the data using descriptive statistics and visualizations before cleaning.

#    - **Document Changes:** Keep a record of the changes made during the cleaning process for transparency and reproducibility.

#    - **Iterative Process:** Data cleaning is often an iterative process. After an initial cleaning, further issues may be discovered during analysis.

# ### 5. **Conclusion:**
#    - Data cleaning is an essential step in the data science workflow, ensuring that data is reliable and suitable for analysis.

# Remember that the specific steps and techniques used for data cleaning may vary based on the characteristics of the dataset and the goals of the analysis. Always tailor the data cleaning process to the specific requirements of your project.