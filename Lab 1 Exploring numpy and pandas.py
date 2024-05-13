#lab 1 experiment to explore numpy and pandas and the task is gven below
#Exploring Pandas
#Task 1
import pandas as pd

#Task 1.1, read csv file 
df = pd.read_csv(r"D:\Dataset\Salary_Data - Copy.csv")
#Task 1.2, Display the first 5 rows 
print("First 5 rows of the dataset:")
print(df.head())
print("\nChecking for missing values:") # Check for missing values
print(df.isnull().sum())
df['Salary'] = df['Salary'].astype(float) # Handle missing values (replace None with NaN)
print("\nSummary of the dataset:") # Summary of the dataset
print(df.describe())

#task 1.3, Select a subset of columns using label-based indexing
label_based_subset = df[['employee_name', 'Salary']]
print("\nSubset of columns using label-based indexing:")
print(label_based_subset)

# Select a subset of columns using position-based indexing
position_based_subset = df.iloc[:, [0, 1]]
print("\nSubset of columns using position-based indexing:")
print(position_based_subset)

# Create a new DataFrame by filtering rows based on a condition (e.g., Salary greater than 50,000)
filtered_df = df[df['Salary'] > 60000]
print("\nFiltered DataFrame where Salary is greater than 50,000:")
print(filtered_df)

#Task 2
# Identify missing values(Task 2.1)
missing_values = df['Salary'].isnull().sum()
print(f"Number of missing values in 'Salary': {missing_values}")

# Impute missing values with the mean
mean_salary = df['Salary'].mean()
df['Salary'].fillna(mean_salary, inplace=True)

# Display the DataFrame after imputation
print("\nDataFrame after imputation:")
print(df)

#task 2.2, Create a new column by applying a mathematical operation (e.g., multiplying 'Salary' by 1.1)
df['Salary_increase'] = df['Salary'] * 1.1

# Convert categorical variable 'department' into numerical representation using one-hot encoding
one_hot_encoded = pd.get_dummies(df['department'], prefix='department')

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

# Display the modified DataFrame
print("DataFrame after creating a new column and one-hot encoding:")
print(df)

#task 2.3, Convert 'Salary' column to numeric
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# Group the data by 'department'
grouped_data = df.groupby('department')

# Apply aggregation functions to the grouped data
agg_results = grouped_data.agg({
    'Salary': ['sum', 'mean', 'count'],
    'employee_name': 'count'
})

# Present the results
print("Aggregated results grouped by 'department':")
print(agg_results)

#task 3, Merge two datasets and analyze
df1 = pd.read_csv(r"D:\Dataset\data1.csv")
df2 = pd.read_csv(r"D:\Dataset\data2.csv")

# Inner Join
inner_join = pd.merge(df1, df2, on='ID', how='inner')

# Left Join
left_join = pd.merge(df1, df2, on='ID', how='left')

# Right Join
right_join = pd.merge(df1, df2, on='ID', how='right')

# Outer Join
outer_join = pd.merge(df1, df2, on='ID', how='outer')

# Display the results
print("Original DataFrame 1:")
print(df1)
print("\nOriginal DataFrame 2:")
print(df2)

print("\nInner Join:")
print(inner_join)

print("\nLeft Join:")
print(left_join)

print("\nRight Join:")
print(right_join)

print("\nOuter Join:")
print(outer_join)

#Task 4, Visualization
#task 4.1 Bar, line and scatter plot
import matplotlib.pyplot as plt
df3 = pd.read_csv(r"D:\Dataset\Salary_Data - Copy.csv")
# Convert 'Salary' column to numeric
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# Bar Plot: Number of employees in each department
bar_plot_data = df['department'].value_counts()
bar_plot_data.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Employees in Each Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.show()

# Line Plot: Salary trends over employee index
line_plot_data = df['Salary']
line_plot_data.plot(kind='line', marker='o', linestyle='-', color='orange')
plt.title('Salary Trends')
plt.xlabel('Employee Index')
plt.ylabel('Salary')
plt.show()

# Scatter Plot: Relationship between Salary and Employee Index
scatter_plot_data = df[['Salary']].reset_index()
scatter_plot_data.plot(kind='scatter', x='index', y='Salary', color='green', marker='o')
plt.title('Relationship between Salary and Employee Index')
plt.xlabel('Employee Index')
plt.ylabel('Salary')
plt.show()

#task 4.2 correlation matrix of numerical coloumns
import seaborn as sns 
# Create a subset of numerical columns for correlation matrix
numerical_cols = ['Salary']

# Calculate the correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Create a heatmap and highlight highly correlated features
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

#Task 4.3 Histogram
# Create histograms for numerical columns
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Salary'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of Salary')

# Create box plots for numerical columns
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Salary'], color='lightcoral')
plt.title('Box Plot of Salary')

plt.tight_layout()
plt.show()


#Exploring Numpy
import numpy as np

#task 5, basic numpy operations
arr = np.arange(1, 11)
arr2 = np.arange(11, 21)

# Perform operations
addition_result = arr + arr2
subtraction_result = arr - arr2
multiplication_result = arr * arr2
division_result = arr / arr2

# Print results
print("Array 1 (arr):", arr)
print("Array 2 (arr2):", arr2)

print("\nAddition Result:", addition_result)
print("Subtraction Result:", subtraction_result)
print("Multiplication Result:", multiplication_result)
print("Division Result:", division_result)

#Task 6, Array manipulation
# Reshape 'arr' into a 2x5 matrix
arr_reshaped = arr.reshape(2, 5)

# Transpose the matrix obtained in the previous step
arr_transposed = arr_reshaped.T

# Flatten the transposed matrix into a 1D array
flattened_array = arr_transposed.flatten()

# Stack 'arr' and 'arr2' vertically
stacked_arrays = np.vstack((arr, arr2))

# Print the results
print("Original Array (arr):", arr)
print("Reshaped Array (2x5):")
print(arr_reshaped)
print("\nTransposed Array:")
print(arr_transposed)
print("\nFlattened Array:")
print(flattened_array)
print("\nVertically Stacked Arrays:")
print(stacked_arrays)

# Task 7: Statistical Operations
arr = np.arange(1, 11)

# Calculate mean, median, and standard deviation of 'arr'
mean_value = np.mean(arr)
median_value = np.median(arr)
std_deviation = np.std(arr)

# Find maximum and minimum values in 'arr'
max_value = np.max(arr)
min_value = np.min(arr)

# Normalize 'arr'
normalized_arr = (arr - mean_value) / std_deviation

# Print results for Task 7
print("\nTask 7: Statistical Operations")
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_deviation)
print("Maximum Value:", max_value)
print("Minimum Value:", min_value)
print("Normalized Array:", normalized_arr)

# Task 8: Boolean Indexing
bool_arr = arr > 5
filtered_arr = arr[bool_arr]

# Print results for Task 8
print("\nTask 8: Boolean Indexing")
print("Boolean Array:", bool_arr)
print("Filtered Array (elements greater than 5):", filtered_arr)

# Task 9: Random Module
random_matrix = np.random.rand(3, 3)
random_integers = np.random.randint(1, 100, 10)
np.random.shuffle(arr)

# Print results for Task 9
print("\nTask 9: Random Module")
print("Random 3x3 Matrix:", random_matrix)
print("Array of 10 Random Integers:", random_integers)
print("Shuffled 'arr':", arr)

# Task 10: Universal Functions (ufunc)
sqrt_arr = np.sqrt(arr)
exp_arr = np.exp(arr)

# Print results for Task 10
print("\nTask 10: Universal Functions (ufunc)")
print("Square Root of 'arr':", sqrt_arr)
print("Exponential of 'arr':", exp_arr)

# Task 11: Linear Algebra Operations
mat_a = np.random.rand(3, 3)
vec_b = np.random.rand(3, 1)
dot_product_result = np.dot(mat_a, vec_b)

# Print results for Task 11
print("\nTask 11: Linear Algebra Operations")
print("Matrix 'mat_a':\n", mat_a)
print("Vector 'vec_b':\n", vec_b)
print("Dot Product of 'mat_a' and 'vec_b':\n", dot_product_result)

# Task 12: Broadcasting
matrix = np.arange(1, 10).reshape(3, 3)
mean_row = matrix.mean(axis=1, keepdims=True)
broadcast_result = matrix - mean_row

# Print results for Task 12
print("\nTask 12: Broadcasting")
print("Original 2D Array 'matrix':\n", matrix)
print("Result after subtracting mean of each row:\n", broadcast_result)
