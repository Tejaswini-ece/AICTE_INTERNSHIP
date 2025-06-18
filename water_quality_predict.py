import pandas as pd # data manipulation
import numpy as np # numerical python - linear algebra

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
df

# Print dataset info
print(" Dataset Info:")
print(df.info())
print("\n")

# Print shape of dataset
print(" Dataset Shape (rows, columns):", df.shape)
print("\n")

# Print summary statistics
print(" Summary Statistics:")
print(df.describe().T)
print("\n")

# Print missing values count
print(" Missing Values Per Column:")
print(df.isnull().sum())
print("\n")

# Convert 'date' column to datetime 
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df

# Print info again to confirm 'date' conversion
print(" Dataset Info After Date Conversion:")
print(df.info())
print("\n")

# Sort by id and date
df = df.sort_values(by=['id', 'date'])

# Print first few rows
print(" First 5 Rows After Sorting by id and date:")
print(df.head())
print("\n")

# Extract year and month
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Print first few rows again
print(" First 5 Rows After Adding 'year' and 'month':")
print(df.head())
print("\n")

# Print all column names
print(" Column Names:")
print(df.columns)
