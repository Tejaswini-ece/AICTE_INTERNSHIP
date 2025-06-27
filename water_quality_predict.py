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
df.columns = df.columns.str.strip()

##WEEK 2

# Define the list of pollutant columns to be predicted
pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
 
#drop the missing values- (we can use dropna() function)
df=df.dropna(subset=pollutants)
df.head()

# missing values count
print("Missing Values Count")
print(df.isnull().sum())

# Feature and target Selection 
#Feature - independent variable ,Target - dependent Variable
x=df[['id','year']]
y=df[pollutants]

#Encoding -onehotencoder -22 stations
x_encoded=pd.get_dummies(x,columns=['id'],drop_first=True)

# Train ,Test and Split
x_train, x_test, y_train, y_test =train_test_split(
    x_encoded,y,test_size=0.2,random_state=42
)

# Train the model
model=MultiOutputRegressor(RandomForestRegressor(n_estimators=100,random_state=42))
model.fit(x_train,y_train)

#Evaluate model
y_pred=model.predict(x_test)

print("Model performance on the Test Data")
for i,pollutant in enumerate(pollutants):
    print(f'{pollutant}:')
    print('  MSE:',mean_squared_error(y_test.iloc[:,i],y_pred[:,i]))
    print('  R2:',r2_score(y_test.iloc[:,i],y_pred[:,i]))
    print()

station_id = '20'
year_input = 2022

input_data = pd.DataFrame({'year': [year_input], 'id':[station_id]})
input_encoded = pd.get_dummies(input_data, columns=['id'])


# Align with training feature columns
missing_cols = set(x_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
#reorder columns
input_encoded = input_encoded[x_encoded.columns]

# Predict Pollutants
predicted_pollutants = model.predict(input_encoded)[0]

print(f"\nPredicted pollutant levels for station '{station_id}'in {year_input}:")
for p, val in zip(pollutants,predicted_pollutants):
    print(f"  {p}: {val:.2f}")

import joblib
joblib.dump(model,'pollution_model.pkl')
joblib.dump(x_encoded.columns.tolist(),"model_columns.pkl")
print('Model and cols structure are saved!')

