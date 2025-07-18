# ev_forecasting.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("D:/ASHWIN/Ashwin Programing/IBM Edunet July 2025/EV_Forecasting/Electric_Vehicle_Population_Size_History_By_County_.csv")


# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df[df['Date'].notnull()]

# Fill missing values
df['County'] = df['County'].fillna('Unknown')
df['State'] = df['State'].fillna('Unknown')

# Remove commas and convert numeric columns to int
columns_to_clean = [
    'Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Electric Vehicle (EV) Total', 'Non-Electric Vehicle Total', 'Total Vehicles'
]
for col in columns_to_clean:
    df[col] = df[col].astype(str).str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Cap outliers in Percent Electric Vehicles
Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound,
                                np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound,
                                         df['Percent Electric Vehicles']))

# Feature Engineering: extract year & month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Encode categorical columns
le_state = LabelEncoder()
le_county = LabelEncoder()
le_use = LabelEncoder()

df['State'] = le_state.fit_transform(df['State'])
df['County'] = le_county.fit_transform(df['County'])
df['Vehicle Primary Use'] = le_use.fit_transform(df['Vehicle Primary Use'])

# Features & Target
features = [
    'Year', 'Month', 'County', 'State', 'Vehicle Primary Use',
    'Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Non-Electric Vehicle Total', 'Total Vehicles', 'Percent Electric Vehicles'
]
target = 'Electric Vehicle (EV) Total'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, 'ev_adoption_model.pkl')
print("Model saved as 'ev_adoption_model.pkl'")


plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual EV Total")
plt.ylabel("Predicted EV Total")
plt.title("Actual vs Predicted Electric Vehicle Total")
plt.grid(True)
plt.show()
