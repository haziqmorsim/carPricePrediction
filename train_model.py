import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv('used_cars.csv')

# Drop missing values
data.dropna(inplace=True)

# Label encode categorical columns
le = LabelEncoder()
data['Fuel_Type'] = le.fit_transform(data['Fuel_Type'])
data['Transmission'] = le.fit_transform(data['Transmission'])

# Features and target
X = data[['Year', 'Mileage', 'Fuel_Type', 'Transmission', 'Engine_Size']]
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Save model
joblib.dump(model, 'model.pkl')
