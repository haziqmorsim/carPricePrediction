from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        make = request.form['make']
        model_name = request.form['model']
        year = int(request.form['year'])
        transmission = request.form['transmission']
        fuel_type = request.form['fuel_type']
        mileage = float(request.form['mileage'])
        engine_size = float(request.form['engine_size'])
        num_owners = request.form['num_owners']
        accident_history = request.form['accident_history']
        road_tax_status = request.form['road_tax_status']
        car_condition = request.form['car_condition']
        service_score = int(request.form['service_score'])

        # Calculate derived features
        car_age = 2025 - year
        mileage_per_year = mileage / car_age
        engine_performance = engine_size / car_age

        # Create input array in order of model training
        input_data = pd.DataFrame([{
            'Make': make,
            'Model': model_name,
            'Year': year,
            'Transmission': transmission,
            'Fuel_Type': fuel_type,
            'Mileage': mileage,
            'Engine_Size': engine_size,
            'Num_Owners': num_owners,
            'Accident_History': accident_history,
            'Road_Tax_Status': road_tax_status,
            'Car_Condition': car_condition,
            'Service_Score': service_score,
            'Car_Age': car_age,
            'Mileage_per_Year': mileage_per_year,
            'Engine_Performance': engine_performance
        }])

        # Predict price
        predicted_price = model.predict(input_data)[0]
        predicted_price = round(predicted_price, 2)

        return render_template('index.html', prediction_text=f"Estimated Resale Price: RM {predicted_price:.2f}")

# Run app
if __name__ == '__main__':
    app.run(debug=True)