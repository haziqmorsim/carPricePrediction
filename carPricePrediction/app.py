from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('models/car_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        mileage = float(request.form['mileage'])
        fuel_type = int(request.form['fuel_type'])  # assume encoded: 0 = Petrol, 1 = Diesel etc
        transmission = int(request.form['transmission'])  # assume encoded: 0 = Manual, 1 = Auto
        engine_size = float(request.form['engine_size'])

        features = np.array([[year, mileage, fuel_type, transmission, engine_size]])
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f"Estimated Car Price: RM {prediction[0]:,.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
