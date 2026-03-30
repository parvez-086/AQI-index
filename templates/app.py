import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# --- STEP 1: TRAIN THE MODEL ON STARTUP ---
def train_model():
    # Use the filename directly (ensure the CSV is in your GitHub repo)
    csv_path = "AQI-and-Lat-Long-of-Countries.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return None

    data = pd.read_csv(csv_path)
    data = data.dropna()
    # Match the column cleaning from your training script
    data.columns = [col.strip().lower() for col in data.columns]

    X = data[['co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 aqi value']]
    y = data['aqi value']

    # We use RandomForest for better accuracy
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Model trained successfully on startup!")
    return model

# Train the model globally so it stays in memory
trained_model = train_model()

# --- STEP 2: WEB ROUTES ---

@app.route('/')
def home():
    # This looks for 'templates/index.html'
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if trained_model is None:
        return "Model not initialized. Check server logs."

    try:
        # Get values from your HTML form fields
        co = float(request.form['co'])
        ozone = float(request.form['ozone'])
        no2 = float(request.form['no2'])
        pm25 = float(request.form['pm25'])

        # Create a DataFrame that matches the training features exactly
        input_data = pd.DataFrame({
            'co aqi value': [co],
            'ozone aqi value': [ozone],
            'no2 aqi value': [no2],
            'pm2.5 aqi value': [pm25]
        })

        # Make the prediction
        prediction = trained_model.predict(input_data)
        final_result = round(prediction[0], 2)

        return render_template('index.html', result=final_result)
    
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)