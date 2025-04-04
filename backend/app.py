from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
import psycopg2
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# === Load Model Safely ===
MODEL_PATH = "/Users/dayanandks/Desktop/final_proj_1/backend/models/best_intrusion_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

# === Connect to PostgreSQL Safely ===
try:
    conn = psycopg2.connect(
        database="network_db",
        user="your_user",
        password="1234",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL successfully!")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit(1)

# === Create Table if Not Exists ===


# === Load Scaler (Ensure it is Fitted) ===
# Load training data to fit the scaler (Ensure consistency)
train_data = pd.read_csv('/Users/dayanandks/Desktop/final_proj_1/backend/archive/Train_data.csv')

# Select the same features used in training
#selected_features = ['feature1', 'feature2', 'feature3']  # Replace with actual features
#scaler = StandardScaler()
#scaler.fit(train_data[selected_features])  # Fit the scaler on training data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Standardize the input features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]  # 0 = Normal, 1 = Anomaly
        result = "anomaly" if prediction == 1 else "normal"

        # Save to PostgreSQL
        cursor.execute(
            "INSERT INTO anomaly_logs (features, prediction) VALUES (%s, %s)", 
            (str(data["features"]), result)
        )
        conn.commit()

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    try:
        cursor.execute("SELECT * FROM anomaly_logs")
        logs = cursor.fetchall()
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
