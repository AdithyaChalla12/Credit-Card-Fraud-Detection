from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the Best ML Model and Scaler
with open("best_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Fraud detection threshold (lowered for better sensitivity)
FRAUD_THRESHOLD = 0.3  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert input data into model-friendly format
        input_data = np.array([[data["Age"], data["Transaction_Amount"], data["Account_Balance"]]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        print("Scaled Input:", input_scaled)  # Debugging
        
        # Get fraud probability
        fraud_probability = model.predict_proba(input_scaled)[0][1]  # Probability of fraud
        print(f"Fraud Probability: {fraud_probability:.4f}")  # Debugging
        
        # Determine fraud status based on threshold
        is_fraud = fraud_probability > FRAUD_THRESHOLD

        return jsonify({
            "fraud": bool(is_fraud),
            "fraud_probability": round(fraud_probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
