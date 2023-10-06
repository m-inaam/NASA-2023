import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('omni_rnn_0.h5')

# Create a scaler for input data
scaler = MinMaxScaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        input_data = data.get("input_data")

        # Ensure the input data is a list
        if not isinstance(input_data, list):
            return jsonify({"error": "Input data should be a list"}), 400

        # Ensure the input data has the correct number of features
        if len(input_data) != 3:  # Adjust the number of features as needed
            return jsonify({"error": "Input data should have 3 features"}), 400

        # Convert input data to a NumPy array and scale it
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        # Inverse transform the scaled predictions to the original scale
        predictions = scaler.inverse_transform(predictions)

        # Format the predictions as a dictionary
        result = {
            "predicted_speed": float(predictions[0, 0]),
            "predicted_field_magnitude": float(predictions[0, 1])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
