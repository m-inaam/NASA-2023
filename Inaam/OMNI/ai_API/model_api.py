import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow import keras

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

# Enable CORS for all routes
CORS(app, resources={r'*': {'origins': '*'}})

# Load the pre-trained model
model = keras.models.load_model('omni_rnn_0.h5')

@app.route('/')
def index():
    return render_template('index.html')

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
        if len(input_data) != 3:
            return jsonify({"error": "Input data should have 3 features"}), 400

        # Convert input data to a NumPy array
        input_data = np.array(input_data).reshape(1, -1)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

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
