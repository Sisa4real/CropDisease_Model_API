from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load the Keras model
model = load_model("Crop_Disease_model_98_Acc.h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()

        # Ensure the image data is provided
        if 'image' not in data:
            return jsonify({"error": "Missing 'image' key in the request data"}), 400

        # Flattened image to numpy array
        flattened_image = np.array(data['image'], dtype=np.float32)

        # Reshape the image into (224, 224, 3)
        if flattened_image.size != 224 * 224 * 3:
            return jsonify({"error": f"Invalid input size. Expected {224 * 224 * 3}, got {flattened_image.size}"}), 400

        image = np.reshape(flattened_image, (224, 224, 3))

        # Add batch dimension to make it (1, 224, 224, 3)
        image = np.expand_dims(image, axis=0)

        # Preprocess input for MobileNetV2
        image = preprocess_input(image)

        # Perform prediction
        predictions = model.predict(image)

        # Return predictions
        return jsonify({"prediction": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
