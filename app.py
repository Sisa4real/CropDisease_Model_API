from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="CropDisease_model3.tflite")
interpreter.allocate_tensors()

# Get input and output details from the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

        # Reshape the image into (128, 128, 3)
        if flattened_image.size != 128 * 128 * 3:
            return jsonify({"error": f"Invalid input size. Expected 49152, got {flattened_image.size}"}), 400

        image = np.reshape(flattened_image, (128, 128, 3))

        # Add batch dimension to make it (1, 128, 128, 3)
        image = np.expand_dims(image, axis=0)

        # Preprocess input
        image = preprocess_input(image)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # Return predictions
        return jsonify({"prediction": output.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
