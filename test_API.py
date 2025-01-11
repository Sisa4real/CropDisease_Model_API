import requests
import numpy as np

# Generate a random dummy image of shape (128, 128, 3)
dummy_image = np.random.rand(128, 128, 3).astype(np.float32)

# Flatten the image for JSON serialization
flattened_image = dummy_image.flatten().tolist()

# URL of the running Flask app
url = "http://127.0.0.1:5000/predict"

# Send the POST request
response = requests.post(url, json={"image": flattened_image})

# Print the response
print("Response from API:", response.json())