from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("model.h5")

# Check if the 'temp' folder exists, if not create it
if not os.path.exists("temp"):
    os.makedirs("temp")

# Helper function to validate the file type
def allowed_file(filename):
    allowed_extensions = {"jpg", "jpeg", "png"}
    return filename.lower().split(".")[-1] in allowed_extensions

# Default route (root)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = request.files["image"]

    if img.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(img.filename):
        return jsonify({"error": "Invalid file type. Only .jpg, .jpeg, .png files are allowed."}), 400

    try:
        # Save the image temporarily
        img_path = os.path.join("temp", img.filename)
        img.save(img_path)

        # Preprocess the image
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Make the prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Remove the temporary image after prediction
        os.remove(img_path)

        # Render the result on the HTML page
        return render_template("index.html", prediction=int(predicted_class[0]))

    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
