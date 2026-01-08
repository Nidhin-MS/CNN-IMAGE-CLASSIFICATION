
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("cnn_model.h5")

class_names = ["cats", "dogs"]

def preprocess_image(image):
    IMG_SIZE =(64,64)
    BATCH_SIZE =32
    image_size=IMG_SIZE
    batch_size=BATCH_SIZE
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    image = preprocess_image(image)

    preds = model.predict(image)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "prediction": class_names[class_index],
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
