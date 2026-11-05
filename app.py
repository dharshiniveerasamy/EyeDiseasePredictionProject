from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("model/eye_disease_model.keras")

class_names = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No image uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "Please select an image"

    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    img = image.load_img(
        file_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        image_path=file_path
    )


if __name__ == "__main__":
    app.run(debug=True)