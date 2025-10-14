from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

model = load_model("mobilenetv2_leaf_classifier2.h5")

class_labels = [
    'Artocarpus heterophyllus', 'Bambusa vulgaris', 'Coccinia grandis',
    'Codiaeum variegatum', 'Ficus benghalensis', 'Foliorum Forma - Orbicularis',
    'Herbarium Oxalis', 'Hevea brasiliensis', 'Indonesian Bay',
    'Lembu', 'Mangifera indica', 'Psidium guajava'
]

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # shows upload page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # If confidence too low, mark as unknown
    threshold = 0.6
    if confidence < threshold:
        predicted_label = "Unknown Leaf"

    return render_template(
        'result.html',
        leaf_name=predicted_label,
        confidence=f"{confidence*100:.2f}%",
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)
