from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = load_model("leaf_classifier_model.h5")

# Load class labels from your dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    r"D:\Project\Leaf Class\dataset\test",  # Path to your test folder
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
class_labels = list(test_gen.class_indices.keys())

CONFIDENCE_THRESHOLD = 0.70  # Set your threshold

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', label="No file uploaded", img_path=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', label="No selected file", img_path=None)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # --- Preprocess the image ---
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # --- Predict ---
        prediction = model.predict(img_array)[0]
        top_idx = np.argmax(prediction)
        confidence = prediction[top_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            label = f"❌ This leaf is unknown to me (Confidence: {confidence*100:.2f}%)"
        else:
            label = f"🪴 Predicted: {class_labels[top_idx]} ({confidence*100:.2f}% confidence)"

        return render_template('result.html', label=label, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
