from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('facial_emotion_detection_model.h5')

# Define class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Emotion detection function
def detect_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(prediction[0][predicted_index] * 100, 2)

    return predicted_class, confidence

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded!'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected!'

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Detect emotion
            emotion, confidence = detect_emotion(file_path)

            return render_template(r'D:\project\Mood detection\templates\index.html', image_path=file_path, emotion=emotion, confidence=confidence)

    return render_template(r'D:\project\Mood detection\templates\index.html')

from flask import request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Add this function if not already in your file
def predict_from_pil(img_pil):
    # Convert to grayscale and resize to match your model's expected input
    img = img_pil.convert('L').resize((48, 48))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # shape: (1,48,48,1)
    preds = model.predict(arr)  # 'model' should be your loaded Keras model
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[0][idx])

# Add the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image provided'}), 400

        # Extract base64 part from "data:image/png;base64,...."
        header, b64 = data.split(',', 1)
        img_bytes = base64.b64decode(b64)
        img = Image.open(BytesIO(img_bytes))

        label, conf = predict_from_pil(img)
        return jsonify({'mood': label, 'confidence': round(conf*100, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)