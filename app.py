from flask import Flask, send_from_directory, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import base64

app = Flask(__name__)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained model
model = load_model('fer_model.keras')  # Replace with your model path

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect face
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return frame[y:y+h, x:x+w]  # Return the cropped face
    return None

# Function to preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (48, 48))  # Resize to 48x48
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict emotion
def predict_emotion(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

# Route for the home page
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Serve index.html from the root directory

# Route to handle webcam image
@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    # Decode the base64 image
    img_data = data['image'].split(',')[1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)

    # Detect face
    face = detect_face(img)
    if face is None:
        return jsonify({'error': 'No face detected'}), 400

    # Predict emotion
    emotion = predict_emotion(face)
    return jsonify({'emotion': emotion})

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read the image file
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect face
        face = detect_face(img)
        if face is None:
            return jsonify({'error': 'No face detected'}), 400

        # Predict emotion
        emotion = predict_emotion(face)
        return jsonify({'emotion': emotion})
    except Exception as e:
        print("Error in /upload route:", str(e))  # Debug statement
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)