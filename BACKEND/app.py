from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
scaler = joblib.load('models/scaler.pkl')
svm_model = joblib.load('models/svm_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
cnn_model = load_model('models/cnn_model.keras')

# Helper: allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Helper: LBP
def extract_lbp(image):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))[0]

# Helper: feature extraction
def extract_image_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    mean_intensity = np.mean(image)
    contrast = np.std(image)
    energy = np.sum(image**2)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    lbp_features = extract_lbp(image)
    return np.concatenate([[energy, contrast, entropy, mean_intensity], lbp_features])

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        features = extract_image_features(path).reshape(1, -1)
        basic_features = features[:, :4]
        lbp_features = features[:, 4:]
        scaled = scaler.transform(basic_features)
        final_features = np.concatenate([scaled, lbp_features], axis=1)
        padded = np.pad(final_features, ((0, 0), (0, 128 - final_features.shape[1])), mode='constant')

        cnn_prob = cnn_model.predict(padded)[0][0]
        svm_prob = svm_model.predict_proba(padded)[0][1]
        rf_prob = rf_model.predict_proba(padded)[0][1]

        final_prob = cnn_prob * 0.1 + svm_prob * 0.1 + rf_prob * 0.8
        prediction = "YES" if final_prob > 0.92 else "NO"

        return jsonify({'result': prediction, 'probability': float(final_prob)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000, debug=True)
