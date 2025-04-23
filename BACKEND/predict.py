import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
from classification import reshape_for_cnn


# Flask app configuration
app = Flask(__name__)
app.secret_key = "secret_key"  # Replace with a secure key in production
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# LBP extraction helper function
def extract_lbp(image):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))[0]

# Feature extraction function (enhanced with LBP)
def extract_image_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded")
    image = cv2.resize(image, (128, 128))
    mean_intensity = np.mean(image)
    contrast = np.std(image)
    energy = np.sum(image**2)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    lbp_features = extract_lbp(image)
    return np.concatenate([np.array([energy, contrast, entropy, mean_intensity]), lbp_features])

# Prediction function using ensemble models
def predict_stone(image_path):
    # Load the trained models and scaler
    scaler = joblib.load("models/scaler.pkl")
    svm_model = joblib.load("models/svm_model.pkl")
    rf_model = joblib.load("models/rf_model.pkl")
    cnn_model = load_model("models/cnn_model.keras")

    # Extract features and process them
    features = extract_image_features(image_path).reshape(1, -1)
    basic_features = features[:, :4]
    lbp_features = features[:, 4:]
    scaled_basic_features = scaler.transform(basic_features)
    final_features = np.concatenate([scaled_basic_features, lbp_features], axis=1)

    # Pad features to 128 dimensions if needed
    if final_features.shape[1] < 128:
        padded_features = np.pad(final_features, ((0, 0), (0, 128 - final_features.shape[1])), mode='constant', constant_values=0)
    else:
        padded_features = final_features[:, :128]

    scaled_features_cnn = reshape_for_cnn(padded_features)

    # Get predictions from each model
    cnn_prob = cnn_model.predict(scaled_features_cnn)[0][0]
    svm_prob = svm_model.predict_proba(padded_features)[0][1]
    rf_prob = rf_model.predict_proba(padded_features)[0][1]

    # Define model weights and calculate ensemble probability
    cnn_weight = 0.1
    svm_weight = 0.1
    rf_weight = 0.8
    ensemble_prob = (cnn_prob * cnn_weight + svm_prob * svm_weight + rf_prob * rf_weight)

    threshold = 0.92
    prediction = "YES" if ensemble_prob > threshold else "NO"
    return prediction, ensemble_prob

# Route: Home page with upload form
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                prediction, prob = predict_stone(file_path)
                return render_template('result.html', prediction=prediction, prob=f"{prob:.4f}", filename=filename)
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                return redirect(request.url)
    return render_template('index.html')

# Route: Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
