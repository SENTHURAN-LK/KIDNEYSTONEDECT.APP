import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib

def load_features(feature_csv):
    """Load features and labels from a CSV file."""
    data = pd.read_csv(feature_csv)
    X = data[['Energy', 'Contrast', 'Entropy', 'Mean_Intensity']].values
    y = data['Label'].apply(lambda label: 1 if label == 'YES' else 0).values
    return X, y

def reshape_for_cnn(X):
    """Reshape input data for CNN compatibility (for 14 features)."""
    # Ensure the data is reshaped as (samples, features, 1)
    return X.reshape(X.shape[0], X.shape[1], 1)  # Shape will be (samples, 14, 1)

def build_cnn_model(input_shape):
    """Build and compile a CNN model that accepts 128 features."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.5),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, padding='same'),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(train_csv, val_csv, model_save_path):
    """Train and evaluate the Ensemble CNN-SVM-Random Forest model with class weights and performance visualization."""
    X_train, y_train = load_features(train_csv)
    X_val, y_val = load_features(val_csv)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Pad features to 128 dimensions before reshaping for CNN
    if X_train.shape[1] < 128:
        X_train = np.pad(X_train, ((0, 0), (0, 128 - X_train.shape[1])), mode='constant', constant_values=0)
    else:
        X_train = X_train[:, :128]  # Truncate if there are more than 128 features

    if X_val.shape[1] < 128:
        X_val = np.pad(X_val, ((0, 0), (0, 128 - X_val.shape[1])), mode='constant', constant_values=0)
    else:
        X_val = X_val[:, :128]  # Truncate if there are more than 128 features

    # Reshape for CNN
    X_train_cnn = reshape_for_cnn(X_train)
    X_val_cnn = reshape_for_cnn(X_val)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Initialize and train CNN
    cnn_model = build_cnn_model(input_shape=(X_train_cnn.shape[1], 1))
    cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_data=(X_val_cnn, y_val), 
                  class_weight=class_weight_dict, verbose=1)

    # Train SVM and Random Forest
    svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True)
    svm_classifier.fit(X_train, y_train)

    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)

    # Evaluate individual models
    cnn_acc = cnn_model.evaluate(X_val_cnn, y_val, verbose=0)[1]
    svm_acc = accuracy_score(y_val, svm_classifier.predict(X_val))
    rf_acc = accuracy_score(y_val, rf_classifier.predict(X_val))

    print(f"CNN Accuracy: {cnn_acc:.4f}")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Ensemble predictions
    cnn_probs = cnn_model.predict(X_val_cnn, verbose=0).flatten()
    svm_probs = svm_classifier.predict_proba(X_val)[:, 1]
    rf_probs = rf_classifier.predict_proba(X_val)[:, 1]
    ensemble_probs = (cnn_probs + svm_probs + rf_probs) / 3
    ensemble_predictions = (ensemble_probs > 0.5).astype(int)

    ensemble_acc = accuracy_score(y_val, ensemble_predictions)
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    print("Classification Report:\n", classification_report(y_val, ensemble_predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_val, ensemble_predictions))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_val, ensemble_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Ensemble ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.calibration import CalibratedClassifierCV

    # Train SVM with class weight 'balanced'
    svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True)
    svm_classifier.fit(X_train, y_train)

    # Train Random Forest with class weight 'balanced'
    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)

    # Calibrate the SVM and Random Forest models
    svm_classifier = CalibratedClassifierCV(svm_classifier, method='sigmoid')
    rf_classifier = CalibratedClassifierCV(rf_classifier, method='sigmoid')

    # Re-fit the models with calibration
    svm_classifier.fit(X_train, y_train)
    rf_classifier.fit(X_train, y_train)

    # Save models and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(svm_classifier, "models/svm_model.pkl")
    joblib.dump(rf_classifier, "models/rf_model.pkl")
    cnn_model.save("models/cnn_model.keras")
    joblib.dump((svm_classifier, rf_classifier, cnn_model), model_save_path)

def predict(test_csv, model_path):
    """Predict stone presence using the trained Ensemble CNN-SVM-Random Forest model."""
    X_test, y_test = load_features(test_csv)

    # Load saved models and scaler
    scaler = joblib.load("models/scaler.pkl")
    svm_classifier, rf_classifier, cnn_model = joblib.load(model_path)

    # Scale and pad the test data
    X_test = scaler.transform(X_test)

    if X_test.shape[1] < 128:
        X_test = np.pad(X_test, ((0, 0), (0, 128 - X_test.shape[1])), mode='constant', constant_values=0)
    else:
        X_test = X_test[:, :128]  # Truncate if there are more than 128 features

    # Reshape for CNN
    X_test_cnn = reshape_for_cnn(X_test)

    # Ensemble prediction
    cnn_probs = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    svm_probs = svm_classifier.predict_proba(X_test)[:, 1]
    rf_probs = rf_classifier.predict_proba(X_test)[:, 1]
    ensemble_probs = (cnn_probs + svm_probs + rf_probs) / 3
    ensemble_predictions = (ensemble_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, ensemble_predictions)
    print("Test Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, ensemble_predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, ensemble_predictions))


if __name__ == "__main__":
    train_features = "data/Features/train_features.csv"
    val_features = "data/Features/val_features.csv"
    test_features = "data/Features/test_features.csv"
    model_path = "models/ensemble_model.pkl"

    os.makedirs("models", exist_ok=True)

    train_and_evaluate(train_features, val_features, model_path)
    predict(test_features, model_path)
