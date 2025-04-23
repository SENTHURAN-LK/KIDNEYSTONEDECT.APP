import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
import csv


def extract_glcm_features(image):
    """
    Extract features using Gray Level Co-occurrence Matrix (GLCM).

    Args:
        image (np.ndarray): Segmented grayscale image.

    Returns:
        dict: Extracted features (energy, contrast, entropy, mean_intensity).
    """
    # Ensure image has the expected value range (0-255) for GLCM
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Extract features
    energy = graycoprops(glcm, 'energy')[0, 0]
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    mean_intensity = np.mean(image)

    return {
        'energy': energy,
        'contrast': contrast,
        'entropy': entropy,
        'mean_intensity': mean_intensity
    }


def extract_features_from_image(image_path):
    """
    Extract features from a single image.

    Args:
        image_path (str): Path to the segmented image.

    Returns:
        dict: Features extracted from the image.
    """
    # Load the segmented image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Extract features
    features = extract_glcm_features(image)
    return features


def extract_features_in_directory(input_dir, output_csv):
    """
    Extract features from all images in a directory and save to a CSV file.

    Args:
        input_dir (str): Path to the directory containing segmented images.
        output_csv (str): Path to save the extracted features as a CSV file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Prepare CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label', 'Energy', 'Contrast', 'Entropy', 'Mean_Intensity'])

        for label in ["NO", "YES"]:
            label_dir = os.path.join(input_dir, label)

            if not os.path.exists(label_dir):
                print(f"Label directory {label_dir} does not exist. Skipping.")
                continue

            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                try:
                    features = extract_features_from_image(image_path)
                    writer.writerow([filename, label, features['energy'], features['contrast'], features['entropy'], features['mean_intensity']])
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    # Example usage for TRAIN dataset
    input_directory = "data/Segmented/TRAIN"
    output_csv = "data/Features/train_features.csv"
    extract_features_in_directory(input_directory, output_csv)

    # Example usage for VAL dataset
    input_directory = "data/Segmented/VAL"
    output_csv = "data/Features/val_features.csv"
    extract_features_in_directory(input_directory, output_csv)

    # Example usage for TEST dataset
    input_directory = "data/Segmented/TEST"
    output_csv = "data/Features/test_features.csv"
    extract_features_in_directory(input_directory, output_csv)
