import cv2
import numpy as np

def adaptive_contrast_enhancement(image):
    """Apply Adaptive Contrast Enhancement (CLAHE)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def median_filter(image):
    """Apply Median Filtering"""
    return cv2.medianBlur(image, 5)

def preprocess_image(image_path):
    """Preprocess a single image by enhancing contrast and reducing noise.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        np.ndarray: Preprocessed image.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Apply Adaptive Contrast Enhancement
    enhanced = adaptive_contrast_enhancement(image)

    # Apply Median Filtering
    filtered = median_filter(enhanced)

    return filtered

def preprocess_images_in_directory(input_dir, output_dir):
    """Preprocess all images in a directory and save the results.
    
    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory to save preprocessed images.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in ["NO", "YES"]:
        input_label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)

        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        for filename in os.listdir(input_label_dir):
            input_path = os.path.join(input_label_dir, filename)
            output_path = os.path.join(output_label_dir, filename)

            try:
                preprocessed_image = preprocess_image(input_path)
                cv2.imwrite(output_path, preprocessed_image)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Example usage for TRAIN dataset
    input_directory = "data/TRAIN"
    output_directory = "data/Preprocessed/TRAIN"
    preprocess_images_in_directory(input_directory, output_directory)

    # Example usage for VAL dataset
    input_directory = "data/VAL"
    output_directory = "data/Preprocessed/VAL"
    preprocess_images_in_directory(input_directory, output_directory)

    # Example usage for TEST dataset
    input_directory = "data/TEST"
    output_directory = "data/Preprocessed/TEST"
    preprocess_images_in_directory(input_directory, output_directory)
