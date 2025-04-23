import os
import shutil
import numpy as np
from skimage import io, color, filters, exposure
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import zscore

def preprocess_image(image_path):
    """Preprocess MRI scan: normalize and denoise."""
    image = io.imread(image_path, as_gray=True)  # Read and convert to grayscale

    # Normalize image to range 0–1
    image = exposure.rescale_intensity(image, out_range=(0, 1))

    # Apply Gaussian filter for denoising
    image = filters.gaussian(image, sigma=1)

    return (image * 255).astype(np.uint8)  # Convert back to uint8 for GLCM

def extract_features(image_path):
    """Extract features specific to MRI scans."""
    try:
        image = preprocess_image(image_path)

        # Compute GLCM and texture features
        glcm = graycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True, levels=256)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        entropy = -np.sum(glcm * np.log2(glcm + 1e-6))  # Add small constant to avoid log(0)

        # Mean intensity of the normalized image
        mean_intensity = np.mean(image) / 255.0  # Normalize mean intensity to range 0–1

        return np.array([energy, contrast, entropy, mean_intensity])
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def filter_images(input_dir, output_dir_removed, zscore_threshold=3):
    """Filter MRI images based on features and z-score."""
    features = []
    image_paths = []

    # Load all images and extract features
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                feature = extract_features(image_path)
                if feature is not None:
                    features.append(feature)
                    image_paths.append(image_path)

    # Convert to numpy array
    features = np.array(features)

    # Calculate Z-scores
    z_scores = np.abs(zscore(features, axis=0))
    max_z_scores = np.max(z_scores, axis=1)

    # Separate normal and outlier images
    normal_indices = np.where(max_z_scores <= zscore_threshold)[0]
    outlier_indices = np.where(max_z_scores > zscore_threshold)[0]

    # Move normal images (filtered) to their original directories
    for idx in normal_indices:
        normal_image_path = image_paths[idx]
        print(f"Keeping: {normal_image_path}")

    # Move outlier images to removed folder
    for idx in outlier_indices:
        outlier_image_path = image_paths[idx]
        relative_path = os.path.relpath(outlier_image_path, input_dir)
        dest_path = os.path.join(output_dir_removed, relative_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(outlier_image_path, dest_path)
        print(f"Removed: {outlier_image_path} -> {dest_path}")

if __name__ == "__main__":
    base_dir = "data"
    removed_dir = "data/removed"

    # Directories for YES and NO
    yes_dir = os.path.join(base_dir, "YES")
    no_dir = os.path.join(base_dir, "NO")
    yes_removed_dir = os.path.join(removed_dir, "YES")
    no_removed_dir = os.path.join(removed_dir, "NO")

    # Filter YES and NO images
    print("Processing YES images...")
    filter_images(yes_dir, yes_removed_dir)
    print("Processing NO images...")
    filter_images(no_dir, no_removed_dir)
