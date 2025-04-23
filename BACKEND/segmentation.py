import cv2
import numpy as np
from skimage.measure import label  # Correct import for the label function
import os

def fuzzy_c_means_segmentation(image):
    """
    Apply segmentation to an image using adaptive thresholding followed by 
    connected component analysis to isolate the largest region.

    Args:
        image (np.ndarray): Preprocessed grayscale image.

    Returns:
        np.ndarray: Binary mask of the segmented stone region.
    """
    # Normalize the image to range [0, 255] for OpenCV adaptive thresholding
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply adaptive thresholding instead of Otsu's method.
    # Using Gaussian adaptive thresholding with blockSize=11 and C=2.
    binary_mask = cv2.adaptiveThreshold(
        image_uint8, 
        maxValue=1, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )

    # Label the connected components
    labeled_image = label(binary_mask, connectivity=2)

    # Identify the largest connected component (assuming stone is largest)
    label_counts = np.bincount(labeled_image.flat)
    if len(label_counts) <= 1:
        # If no components found, return the binary mask as is.
        return binary_mask
    label_counts[0] = 0  # Exclude background label
    largest_label = label_counts.argmax()
    largest_component = (labeled_image == largest_label).astype(np.uint8)

    return largest_component

def segment_image(image_path, output_path):
    """
    Segment a single image to identify the stone region and save the result.

    Args:
        image_path (str): Path to the preprocessed image.
        output_path (str): Path to save the segmented image.
    """
    # Load the preprocessed image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Apply segmentation
    segmented_image = fuzzy_c_means_segmentation(image)

    # Save the segmented image (convert binary mask to 0 and 255)
    cv2.imwrite(output_path, segmented_image * 255)
    print(f"Segmented image saved: {output_path}")

def segment_images_in_directory(input_dir, output_dir):
    """
    Segment all images in a directory and save the results.

    Args:
        input_dir (str): Path to the directory containing preprocessed images.
        output_dir (str): Path to the directory to save segmented images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in ["NO", "YES"]:
        input_category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)

        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        for filename in os.listdir(input_category_dir):
            input_path = os.path.join(input_category_dir, filename)
            output_path = os.path.join(output_category_dir, filename)

            try:
                segment_image(input_path, output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Process TRAIN dataset
    train_input_dir = "data/Preprocessed/TRAIN"
    train_output_dir = "data/Segmented/TRAIN"
    segment_images_in_directory(train_input_dir, train_output_dir)

    # Process VAL dataset
    val_input_dir = "data/Preprocessed/VAL"
    val_output_dir = "data/Segmented/VAL"
    segment_images_in_directory(val_input_dir, val_output_dir)

    # Process TEST dataset
    test_input_dir = "data/Preprocessed/TEST"
    test_output_dir = "data/Segmented/TEST"
    segment_images_in_directory(test_input_dir, test_output_dir)
