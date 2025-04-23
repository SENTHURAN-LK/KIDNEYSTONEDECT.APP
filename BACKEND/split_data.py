import os
import shutil
import random

def copy_images(src_dir, train_dir, val_dir, test_dir, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    """
    Copies images from YES and NO directories in the source directory to respective directories in the TRAIN, 
    VAL, and TEST directories, based on the given ratios.
    """
    categories = ['YES', 'NO']
    
    for category in categories:
        # Get the source and destination directories for each category
        category_src_dir = os.path.join(src_dir, category)
        category_train_dir = os.path.join(train_dir, category)
        category_val_dir = os.path.join(val_dir, category)
        category_test_dir = os.path.join(test_dir, category)
        
        # Create destination directories if they do not exist
        os.makedirs(category_train_dir, exist_ok=True)
        os.makedirs(category_val_dir, exist_ok=True)
        os.makedirs(category_test_dir, exist_ok=True)
        
        # List all image files in the source directory
        images = [f for f in os.listdir(category_src_dir) if os.path.isfile(os.path.join(category_src_dir, f))]
        
        # Shuffle the images to randomize the selection
        random.shuffle(images)
        
        # Determine the number of images for each set based on the specified ratios
        total_images = len(images)
        num_train = int(total_images * train_ratio)
        num_val = int(total_images * val_ratio)
        num_test = total_images - num_train - num_val  # The rest goes to the test set
        
        # Copy images to respective directories
        for i, image in enumerate(images):
            src_image_path = os.path.join(category_src_dir, image)
            
            if i < num_train:
                dest_image_path = os.path.join(category_train_dir, image)
            elif i < num_train + num_val:
                dest_image_path = os.path.join(category_val_dir, image)
            else:
                dest_image_path = os.path.join(category_test_dir, image)
            
            shutil.copy(src_image_path, dest_image_path)

if __name__ == "__main__":
    # Define the source and destination directories
    src_dir = 'data'  # The directory where YES and NO folders are located
    train_dir = 'data/TRAIN'
    val_dir = 'data/VAL'
    test_dir = 'data/TEST'

    # Copy images from the YES and NO directories to TRAIN, VAL, and TEST directories
    copy_images(src_dir, train_dir, val_dir, test_dir)
    print("Images have been successfully copied to TRAIN, VAL, and TEST directories.")
