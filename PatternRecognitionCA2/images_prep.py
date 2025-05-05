import os
import cv2

# ======= Get Images from the Dataset =======
def get_images(root_path, selected_idx):
    images_training = []
    labels_training = []
    images_test = []
    labels_test = []

    # Iterate through each person folder，skip if not in selected_idx
    for person_idx in sorted(os.listdir(root_path)):
        if int(person_idx) not in selected_idx:
            continue

        person_folder = os.path.join(root_path, person_idx) # Contactenate the folder path 

        image_files = sorted([
            file for file in os.listdir(person_folder)
            if file.endswith(".jpg")
        ]) # Get all the sorted image files (according to their indices) in the folder
        
        total_images = len(image_files) # Get the total number of images in the folder
        train_count = int(0.7 * total_images) # 70% of the images are used for training, the rest for testing
        
        for file in image_files[:train_count]: # Indices from 0 to train_count - 1
            file_path = os.path.join(person_folder, file) # Concatenate the file path
            img_current = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # Read the image in grayscale
            images_training.append(img_current)# Append the image to the list
            labels_training.append(int(person_idx)) # Append the corresponding label to the list

        for file in image_files[train_count:]:
            file_path = os.path.join(person_folder, file)
            img_current = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            images_test.append(img_current)
            labels_test.append(int(person_idx))

    return images_training, labels_training, images_test, labels_test







        



        








