import os
import cv2

# ======= Process the Selfie Dataset =======
def process_selfie(root_path, target_path):
    selfie_files = sorted([
        file for file in os.listdir(root_path)
        if file.endswith(".jpg")
    ]) # Get all the sorted image files (according to their indices) in the folder
    
    for file in selfie_files:
        file_path = os.path.join(root_path, file) # Concatenate the file path
        img_current = cv2.imread(file_path) # Read the image
        img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        img_current = cv2.resize(img_current, (32, 32)) # Resize the image to 32x32
        cv2.imwrite(os.path.join(target_path, file), img_current) # Save the processed image