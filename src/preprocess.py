import cv2
import os

DATA_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def preprocess_image(image_path, save_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
    resized = cv2.resize(image, (200, 200))  # Standardize size
    cv2.imwrite(save_path, resized)

for img_name in os.listdir(DATA_DIR):
    preprocess_image(os.path.join(DATA_DIR, img_name), os.path.join(PROCESSED_DIR, img_name))

print("✅ Image preprocessing complete!")
