import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_dominant_color(image, k=1):
    """
    Detect the dominant color in the image using K-means clustering.
    Returns the RGB values of the dominant color as integers.
    """
    # Convert image to RGB if it's BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Cluster the pixel intensities
    clt = KMeans(n_clusters=k)
    clt.fit(pixels)
    
    # Return the dominant color (centroid) as integers
    return np.round(clt.cluster_centers_[0]).astype(int)

def extract_color_features(image):
    """
    Extract color features from the image.
    Returns a feature vector containing color information.
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Calculate mean values for each channel
    bgr_means = np.mean(image, axis=(0, 1))
    hsv_means = np.mean(hsv, axis=(0, 1))
    lab_means = np.mean(lab, axis=(0, 1))
    
    # Calculate standard deviations for each channel
    bgr_stds = np.std(image, axis=(0, 1))
    hsv_stds = np.std(hsv, axis=(0, 1))
    lab_stds = np.std(lab, axis=(0, 1))
    
    # Get dominant color
    dominant_color = detect_dominant_color(image)
    
    # Combine all features
    features = np.concatenate([
        bgr_means, bgr_stds,
        hsv_means, hsv_stds,
        lab_means, lab_stds,
        dominant_color
    ])
    
    return features.reshape(1, -1)
