import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import base64
from datetime import datetime
import pandas as pd
from color_chart_utils import ColorChart
import colorsys
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import json
from flask import jsonify, request
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = "data/dataset.csv"

# Load color chart
color_chart = ColorChart()

# Flask app
app = Flask(__name__)
CORS(app)



@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running!"}), 200

# Define the dataset path
DATASET_PATH = os.path.join('data', 'nessler_dataset.csv')

def get_concentration_history():
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            # Use the correct column name from your CSV
            return df[['timestamp', 'concentration_linear', 'color_hex']].rename(
                columns={'concentration_linear': 'concentration'}
            ).to_dict('records')
    except Exception as e:
        logger.error(f"Error reading concentration history: {str(e)}")
    return []

@app.route('/history', methods=['GET'])
def get_history():
    # Get prediction history
    history = get_concentration_history()
    
    # Create color chart from your CSV data if available
    chart = []
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            # Create chart from existing data
            for _, row in df.iterrows():
                chart.append({
                    'concentration': float(row['concentration_linear']),
                    'hex': str(row['color_hex']),
                    'rgb': {
                        'r': int(row['color_r']),
                        'g': int(row['color_g']),
                        'b': int(row['color_b'])
                    }
                })
    except Exception as e:
        logger.error(f"Error creating color chart: {str(e)}")
        # Fallback: create a basic chart if your color_chart object exists
        # Uncomment and modify this section if you have a separate color chart
        """
        try:
            for i, row in color_chart.df.iterrows():
                chart.append({
                    'concentration': float(row['Concentration_mg_L']),
                    'hex': str(row['Hex']),
                    'rgb': {
                        'r': int(row['Red']),
                        'g': int(row['Green']),
                        'b': int(row['Blue'])
                    }
                })
        except Exception as e2:
            logger.error(f"Error with color_chart fallback: {str(e2)}")
        """
    
    return jsonify({
        "history": history,
        "chart": chart
    })

# Alternative version if you want to use both concentration values
def get_concentration_history_extended():
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            # Include both concentration values
            return df[['timestamp', 'concentration_linear', 'concentration_quadratic', 'color_hex']].to_dict('records')
    except Exception as e:
        logger.error(f"Error reading concentration history: {str(e)}")
    return []

# Updated Nessler reagent color chart with new values and calibration points
NESSLER_COLOR_CHART = [
    {'concentration': 0.0, 'description': 'Clear', 'rgb': (240, 240, 220), 'hex': '#F0F0DC'},
    {'concentration': 0.5, 'description': 'Very pale yellow', 'rgb': (235, 225, 180), 'hex': '#EBE1B4'},
    {'concentration': 1.0, 'description': 'Pale yellow', 'rgb': (230, 215, 160), 'hex': '#E6D79F'},
    {'concentration': 2.0, 'description': 'Yellow', 'rgb': (220, 200, 130), 'hex': '#DCC882'},
    {'concentration': 3.0, 'description': 'Yellow-orange', 'rgb': (210, 180, 100), 'hex': '#D2B464'},
    {'concentration': 4.0, 'description': 'Golden-orange', 'rgb': (200, 160, 80), 'hex': '#C8A050'},
    {'concentration': 5.0, 'description': 'Orange', 'rgb': (185, 140, 60), 'hex': '#B98C3C'},
    {'concentration': 6.0, 'description': 'Light brown', 'rgb': (170, 120, 50), 'hex': '#AA7832'},
    {'concentration': 7.0, 'description': 'Brown', 'rgb': (150, 100, 40), 'hex': '#966428'},
    {'concentration': 8.0, 'description': 'Darker brown', 'rgb': (135, 85, 35), 'hex': '#875523'},
    {'concentration': 9.0, 'description': 'Deep brown', 'rgb': (120, 70, 30), 'hex': '#78461E'},
    {'concentration': 10.0, 'description': 'Very deep brown', 'rgb': (105, 60, 25), 'hex': '#693C19'},
    {'concentration': 11.0, 'description': 'Almost black-brown', 'rgb': (90, 50, 20), 'hex': '#5A3214'},
    {'concentration': 12.0, 'description': 'Dark brown-black', 'rgb': (75, 40, 15), 'hex': '#4B280F'},
    {'concentration': 13.0, 'description': 'Nearly black', 'rgb': (60, 30, 10), 'hex': '#3C1E0A'},
    {'concentration': 13.4, 'description': 'Max color depth', 'rgb': (50, 25, 8), 'hex': '#321908'}
]

# Add calibration points for more precise measurements
CALIBRATION_POINTS = [
    {'concentration': 0.0, 'rgb': (240, 240, 220), 'hex': '#F0F0DC'},
    {'concentration': 0.25, 'rgb': (238, 233, 200), 'hex': '#EEE9C8'},
    {'concentration': 0.5, 'rgb': (235, 225, 180), 'hex': '#EBE1B4'},
    {'concentration': 0.75, 'rgb': (233, 220, 170), 'hex': '#E9DCAA'},
    {'concentration': 1.0, 'rgb': (230, 215, 160), 'hex': '#E6D79F'},
    {'concentration': 1.5, 'rgb': (225, 208, 145), 'hex': '#E1D091'},
    {'concentration': 2.0, 'rgb': (220, 200, 130), 'hex': '#DCC882'},
    {'concentration': 2.5, 'rgb': (215, 190, 115), 'hex': '#D7BE73'},
    {'concentration': 3.0, 'rgb': (210, 180, 100), 'hex': '#D2B464'},
    {'concentration': 3.5, 'rgb': (205, 170, 90), 'hex': '#CDAA5A'},
    {'concentration': 4.0, 'rgb': (200, 160, 80), 'hex': '#C8A050'},
    {'concentration': 4.5, 'rgb': (193, 150, 70), 'hex': '#C19646'},
    {'concentration': 5.0, 'rgb': (185, 140, 60), 'hex': '#B98C3C'},
    {'concentration': 5.5, 'rgb': (178, 130, 55), 'hex': '#B28237'},
    {'concentration': 6.0, 'rgb': (170, 120, 50), 'hex': '#AA7832'},
    {'concentration': 6.5, 'rgb': (160, 110, 45), 'hex': '#A06E2D'},
    {'concentration': 7.0, 'rgb': (150, 100, 40), 'hex': '#966428'},
    {'concentration': 7.5, 'rgb': (143, 93, 38), 'hex': '#8F5D26'},
    {'concentration': 8.0, 'rgb': (135, 85, 35), 'hex': '#875523'},
    {'concentration': 8.5, 'rgb': (128, 78, 33), 'hex': '#804E21'},
    {'concentration': 9.0, 'rgb': (120, 70, 30), 'hex': '#78461E'},
    {'concentration': 9.5, 'rgb': (113, 65, 28), 'hex': '#71411C'},
    {'concentration': 10.0, 'rgb': (105, 60, 25), 'hex': '#693C19'},
    {'concentration': 10.5, 'rgb': (98, 55, 23), 'hex': '#623717'},
    {'concentration': 11.0, 'rgb': (90, 50, 20), 'hex': '#5A3214'},
    {'concentration': 11.5, 'rgb': (83, 45, 18), 'hex': '#532D12'},
    {'concentration': 12.0, 'rgb': (75, 40, 15), 'hex': '#4B280F'},
    {'concentration': 12.5, 'rgb': (68, 35, 13), 'hex': '#44230D'},
    {'concentration': 13.0, 'rgb': (60, 30, 10), 'hex': '#3C1E0A'},
    {'concentration': 13.2, 'rgb': (55, 28, 9), 'hex': '#371C09'},
    {'concentration': 13.4, 'rgb': (50, 25, 8), 'hex': '#321908'}
]

class NesslerColorChart:
    def __init__(self):
        self.df = pd.DataFrame(NESSLER_COLOR_CHART)
        self.calibration_df = pd.DataFrame(CALIBRATION_POINTS)
        # Expand RGB tuples into separate columns
        self.df[['Red', 'Green', 'Blue']] = pd.DataFrame(self.df['rgb'].tolist(), index=self.df.index)
        self.calibration_df[['Red', 'Green', 'Blue']] = pd.DataFrame(self.calibration_df['rgb'].tolist(), index=self.calibration_df.index)
        
    def find_closest(self, rgb):
        """Find the closest color match in the chart using Euclidean distance"""
        r, g, b = rgb
        distances = []
        
        for _, row in self.df.iterrows():
            chart_r, chart_g, chart_b = row['Red'], row['Green'], row['Blue']
            distance = np.sqrt((r - chart_r)**2 + (g - chart_g)**2 + (b - chart_b)**2)
            distances.append(distance)
        
        min_idx = np.argmin(distances)
        closest_match = self.df.iloc[min_idx]
        
        return {
            'concentration': float(closest_match['concentration']),
            'hex': str(closest_match['hex']),
            'rgb': (int(closest_match['Red']), int(closest_match['Green']), int(closest_match['Blue'])),
            'distance': float(distances[min_idx]),
            'description': str(closest_match['description'])
        }
    
    def get_calibration_data(self):
        """Get calibration data for chart visualization"""
        return self.calibration_df.to_dict('records')

# Initialize the color chart
color_chart = NesslerColorChart()

def get_center_circle_color(image, circle_radius_percent=0.1):
    """Get the dominant color from the center circle of the image"""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = int(min(width, height) * circle_radius_percent)
    
    # Create a mask for the circle
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = dist_from_center <= radius
    
    # Extract pixels within the circle
    circle_pixels = image[mask]
    
    if len(circle_pixels) == 0:
        return None
    
    # Get color distribution
    pixels = circle_pixels.reshape(-1, 3)
    
    # Remove any pure black or white pixels (likely noise)
    mask = ~((pixels[:, 0] < 10) & (pixels[:, 1] < 10) & (pixels[:, 2] < 10) |
             (pixels[:, 0] > 245) & (pixels[:, 1] > 245) & (pixels[:, 2] > 245))
    pixels = pixels[mask]
    
    if len(pixels) == 0:
        return None
    
    # Calculate mean color
    mean_color = np.mean(pixels, axis=0)
    
    # Convert to Python native int type
    return [int(x) for x in mean_color]

def predict_concentration_linear(r):
    """
    Predict ammonia concentration using linear regression equation:
    R = -14.35 * C + 248.78
    Solving for C: C = (248.78 - R) / 14.35
    """
    concentration = (248.78 - r) / 14.35
    return max(0, min(13.4, concentration))  # Clamp between 0 and 13.4 mg/L

def predict_concentration_quadratic(r):
    """
    Predict ammonia concentration using quadratic regression equation:
    R = -0.265 * C^2 - 10.83 * C + 242.21
    
    Using quadratic formula: C = (-b ± √(b² - 4ac)) / 2a
    where: a = -0.265, b = -10.83, c = (242.21 - R)
    """
    a = -0.265
    b = -10.83
    c = 242.21 - r
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        # No real solution, use linear fallback
        return predict_concentration_linear(r)
    
    # Use the positive root (concentration should be positive)
    c1 = (-b + np.sqrt(discriminant)) / (2*a)
    c2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    # Choose the positive concentration within range
    concentrations = [c for c in [c1, c2] if 0 <= c <= 13.4]
    
    if not concentrations:
        return predict_concentration_linear(r)
    
    return concentrations[0]

def predict_concentration_from_red_channel(rgb, method='quadratic'):
    """
    Predict ammonia concentration based on the red channel value
    
    Args:
        rgb: tuple of (r, g, b) values
        method: 'linear' or 'quadratic' regression method
    
    Returns:
        dict with prediction results
    """
    r, g, b = rgb
    
    if method == 'quadratic':
        predicted_concentration = predict_concentration_quadratic(r)
    else:
        predicted_concentration = predict_concentration_linear(r)
    
    # Find closest match in color chart
    closest_match = color_chart.find_closest(rgb)
    
    # Calculate confidence based on how close the red value is to expected
    expected_r_linear = 248.78 - 14.35 * predicted_concentration
    expected_r_quadratic = -0.265 * predicted_concentration**2 - 10.83 * predicted_concentration + 242.21
    
    if method == 'quadratic':
        r_error = abs(r - expected_r_quadratic)
    else:
        r_error = abs(r - expected_r_linear)
    
    # Confidence decreases with larger red channel error
    confidence = max(0, 100 - (r_error / 2.0))  # Arbitrary scaling
    
    return {
        'predicted_concentration': float(predicted_concentration),
        'chart_match': closest_match,
        'method': method,
        'red_channel': r,
        'expected_red': float(expected_r_quadratic if method == 'quadratic' else expected_r_linear),
        'red_error': float(r_error),
        'confidence': float(confidence)
    }

def calculate_color_metrics(r, g, b):
    """Calculate various color metrics for analysis"""
    # Luminance intensity
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Yellow-brown ratio (useful for Nessler reagent)
    yellow_component = (r + g) / 2
    brown_component = min(r, g, b)
    yellow_brown_ratio = yellow_component / max(brown_component, 1)
    
    # Color saturation
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    saturation = (max_val - min_val) / max(max_val, 1) * 100
    
    return {
        'luminance': float(luminance),
        'yellow_brown_ratio': float(yellow_brown_ratio),
        'saturation': float(saturation)
    }

def enhance_scanned_document(image):
    """Enhance the scanned document image for better color detection"""
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug(f"Request received - Content-Type: {request.content_type}")
        logger.debug(f"Files in request: {request.files}")
        logger.debug(f"Form data: {request.form}")
        
        # Handle file upload (keeping your existing file handling logic)
        file_bytes = None
        
        if 'file' in request.form or 'file' in request.files:
            if 'file' in request.files:
                file = request.files['file']
                file_bytes = file.read()
            else:
                form_data = request.form['file']
                try:
                    import json
                    image_data = json.loads(form_data)
                    if 'uri' in image_data:
                        image_uri = image_data['uri']
                        if image_uri.startswith('file://'):
                            with open(image_uri[7:], 'rb') as f:
                                file_bytes = f.read()
                        else:
                            with open(image_uri, 'rb') as f:
                                file_bytes = f.read()
                except:
                    try:
                        if form_data.startswith('data:image'):
                            file_bytes = base64.b64decode(form_data.split(',')[1])
                        else:
                            file_bytes = base64.b64decode(form_data)
                    except:
                        pass
        
        if file_bytes is None and request.content_length:
            logger.debug("Attempting to read raw request data")
            file_bytes = request.get_data()
            if file_bytes.startswith(b'--'):
                logger.debug("Detected multipart data, attempting to parse")
                try:
                    double_crlf = b'\r\n\r\n'
                    start_idx = file_bytes.find(double_crlf)
                    if start_idx != -1:
                        start_idx += len(double_crlf)
                        boundary = b'--' + request.content_type.split('boundary=')[1].encode()
                        end_idx = file_bytes.rfind(boundary)
                        if end_idx != -1:
                            file_bytes = file_bytes[start_idx:end_idx].strip(b'\r\n')
                except Exception as e:
                    logger.error(f"Error parsing multipart data: {str(e)}")

        if file_bytes is None:
            logger.error("No valid file data found in request")
            return jsonify({
                "error": "No file uploaded",
                "details": "No valid file data found in the request"
            }), 400

        # Save the image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_dir = os.path.join('data', 'images')
        os.makedirs(img_dir, exist_ok=True)
        img_filename = f'nessler_image_{timestamp}.jpg'
        img_path = os.path.join(img_dir, img_filename)
        
        with open(img_path, 'wb') as f:
            f.write(file_bytes)
        logger.debug(f"Saved image to: {img_path}")

        # Process the image
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image data")
                return jsonify({
                    "error": "Invalid image format",
                    "details": "Could not decode the image data. Ensure it's a valid JPEG/PNG file."
                }), 400
                
            logger.debug(f"Successfully decoded image with shape: {image.shape}")
            
            # Enhance the image
            enhanced_image = enhance_scanned_document(image)
            
            # Save enhanced image and convert to base64
            success, buffer = cv2.imencode('.jpg', enhanced_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not success:
                raise Exception("Failed to encode enhanced image")
            
            enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
            enhanced_image_uri = f"data:image/jpeg;base64,{enhanced_base64}"
            
            # Get color from center circle using the enhanced image
            dominant_color = get_center_circle_color(enhanced_image)
            if dominant_color is None:
                return jsonify({
                    "error": "Invalid sample",
                    "details": "Could not detect valid color in the center circle. Please ensure the sample is properly prepared."
                }), 400
            
            # OpenCV loads as BGR, convert to RGB
            b, g, r = dominant_color
            color_hex = '#{:02x}{:02x}{:02x}'.format(r, g, b)
            logger.debug(f"Detected color: {color_hex} RGB: ({r},{g},{b})")

            # Predict concentration using both methods
            linear_result = predict_concentration_from_red_channel((r, g, b), method='linear')
            quadratic_result = predict_concentration_from_red_channel((r, g, b), method='quadratic')
            
            # Use quadratic as primary method (more accurate)
            primary_result = quadratic_result
            
            # Calculate additional color metrics
            color_metrics = calculate_color_metrics(r, g, b)
            
            logger.debug(f"Linear prediction: {linear_result['predicted_concentration']:.2f} mg/L")
            logger.debug(f"Quadratic prediction: {quadratic_result['predicted_concentration']:.2f} mg/L")
            logger.debug(f"Chart match: {primary_result['chart_match']['concentration']} mg/L")

            # Get calibration data for chart
            calibration_data = color_chart.get_calibration_data()

            return jsonify({
                "ammonia_concentration": primary_result['predicted_concentration'],
                "success": True,
                "saved_image": img_filename,
                "enhanced_image": enhanced_image_uri,
                "original_color":  {
                    "hex": color_hex,
                    "rgb": {
                        "r": r,
                        "g": g,
                        "b": b
                    }
                },
                "color": {
                    "hex": str(primary_result['chart_match']['hex']),
                    "rgb": {
                        "r": primary_result['chart_match']['rgb'][0],
                        "g": primary_result['chart_match']['rgb'][1],
                        "b": primary_result['chart_match']['rgb'][2]
                    }
                },
                "distance": primary_result['chart_match']['distance'],
                "history": get_concentration_history(),
                "chart": calibration_data
            })
            
        except Exception as e:
            logger.exception("Error processing image")
            return jsonify({
                "error": "Image processing error",
                "details": str(e),
                "type": str(type(e).__name__)
            }), 500
        
    except Exception as e:
        logger.exception("Unexpected error in predict endpoint")
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e),
            "type": str(type(e).__name__)
        }), 500

@app.route('/train', methods=['POST'])
def train_data():
    try:
        if 'concentration' not in request.form or 'hex' not in request.form or 'r' not in request.form or 'g' not in request.form or 'b' not in request.form:
            return jsonify({"error": "concentration, hex, r, g, b required"}), 400
        concentration = float(request.form['concentration'])
        hex_code = request.form['hex']
        r = int(request.form['r'])
        g = int(request.form['g'])
        b = int(request.form['b'])
        color_chart.add_entry(concentration, hex_code, r, g, b)
        return jsonify({
            "message": "Color chart updated",
            "concentration": concentration,
            "hex": hex_code,
            "rgb": {"r": r, "g": g, "b": b}
        }), 200
    except Exception as e:
        logger.exception("Error in training data upload")
        return jsonify({
            "error": "Training data upload failed",
            "details": str(e)
        }), 500

def is_color_near_black(r, g, b, threshold=50):
    """Check if color is close to black"""
    return r < threshold and g < threshold and b < threshold

def is_color_near_white(r, g, b, threshold=200):
    """Check if color is close to white"""
    return r > threshold and g > threshold and b > threshold

def get_color_distance(color1, color2):
    """Calculate Euclidean distance between two RGB colors"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

@app.route('/validate-color', methods=['POST'])
def validate_color():
    try:
        # Handle file upload similar to predict endpoint
        file_bytes = None
        
        if 'file' in request.form or 'file' in request.files:
            if 'file' in request.files:
                file = request.files['file']
                file_bytes = file.read()
            else:
                form_data = request.form['file']
                try:
                    import json
                    image_data = json.loads(form_data)
                    if 'uri' in image_data:
                        image_uri = image_data['uri']
                        if image_uri.startswith('file://'):
                            with open(image_uri[7:], 'rb') as f:
                                file_bytes = f.read()
                        else:
                            with open(image_uri, 'rb') as f:
                                file_bytes = f.read()
                except:
                    try:
                        if form_data.startswith('data:image'):
                            file_bytes = base64.b64decode(form_data.split(',')[1])
                        else:
                            file_bytes = base64.b64decode(form_data)
                    except:
                        pass

        if file_bytes is None:
            return jsonify({
                "error": "No file uploaded",
                "details": "No valid file data found in the request"
            }), 400

        # Process the image
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "error": "Invalid image format",
                "details": "Could not decode the image data"
            }), 400

        # Extract middle area color
        height, width = image.shape[:2]
        middle_h = int(height * 0.25)
        middle_w = int(width * 0.25)
        start_h = (height - middle_h) // 2
        end_h = start_h + middle_h
        start_w = (width - middle_w) // 2
        end_w = start_w + middle_w
        
        middle_area = image[start_h:end_h, start_w:end_w]
        avg_color = middle_area.mean(axis=(0, 1))
        r, g, b = int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
        color_hex = '#{:02x}{:02x}{:02x}'.format(r, g, b)

        # Check if color is near black or white
        if is_color_near_black(r, g, b):
            return jsonify({
                "status": "error",
                "message": "Sample is too concentrated",
                "action": "Please dilute the sample and try again",
                "color": {
                    "hex": color_hex,
                    "rgb": {"r": r, "g": g, "b": b}
                }
            }), 200

        if is_color_near_white(r, g, b):
            return jsonify({
                "status": "error",
                "message": "Sample is too diluted",
                "action": "Please increase the concentration and try again",
                "color": {
                    "hex": color_hex,
                    "rgb": {"r": r, "g": g, "b": b}
                }
            }), 200

        # Check if color matches any color in the chart
        min_distance = float('inf')
        closest_match = None
        
        for _, row in color_chart.df.iterrows():
            chart_color = (int(row['Red']), int(row['Green']), int(row['Blue']))
            distance = get_color_distance((r, g, b), chart_color)
            if distance < min_distance:
                min_distance = distance
                closest_match = {
                    'concentration': float(row['Concentration_mg_L']),
                    'hex': str(row['Hex']),
                    'rgb': {
                        'r': int(row['Red']),
                        'g': int(row['Green']),
                        'b': int(row['Blue'])
                    }
                }

        # If the closest match is too far from any known color
        if min_distance > 100:  # Threshold for color matching
            return jsonify({
                "status": "error",
                "message": "Color not in expected range",
                "action": "Please ensure the sample is properly prepared and try again",
                "color": {
                    "hex": color_hex,
                    "rgb": {"r": r, "g": g, "b": b}
                },
                "closest_match": closest_match
            }), 200

        # If we get here, the color is valid
        return jsonify({
            "status": "success",
            "message": "Color is within acceptable range",
            "color": {
                "hex": color_hex,
                "rgb": {"r": r, "g": g, "b": b}
            },
            "closest_match": closest_match
        }), 200

    except Exception as e:
        logger.exception("Error in color validation")
        return jsonify({
            "error": "Color validation failed",
            "details": str(e)
        }), 500

def enhance_scanned_document(image, brightness=1.2, contrast=1.3, temperature=1.1, gamma=0.9):
    """
    Enhanced image processing with minimal color distortion and center area preservation
    """
    try:
        # Get image dimensions for center area preservation
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        radius = int(min(width, height) * 0.1)  # 10% of image size
        
        # Create a mask for the center circle
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        center_mask = dist_from_center <= radius
        
        # Store original center area
        original_center = image[center_mask].copy()
        
        # Convert BGR to RGB for processing
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        
        # More conservative enhancement
        # Adjust brightness
        img = img * brightness
        
        # Adjust contrast
        img = (img - 128) * contrast + 128
        
        # Gamma correction
        img = np.clip(img, 0, 255)
        img_normalized = img / 255.0
        img = np.power(img_normalized, gamma) * 255.0
        
        # Temperature adjustment (more subtle)
        if temperature > 1.0:
            img[:,:,0] = img[:,:,0] * temperature  # Red
            img[:,:,2] = img[:,:,2] / temperature  # Blue
        
        # Clip values
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Restore the original center area
        enhanced[center_mask] = original_center
        
        logger.debug("Image enhancement completed with center area preservation")
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in enhance_scanned_document: {str(e)}")
        return image  # Return original image if enhancement fails

def auto_enhance_document(image):
    """
    Automatic image enhancement with default parameters
    """
    try:
        # Use default values that match the React Native code
        brightness = 1.2
        contrast = 1.3
        temperature = 1.1  # Slightly warmer by default
        
        return enhance_scanned_document(image, brightness, contrast, temperature)
        
    except Exception as e:
        logger.error(f"Error in auto_enhance_document: {str(e)}")
        raise

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        logger.debug(f"Process image request received - Content-Type: {request.content_type}")
        
        # Get enhancement parameters
        brightness = float(request.form.get('brightness', 1.2))
        contrast = float(request.form.get('contrast', 1.3))
        temperature = float(request.form.get('temperature', 1.1))
        
        # Handle file upload
        file_bytes = None
        
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            file_bytes = file.read()
            logger.debug(f"File uploaded: {file.filename}, size: {len(file_bytes)} bytes")
            
        elif 'file' in request.form:
            form_data = request.form['file']
            try:
                # Try JSON format first
                image_data = json.loads(form_data)
                if 'uri' in image_data:
                    image_uri = image_data['uri']
                    if image_uri.startswith('file://'):
                        image_uri = image_uri[7:]
                    with open(image_uri, 'rb') as f:
                        file_bytes = f.read()
            except (json.JSONDecodeError, FileNotFoundError):
                # Try base64 format
                try:
                    if form_data.startswith('data:image'):
                        header, data = form_data.split(',', 1)
                        file_bytes = base64.b64decode(data)
                    else:
                        file_bytes = base64.b64decode(form_data)
                except Exception as e:
                    logger.error(f"Failed to decode base64 data: {e}")

        if file_bytes is None or len(file_bytes) == 0:
            return jsonify({
                "error": "No file uploaded",
                "details": "No valid file data found in the request"
            }), 400

        # Decode image
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "error": "Invalid image format",
                "details": "Could not decode the image data. Supported formats: JPG, PNG, BMP, TIFF"
            }), 400

        logger.debug(f"Image decoded successfully: {image.shape}")

        # Apply enhancement
        enhanced_image = enhance_scanned_document(image, brightness, contrast, temperature)
        logger.debug(f"Applied enhancement - brightness: {brightness}, contrast: {contrast}, temperature: {temperature}")

        # Encode to JPEG with quality matching React Native (0.9 = 90%)
        try:
            success, buffer = cv2.imencode('.jpg', enhanced_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            if not success:
                return jsonify({
                    "error": "Encoding error",
                    "details": "Failed to encode the processed image"
                }), 500
            
        except Exception as e:
            logger.error(f"Error during image encoding: {str(e)}")
            return jsonify({
                "error": "Encoding error",
                "details": f"Failed to encode the processed image: {str(e)}"
            }), 500
        
        # Convert to base64
        try:
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error during base64 encoding: {str(e)}")
            return jsonify({
                "error": "Encoding error",
                "details": "Failed to convert image to base64"
            }), 500
        
        logger.debug("Image processing completed successfully")
        
        return jsonify({
            "success": True,
            "imageUri": f"data:image/jpeg;base64,{image_base64}",
            "processingInfo": {
                "brightness": brightness,
                "contrast": contrast,
                "temperature": temperature
            }
        })

    except ValueError as e:
        logger.error(f"Parameter error: {e}")
        return jsonify({
            "error": "Invalid parameters",
            "details": str(e)
        }), 400
        
    except Exception as e:
        logger.exception("Unexpected error processing image")
        return jsonify({
            "error": "Image processing error",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Image Enhancement API"})

@app.route('/calibration-data', methods=['GET'])
def get_calibration_data():
    """Get calibration data for chart visualization"""
    try:
        calibration_data = color_chart.get_calibration_data()
        return jsonify({
            "success": True,
            "calibration_data": calibration_data
        })
    except Exception as e:
        logger.exception("Error getting calibration data")
        return jsonify({
            "error": "Failed to get calibration data",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=5000, debug=True)