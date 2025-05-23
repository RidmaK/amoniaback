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

def get_concentration_history():
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            return df[['timestamp', 'concentration', 'color_hex']].to_dict('records')
    except Exception as e:
        logger.error(f"Error reading concentration history: {str(e)}")
    return []

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running!"}), 200

@app.route('/history', methods=['GET'])
def get_history():
    # Get prediction history
    history = get_concentration_history()
    # Get color chart for display
    chart = []
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
    return jsonify({
        "history": history,
        "chart": chart
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug(f"Request received - Content-Type: {request.content_type}")
        logger.debug(f"Files in request: {request.files}")
        logger.debug(f"Form data: {request.form}")
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Content Length: {request.content_length}")
        
        # Handle file upload
        file_bytes = None
        
        # Check if we have form data
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
                "details": "No valid file data found in the request",
                "help": "In React Native, use: const imageResponse = await fetch(imageUri); const blob = await imageResponse.blob(); const formData = new FormData(); formData.append('file', blob, 'image.jpg');"
            }), 400

        # Save the image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_dir = os.path.join('data', 'images')
        os.makedirs(img_dir, exist_ok=True)
        img_filename = f'image_{timestamp}.jpg'
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
            
            # Extract middle area color (OpenCV loads as BGR)
            height, width = image.shape[:2]
            # Calculate middle area (25% of the image size)
            middle_h = int(height * 0.25)
            middle_w = int(width * 0.25)
            # Calculate start and end coordinates for middle area
            start_h = (height - middle_h) // 2
            end_h = start_h + middle_h
            start_w = (width - middle_w) // 2
            end_w = start_w + middle_w
            
            # Extract middle area
            middle_area = image[start_h:end_h, start_w:end_w]
            avg_color = middle_area.mean(axis=(0, 1))
            r, g, b = int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
            color_hex = '#{:02x}{:02x}{:02x}'.format(r, g, b)
            logger.debug(f"Detected middle area color: {color_hex} RGB: ({r},{g},{b})")

            # Find closest color in chart
            match = color_chart.find_closest((r, g, b))
            logger.debug(f"Matched chart color: {match['hex']} RGB: {match['rgb']} -> {match['concentration']} mg/L (distance: {match['distance']:.2f})")

            # Save to dataset
            if not os.path.exists(DATASET_PATH):
                pd.DataFrame(columns=[
                    'timestamp', 'image_path', 'concentration', 
                    'color_hex', 'color_r', 'color_g', 'color_b'
                ]).to_csv(DATASET_PATH, index=False)
            
            new_data = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(),
                'image_path': img_path,
                'concentration': match['concentration'],
                'color_hex': match['hex'],
                'color_r': r,
                'color_g': g,
                'color_b': b
            }])
            new_data.to_csv(DATASET_PATH, mode='a', header=False, index=False)
            
            history = get_concentration_history()
            
            return jsonify({
                "ammonia_concentration": float(match['concentration']),
                "success": True,
                "saved_image": img_filename,
                "original_color":  {
                    "hex": color_hex,
                    "rgb": {
                        "r": int(match['rgb'][0]),
                        "g": int(match['rgb'][1]),
                        "b": int(match['rgb'][2])
                    }
                },
                "color": {
                    "hex": str(match['hex']),
                    "rgb": {
                        "r": int(match['rgb'][0]),
                        "g": int(match['rgb'][1]),
                        "b": int(match['rgb'][2])
                    }
                },
                "distance": float(match['distance']),
                "history": history,
                "chart": [
                    {
                        'concentration': float(row['Concentration_mg_L']),
                        'hex': str(row['Hex']),
                        'rgb': {
                            'r': int(row['Red']),
                            'g': int(row['Green']),
                            'b': int(row['Blue'])
                        }
                    } for _, row in color_chart.df.iterrows()
                ]
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
