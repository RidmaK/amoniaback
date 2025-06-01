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
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(pixels)
    
    # Get the dominant color (cluster with most points)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_color = colors[counts.argmax()]
    
    # Convert to Python native int type
    return [int(x) for x in dominant_color]

def find_concentration_by_primary_color(rgb, color_chart):
    """Find concentration based on the highest RGB value"""
    r, g, b = rgb
    
    # Find the primary color (highest value)
    max_val = max(r, g, b)
    primary_color = 'r' if r == max_val else 'g' if g == max_val else 'b'
    
    # Get all concentrations where the primary color matches
    chart_df = color_chart.df
    if primary_color == 'r':
        matches = chart_df[chart_df['Red'] == max_val]
    elif primary_color == 'g':
        matches = chart_df[chart_df['Green'] == max_val]
    else:
        matches = chart_df[chart_df['Blue'] == max_val]
    
    if len(matches) == 0:
        # If no exact match, find closest
        return color_chart.find_closest(rgb)
    
    # Get the closest match from the filtered results
    matches_colors = matches[['Red', 'Green', 'Blue']].values
    dists = np.linalg.norm(matches_colors - np.array(rgb), axis=1)
    idx = np.argmin(dists)
    
    return {
        'concentration': float(matches.iloc[idx]['Concentration_mg_L']),
        'hex': str(matches.iloc[idx]['Hex']),
        'rgb': (int(matches.iloc[idx]['Red']), 
                int(matches.iloc[idx]['Green']), 
                int(matches.iloc[idx]['Blue'])),
        'distance': float(dists[idx])
    }

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
            
            # Get color from center circle
            dominant_color = get_center_circle_color(image)
            if dominant_color is None:
                return jsonify({
                    "error": "Invalid sample",
                    "details": "Could not detect valid color in the center circle. Please ensure the sample is properly prepared."
                }), 400
            
            # OpenCV loads as BGR, convert to RGB
            b, g, r = dominant_color
            color_hex = '#{:02x}{:02x}{:02x}'.format(r, g, b)
            logger.debug(f"Detected color: {color_hex} RGB: ({r},{g},{b})")

            # Find concentration based on primary color
            match = find_concentration_by_primary_color((r, g, b), color_chart)
            logger.debug(f"Matched chart color: {match['hex']} RGB: {match['rgb']} -> {match['concentration']} mg/L (distance: {match['distance']:.2f})")

            # Convert match values to Python native types
            match_rgb = [int(x) for x in match['rgb']]
            match_distance = float(match['distance'])
            match_concentration = float(match['concentration'])

            # Save to dataset
            if not os.path.exists(DATASET_PATH):
                pd.DataFrame(columns=[
                    'timestamp', 'image_path', 'concentration', 
                    'color_hex', 'color_r', 'color_g', 'color_b'
                ]).to_csv(DATASET_PATH, index=False)
            
            new_data = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(),
                'image_path': img_path,
                'concentration': match_concentration,
                'color_hex': match['hex'],
                'color_r': r,
                'color_g': g,
                'color_b': b
            }])
            new_data.to_csv(DATASET_PATH, mode='a', header=False, index=False)
            
            history = get_concentration_history()
            
            return jsonify({
                "ammonia_concentration": match_concentration,
                "success": True,
                "saved_image": img_filename,
                "original_color":  {
                    "hex": color_hex,
                    "rgb": {
                        "r": r,
                        "g": g,
                        "b": b
                    }
                },
                "color": {
                    "hex": str(match['hex']),
                    "rgb": {
                        "r": match_rgb[0],
                        "g": match_rgb[1],
                        "b": match_rgb[2]
                    }
                },
                "distance": match_distance,
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

def enhance_scanned_document(image, brightness=1.2, contrast=1.5):
    """
    Advanced image enhancement specifically designed for scanned documents
    """
    # Convert to float for better precision
    img = image.astype(np.float32)
    
    # Step 1: Noise reduction while preserving edges
    # Use bilateral filter to reduce noise while keeping edges sharp
    img = cv2.bilateralFilter(img.astype(np.uint8), 9, 75, 75).astype(np.float32)
    
    # Step 2: Correct perspective and skew (basic version)
    # For production, you'd want to add automatic skew detection
    
    # Step 3: Advanced contrast enhancement using CLAHE
    # Convert to LAB color space for better luminance processing
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    # Merge back
    lab[:, :, 0] = l_channel
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    
    # Step 4: Gamma correction for better midtone contrast
    gamma = 0.8  # Brighten midtones
    img = np.power(img / 255.0, gamma) * 255.0
    
    # Step 5: Advanced brightness and contrast adjustment
    # Use adaptive adjustment based on local statistics
    mean_val = np.mean(img)
    std_val = np.std(img)
    
    # Adaptive brightness - adjust less for already bright images
    adaptive_brightness = brightness * (1.0 - mean_val / 255.0 * 0.5)
    img = img * adaptive_brightness
    
    # Adaptive contrast with sigmoid function for smooth transitions
    img = img / 255.0  # Normalize to 0-1
    img = 1.0 / (1.0 + np.exp(-contrast * (img - 0.5)))
    img = img * 255.0  # Scale back
    
    # Step 6: Sharpening using unsharp masking
    gaussian = cv2.GaussianBlur(img.astype(np.uint8), (0, 0), 2.0)
    unsharp_mask = img - gaussian
    img = img + unsharp_mask * 0.8  # Adjust sharpening strength
    
    # Step 7: Advanced histogram stretching
    # Use percentile-based stretching to avoid outliers
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip((img - p1) / (p99 - p1) * 255.0, 0, 255)
    
    # Step 8: Final cleanup - remove extreme noise
    img = cv2.medianBlur(img.astype(np.uint8), 3)
    
    return img.astype(np.uint8)

def auto_enhance_document(image):
    """
    Automatic document enhancement with optimal settings
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyze image characteristics
    mean_brightness = np.mean(gray)
    contrast_measure = np.std(gray)
    
    # Determine optimal enhancement parameters based on image analysis
    if mean_brightness < 100:  # Dark image
        brightness = 1.8
        contrast = 2.2
    elif mean_brightness > 180:  # Bright image
        brightness = 0.9
        contrast = 1.8
    else:  # Normal brightness
        brightness = 1.3
        contrast = 2.0
    
    # Adjust for low contrast images
    if contrast_measure < 30:
        contrast *= 1.5
    
    return enhance_scanned_document(image, brightness, contrast)

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        logger.debug(f"Process image request received - Content-Type: {request.content_type}")
        
        # Get enhancement mode
        enhancement_mode = request.form.get('mode', 'auto')  # 'auto' or 'manual'
        
        # Get manual parameters if specified
        brightness = float(request.form.get('brightness', 1.3))
        contrast = float(request.form.get('contrast', 2.0))
        
        # Handle file upload - improved handling
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
                        with open(image_uri[7:], 'rb') as f:
                            file_bytes = f.read()
                    else:
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
            logger.error("No valid file data found in request")
            return jsonify({
                "error": "No file uploaded",
                "details": "No valid file data found in the request"
            }), 400

        # Decode image
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image data")
            return jsonify({
                "error": "Invalid image format",
                "details": "Could not decode the image data. Supported formats: JPG, PNG, BMP, TIFF"
            }), 400

        logger.debug(f"Image decoded successfully: {image.shape}")

        # Apply enhancement based on mode
        if enhancement_mode == 'auto':
            enhanced_image = auto_enhance_document(image)
            logger.debug("Applied automatic enhancement")
        else:
            enhanced_image = enhance_scanned_document(image, brightness, contrast)
            logger.debug(f"Applied manual enhancement - brightness: {brightness}, contrast: {contrast}")
        
        # Encode to high-quality JPEG
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 98,
            cv2.IMWRITE_JPEG_OPTIMIZE, True
        ]
        
        success, buffer = cv2.imencode('.jpg', enhanced_image, encode_params)
        
        if not success:
            logger.error("Failed to encode processed image")
            return jsonify({
                "error": "Encoding error",
                "details": "Failed to encode the processed image"
            }), 500
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.debug("Image processing completed successfully")
        
        return jsonify({
            "success": True,
            "imageUri": f"data:image/jpeg;base64,{image_base64}",
            "processingInfo": {
                "mode": enhancement_mode,
                "originalSize": image.shape,
                "brightness": brightness if enhancement_mode == 'manual' else 'auto',
                "contrast": contrast if enhancement_mode == 'manual' else 'auto'
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=5000, debug=True)
