import os
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
import torch
import io
import json
import timm
import torch.nn as nn
import numpy as np
import cv2

app = Flask(__name__)

# Configure the Flask app logger
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)

num_classes = 61

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = timm.create_model("rexnet_150", pretrained=True, num_classes=num_classes).to(device)

# Transform code
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
tfs = transforms.Compose([transforms.Resize((im_size, im_size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

class SaveFeatures():
    """ Extract pretrained activations"""
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def getCAM(conv_fs, linear_weights, class_idx):
    bs, chs, h, w = conv_fs.shape
    cam = linear_weights[class_idx].dot(conv_fs[0, :, :, ].reshape((chs, h * w)))
    cam = cam.reshape(h, w)

    return (cam - np.min(cam)) / np.max(cam)

def test_single_image(model, device, image, final_conv, fc_params, cls_names=None):
    weight = np.squeeze(fc_params[0].cpu().data.numpy())
    activated_features = SaveFeatures(final_conv)

    # Move image to device and make prediction
    image = image.to(device)
    pred_class = torch.argmax(model(image.unsqueeze(0)), dim=1).item()

    # Get CAM for the predicted class
    heatmap = getCAM(activated_features.features, weight, pred_class)
    # Clean up
    activated_features.remove()

    return pred_class, heatmap

# Preprocess the image
def preprocess_image(original_image):
    try:
        # Apply the desired transformations (resize, normalize, etc.)
        transform = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Convert RGBA to RGB if the image has an alpha channel
        if original_image.mode == 'RGBA':
            original_image = original_image.convert('RGB')

        # Apply the transformations to the original image
        transformed_image = transform(original_image)

        # Add a batch dimension to the input tensor
        transformed_image = transformed_image.unsqueeze(0)

        # Make predictions using the model
        prediction_result = predict_image(transformed_image)

        return prediction_result

    except Exception as e:
        app.logger.error(f"Error during photo upload and prediction: {str(e)}")
        return {"status": 500, "message": "Internal Server Error", "error_details": str(e)}

def predict_image(transformed_image):
    try:
        # Ensure the model is in evaluation mode
        m.eval()

        # Forward pass through the model
        with torch.no_grad():
            # Pass the transformed image through the model
            output = m(transformed_image.to(device))

        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=-1)[0]

        # Get the predicted class index
        predicted_class_index = torch.argmax(probabilities).item()

        # Directly access label using index
        predicted_class_label = list(classes.keys())[predicted_class_index]

        # Create a JSON response
        result = {
            "status": 200,
            "message": "Prediction successful",
            "predicted_class_label": predicted_class_label,
            #"probabilities": probabilities.tolist(),
            #"input_tensor_shape": transformed_image.shape[1:]  # Exclude batch dimension
        }

        return result

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return {"status": 500, "message": "Internal Server Error", "error_details": str(e)}

# Load the model weights from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

model_weights_path = config['model_path']
# Example usage:
m.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
m.eval()
final_conv, fc_params = m.features[-1], list(m.head.fc.parameters())

# Dictionary of plant disease classes
classes = {
    'Apple___alternaria_leaf_spot': 0,
    'Apple___black_rot': 1,
    'Apple___brown_spot': 2,
    'Apple___gray_spot': 3,
    'Apple___healthy': 4,
    'Apple___rust': 5,
    'Apple___scab': 6,
    'Bell_pepper___bacterial_spot': 7,
    'Bell_pepper___healthy': 8,
    'Blueberry___healthy': 9,
    'Cassava___bacterial_blight': 10,
    'Cassava___brown_streak_disease': 11,
    'Cassava___green_mottle': 12,
    'Cassava___healthy': 13,
    'Cassava___mosaic_disease': 14,
    'Cherry___healthy': 15,
    'Cherry___powdery_mildew': 16,
    'Corn___common_rust': 17,
    'Corn___gray_leaf_spot': 18,
    'Corn___healthy': 19,
    'Corn___northern_leaf_blight': 20,
    'Grape___black_measles': 21,
    'Grape___black_rot': 22,
    'Grape___healthy': 23,
    'Grape___isariopsis_leaf_spot': 24,
    'Grape_leaf_blight': 25,
    'Orange___citrus_greening': 26,
    'Peach___bacterial_spot': 27,
    'Peach___healthy': 28,
    'Potato___bacterial_wilt': 29,
    'Potato___early_blight': 30,
    'Potato___healthy': 31,
    'Potato___late_blight': 32,
    'Potato___nematode': 33,
    'Potato___pests': 34,
    'Potato___phytophthora': 35,
    'Potato___virus': 36,
    'Raspberry___healthy': 37,
    'Rice___bacterial_blight': 38,
    'Rice___blast': 39,
    'Rice___brown_spot': 40,
    'Rice___tungro': 41,
    'Soybean___healthy': 42,
    'Squash___powdery_mildew': 43,
    'Strawberry___healthy': 44,
    'Strawberry___leaf_scorch': 45,
    'Sugarcane___healthy': 46,
    'Sugarcane___mosaic': 47,
    'Sugarcane___red_rot': 48,
    'Sugarcane___rust': 49,
    'Sugarcane___yellow_leaf': 50,
    'Tomato___bacterial_spot': 51,
    'Tomato___early_blight': 52,
    'Tomato___healthy': 53,
    'Tomato___late_blight': 54,
    'Tomato___leaf_curl': 55,
    'Tomato___leaf_mold': 56,
    'Tomato___mosaic_virus': 57,
    'Tomato___septoria_leaf_spot': 58,
    'Tomato___spider_mites': 59,
    'Tomato___target_spot': 60
}

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if it is from allowed files
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask endpoint to upload a photo and make predictions
@app.route("/upload-photos-and-predict", methods=["POST"])
def upload_photos_and_predict():
    try:
        # Check if the request has the file part
        if 'file' not in request.files:
            result = {"status": 404, "message": "No file part in the request"}
            return jsonify(result), 404

        # Get the file from the request
        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            result = {"status": 404, "message": "No selected file"}
            return jsonify(result), 404

        # Check if the file extension is allowed
        if not allowed_file(file.filename):
            result = {"status": 400, "message": "File extension not allowed"}
            return jsonify(result), 400

        # Read the image file
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        img = preprocess_image(img)

        return jsonify(img), 200

    except Exception as e:
        app.logger.error(f"Unhandled error during photo upload and prediction: {str(e)}")
        result = {"status": 500, "message": "Internal Server Error", "error_details": str(e)}
        return jsonify(result), 500

# Load disease data from JSON file
with open('Treatment_Recommendions.json', 'r') as f:
    disease_data = json.load(f)

# Define a new endpoint to get disease recommendations
@app.route("/get-disease-recommendation", methods=["GET"])
def get_disease_recommendation():
    try:
        # Extract disease name from query parameters
        disease_name = request.args.get('disease_name')

        # Check if disease name is provided
        if not disease_name:
            result = {"status": 400, "message": "Bad Request: 'disease_name' query parameter is missing"}
            return jsonify(result), 400

        # Search for disease recommendation
        recommendation = None
        for disease in disease_data:
            if disease['name'] == disease_name:
                recommendation = disease['description']
                break

        if recommendation:
            result = {"status": 200, "message": "Disease recommendation found", "recommendation": recommendation}
            return jsonify(result), 200
        else:
            result = {"status": 404, "message": "Disease recommendation not found"}
            return jsonify(result), 404

    except Exception as e:
        app.logger.error(f"Error retrieving disease recommendation: {str(e)}")
        result = {"status": 500, "message": "Internal Server Error", "error_details": str(e)}
        return jsonify(result), 500

# Custom error handler for 404 errors
@app.errorhandler(404)
def not_found_error(error):
    result = {"status": 404, "message": "Endpoint not found"}
    return jsonify(result), 404

@app.errorhandler(400)
def not_found_error(error):
    result = {"status": 400, "message": "Bad Request"}
    return jsonify(result), 400

# Custom error handler for generic errors
@app.errorhandler(Exception)
def handle_generic_error(error):
    app.logger.error(f"Unhandled error: {str(error)}")
    result = {"status": 500, "message": "Internal Server Error", "error_details": str(error)}
    return jsonify(result), 500

if __name__ == '__main__':
    app.run(debug=True)
