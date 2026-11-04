
import os
import json
import logging
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from tensorflow.keras.models import load_model


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "plant_model.keras"
CLASS_PATH = "class_names.json"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

IMAGE_SIZE = (224, 224)

model = None
class_names = []


# ==========================================
# 2. PLANT DISEASE TREATMENT DATABASE
# ==========================================

TREATMENTS = {

    "healthy": {
        "status": "Healthy",
        "disease": "No Disease Detected",
        "treatment": [
            "Continue regular watering.",
            "Provide sufficient sunlight.",
            "Use organic fertilizer.",
            "Monitor plant growth regularly."
        ],
        "prevention": "Maintain proper soil moisture and plant hygiene."
    },

    "early_blight": {
        "status": "Diseased",
        "disease": "Early Blight",
        "treatment": [
            "Remove infected leaves.",
            "Avoid watering leaves directly.",
            "Improve air circulation.",
            "Use a suitable fungicide according to its label."
        ],
        "prevention": "Avoid excess moisture and maintain plant spacing."
    },

    "late_blight": {
        "status": "Diseased",
        "disease": "Late Blight",
        "treatment": [
            "Remove severely infected plant parts.",
            "Avoid overhead irrigation.",
            "Keep the garden clean.",
            "Consult an agricultural expert about suitable fungicides."
        ],
        "prevention": "Avoid prolonged leaf wetness and monitor plants regularly."
    },

    "leaf_mold": {
        "status": "Diseased",
        "disease": "Leaf Mold",
        "treatment": [
            "Remove infected leaves.",
            "Improve air circulation.",
            "Avoid excess watering.",
            "Use an appropriate fungicide if recommended."
        ],
        "prevention": "Maintain proper ventilation and avoid high humidity."
    },

    "bacterial_spot": {
        "status": "Diseased",
        "disease": "Bacterial Spot",
        "treatment": [
            "Remove infected leaves carefully.",
            "Avoid touching healthy plants after infected plants.",
            "Disinfect gardening tools.",
            "Use only locally recommended disease management products."
        ],
        "prevention": "Use clean seeds and avoid splashing water between plants."
    },

    "mosaic_virus": {
        "status": "Diseased",
        "disease": "Mosaic Virus",
        "treatment": [
            "Remove severely infected plants.",
            "Control insect vectors such as aphids.",
            "Remove weeds around plants.",
            "Do not reuse infected plant material."
        ],
        "prevention": "Use healthy seeds and control insect pests."
    },

    "powdery_mildew": {
        "status": "Diseased",
        "disease": "Powdery Mildew",
        "treatment": [
            "Remove badly affected leaves.",
            "Improve sunlight and air circulation.",
            "Avoid excessive nitrogen fertilizer.",
            "Use a suitable fungicide when necessary."
        ],
        "prevention": "Maintain proper spacing and avoid excess humidity."
    }
}


# ==========================================
# 3. LOAD AI MODEL
# ==========================================

def load_ai_model():

    global model, class_names

    try:

        if not os.path.exists(MODEL_PATH):
            logger.warning("AI model file not found.")
            return

        model = load_model(MODEL_PATH)

        logger.info("AI Model loaded successfully!")

        if os.path.exists(CLASS_PATH):

            with open(CLASS_PATH, "r", encoding="utf-8") as f:
                class_names = json.load(f)

            if not isinstance(class_names, list) or not class_names:
                raise ValueError("Invalid class_names.json")

            logger.info("Disease classes loaded successfully!")

        else:
            logger.warning("class_names.json not found.")

    except Exception:
        logger.exception("Model loading failed.")
        model = None


load_ai_model()


# ==========================================
# 4. IMAGE VALIDATION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(file):

    image = Image.open(BytesIO(file.read()))

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    # Normalize image pixels
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ==========================================
# 5. FIND DISEASE TREATMENT
# ==========================================

def get_treatment(prediction):

    disease_key = prediction.lower()

    disease_key = disease_key.replace(" ", "_")
    disease_key = disease_key.replace("-", "_")

    for key, value in TREATMENTS.items():

        if key in disease_key:
            return value

    if "healthy" in disease_key:

        return TREATMENTS["healthy"]

    return {
        "status": "Needs Attention",
        "disease": prediction,
        "treatment": [
            "Keep the plant isolated if possible.",
            "Remove severely damaged leaves.",
            "Check watering and sunlight conditions.",
            "Consult an agricultural expert for confirmation."
        ],
        "prevention": "Maintain good plant hygiene and monitor symptoms."
    }


# ==========================================
# 6. HOME API
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "success": True,

        "project": "Plant Health Detection System",

        "message": "Welcome to Plant Health Detection API",

        "model_status": "Loaded" if model else "Not Loaded",

        "version": "1.0.0"

    })


# ==========================================
# 7. PLANT DISEASE PREDICTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:

            return jsonify({

                "success": False,

                "message": "AI model is not loaded. Please check model files."

            }), 503

        if not class_names:

            return jsonify({

                "success": False,

                "message": "Disease class names are missing."

            }), 503

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "message": "Please upload a plant image."

            }), 400

        file = request.files["image"]

        if not file.filename:

            return jsonify({

                "success": False,

                "message": "No image selected."

            }), 400

        filename = secure_filename(file.filename)

        if not allowed_file(filename):

            return jsonify({

                "success": False,

                "message": "Only JPG, PNG, JPEG and WEBP images allowed."

            }), 400

        # Process uploaded image
        image_array = preprocess_image(file)

        # AI Model Prediction
        predictions = model.predict(image_array, verbose=0)

        probabilities = np.asarray(predictions[0]).flatten()

        if len(probabilities) != len(class_names):

            return jsonify({

                "success": False,

                "message": "Model classes do not match class_names.json."

            }), 500

        if not np.all(np.isfinite(probabilities)):

            return jsonify({

                "success": False,

                "message": "Invalid prediction values."

            }), 500

        predicted_index = int(np.argmax(probabilities))

        confidence = float(probabilities[predicted_index]) * 100

        disease_name = str(class_names[predicted_index])

        treatment_info = get_treatment(disease_name)

        # Confidence validation
        if confidence < 60:

            return jsonify({

                "success": True,

                "disease": "Uncertain Prediction",

                "confidence": round(confidence, 2),

                "status": "Low Confidence",

                "message": "Please upload a clear plant leaf image.",

                "recommendation": [
                    "Take another image in good lighting.",
                    "Keep the leaf clearly visible.",
                    "Consult an agricultural expert."
                ]

            })

        # Final response
        return jsonify({

            "success": True,

            "project": "Plant Health Detection",

            "filename": filename,

            "disease": treatment_info["disease"],

            "original_prediction": disease_name,

            "status": treatment_info["status"],

            "confidence": round(confidence, 2),

            "treatment": treatment_info["treatment"],

            "prevention": treatment_info["prevention"],

            "message": "Plant analysis completed successfully!"

        })

    except UnidentifiedImageError:

        return jsonify({

            "success": False,

            "message": "Uploaded file is not a valid image."

        }), 400

    except Exception:

        logger.exception("Prediction error occurred.")

        return jsonify({

            "success": False,

            "message": "Something went wrong during plant analysis."

        }), 500


# ==========================================
# 8. HEALTH CHECK API
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "success": True,

        "server": "Running",

        "model": "Ready" if model else "Not Ready"

    })


# ==========================================
# 9. ERROR HANDLING
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "message": "Image size must be below 10 MB."

    }), 413


@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "message": "API endpoint not found."

    }), 404


# ==========================================
# 10. START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(os.environ.get("PORT", 5000)),

        debug=False

    )

