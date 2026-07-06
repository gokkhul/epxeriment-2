import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from model_handler import GestureClassifier

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

DATA_DIR = "data_collected"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Classifier
classifier = GestureClassifier()

@app.route('/')
def index():
    """Serves the main application page."""
    return render_template('index.html')

@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    """Gets the list of collected gestures and the current trained model status."""
    # Gestures with saved data
    collected_gestures = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith('.json'):
                gesture_name = os.path.splitext(f)[0]
                try:
                    with open(os.path.join(DATA_DIR, f), 'r') as file:
                        samples = json.load(file)
                        count = len(samples)
                except Exception:
                    count = 0
                collected_gestures.append({
                    "name": gesture_name,
                    "samples": count
                })
                
    return jsonify({
        "gestures": collected_gestures,
        "model_trained": classifier.model is not None,
        "trained_labels": classifier.labels
    })

@app.route('/api/gestures', methods=['POST'])
def add_gesture():
    """Creates a new gesture class by preparing an empty data file."""
    data = request.json or {}
    gesture_name = data.get('name', '').strip()
    
    if not gesture_name:
        return jsonify({"error": "Gesture name cannot be empty"}), 400
        
    # Clean name to be safe for filenames
    safe_name = "".join(c for c in gesture_name if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe_name:
        return jsonify({"error": "Invalid gesture name"}), 400
        
    file_path = os.path.join(DATA_DIR, f"{safe_name}.json")
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)
            
    return jsonify({
        "message": f"Gesture '{safe_name}' initialized.",
        "gesture": {
            "name": safe_name,
            "samples": 0
        }
    })

@app.route('/api/gestures/<name>', methods=['DELETE'])
def delete_gesture(name):
    """Deletes a gesture's saved landmark data."""
    file_path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        # Reload model if necessary, or just reload classifier metadata
        classifier.load_model_and_labels()
        return jsonify({"message": f"Gesture '{name}' deleted successfully."})
    return jsonify({"error": "Gesture not found"}), 404

@app.route('/api/gestures/reset', methods=['POST'])
def reset_dataset():
    """Deletes all collected gesture data and models."""
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith('.json'):
                os.remove(os.path.join(DATA_DIR, f))
                
    # Delete model files
    from model_handler import MODEL_PATH, LABELS_PATH
    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
        except Exception:
            pass
    if os.path.exists(LABELS_PATH):
        try:
            os.remove(LABELS_PATH)
        except Exception:
            pass
            
    classifier.load_model_and_labels()
    return jsonify({"message": "Dataset and models reset successfully."})

@app.route('/api/collect', methods=['POST'])
def collect_data():
    """Appends recorded landmark frames to the selected gesture file."""
    data = request.json or {}
    gesture_name = data.get('gesture', '').strip()
    landmarks_list = data.get('landmarks', []) # List of Lists of 63 floats (multiple frames)
    
    if not gesture_name:
        return jsonify({"error": "Gesture name is required"}), 400
    if not landmarks_list:
        return jsonify({"error": "No landmarks provided"}), 400
        
    file_path = os.path.join(DATA_DIR, f"{gesture_name}.json")
    
    # Read existing
    existing_samples = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                existing_samples = json.load(f)
        except Exception:
            existing_samples = []
            
    # Append new frames (limit data size if too large to prevent server abuse, e.g. max 1000 frames)
    for sample in landmarks_list:
        if len(sample) == 63:
            existing_samples.append(sample)
            
    # Keep it clean
    with open(file_path, 'w') as f:
        json.dump(existing_samples, f)
        
    return jsonify({
        "message": f"Successfully added {len(landmarks_list)} frames.",
        "samples_count": len(existing_samples)
    })

@app.route('/api/train', methods=['POST'])
def train():
    """Triggers model training on the backend."""
    success, message = classifier.train_model(DATA_DIR)
    if success:
        # Reload the trained model into memory
        classifier.load_model_and_labels()
        return jsonify({
            "status": "success",
            "message": message,
            "labels": classifier.labels
        })
    else:
        return jsonify({
            "status": "error",
            "message": message
        }), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predicts the gesture from the raw landmarks sent by the client."""
    data = request.json or {}
    raw_landmarks = data.get('landmarks', [])
    
    if not raw_landmarks or len(raw_landmarks) != 63:
        return jsonify({"error": "Invalid landmarks length. Must be 63 coordinates."}), 400
        
    prediction = classifier.predict(raw_landmarks)
    if prediction is None:
        return jsonify({
            "status": "no_model",
            "message": "Model not trained or available yet."
        })
        
    return jsonify({
        "status": "success",
        "class": prediction["class"],
        "confidence": prediction["confidence"]
    })

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
________________________________________

