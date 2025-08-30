# ================================
# Flask Backend for Forest Incident Detection (Render-ready) with Logging
# ================================

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import geopandas as gpd
from shapely.geometry import Point
from shapely.validation import make_valid
import datetime

# ----------------------
# Flask App
# ----------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------
# Model Paths (place models in project folder)
# ----------------------
dumping_model_path = "dumping_model.pth"
cutting_model_path = "cutting_model.pth"

# ----------------------
# Transform
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------
# Load Models
# ----------------------
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes

dumping_model, dumping_classes = load_model(dumping_model_path)
cutting_model, cutting_classes = load_model(cutting_model_path)

# ----------------------
# Prediction Function
# ----------------------
def predict_image(model, classes, img, threshold=0.75):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    class_name = classes[pred.item()]
    confidence = conf.item()
    if class_name.lower() == "yes" and confidence < threshold:
        class_name = "no"
    return class_name, confidence

# ----------------------
# Mangrove Check
# ----------------------
def is_in_mangrove(lat, lon, filepath="clipped_gmw.json", buffer_km=10):
    try:
        mangrove = gpd.read_file(filepath)
        if mangrove.empty:
            return 0
        mangrove["geometry"] = mangrove.geometry.apply(
            lambda g: make_valid(g) if g is not None and not g.is_valid else g
        )
        mangrove = mangrove[mangrove.geometry.notnull()]
        point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        mangrove_m = mangrove.to_crs(epsg=3857)
        point_m = point.to_crs(epsg=3857)
        buffer_geom = point_m.buffer(buffer_km * 1000)
        intersects = mangrove_m.intersects(buffer_geom.iloc[0])
        return 1 if intersects.any() else 0
    except:
        return 0

# ----------------------
# Logging
# ----------------------
def log_request(data):
    with open("requests_log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} | {data}\n")

# ----------------------
# Flask Route for Prediction
# ----------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    lat = data.get("lat", type=float)
    lon = data.get("lon", type=float)
    
    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude required"}), 400
    
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    img = Image.open(request.files["image"].stream).convert("RGB")

    # First: check mangrove
    mangrove_check = is_in_mangrove(lat, lon)
    
    if mangrove_check == 0:
        final_pred = 0
        response = {"final_result": final_pred, "reason": "Point not in/near mangrove"}
    else:
        # Second: check image models
        dumping_pred, dumping_conf = predict_image(dumping_model, dumping_classes, img)
        cutting_pred, cutting_conf = predict_image(cutting_model, cutting_classes, img)
        final_pred = 1 if dumping_pred.lower() == "yes" or cutting_pred.lower() == "yes" else 0
        response = {
            "final_result": final_pred,
            "dumping": {"prediction": dumping_pred, "confidence": dumping_conf},
            "cutting": {"prediction": cutting_pred, "confidence": cutting_conf},
            "mangrove_check": mangrove_check
        }

    # Log request + response
    log_request({"lat": lat, "lon": lon, "response": response})

    return jsonify(response)

# ----------------------
# Run Flask (Render uses PORT env)
# ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
