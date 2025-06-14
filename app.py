import streamlit as st
import numpy as np
import torch
import cv2
import sys
from pathlib import Path
import types

# Fix for torch.classes error in Streamlit
torch.classes.__path__ = types.SimpleNamespace(_path=[])

# Streamlit config
st.set_page_config(page_title="Backpack Detector", layout="centered")
st.title("🎒 Backpack Detection with YOLOv5")

# Load model
@st.cache_resource
def load_model():
    yolov5_path = Path(__file__).parent / 'yolov5'
    sys.path.append(str(yolov5_path))
    from models.common import DetectMultiBackend
    weights = yolov5_path / 'runs' / 'train' / 'debug-test' / 'weights' / 'best.pt'
    model = DetectMultiBackend(str(weights), device='cpu')
    return model



# Load NMS utility
yolov5_path = Path(__file__).parent / 'yolov5'
sys.path.append(str(yolov5_path))
from utils.general import non_max_suppression

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img0 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocess image
    img_resized = cv2.resize(img0, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Run inference
    with torch.no_grad():
        raw_pred = model(img_tensor)
        pred = non_max_suppression(raw_pred, conf_thres=0.25, iou_thres=0.45)[0]

    # Draw bounding boxes
    if pred is not None and len(pred):
        pred = pred.cpu().numpy()
        for det in pred:
            if len(det) < 6:
                continue
            x1, y1, x2, y2 = [int(coord) for coord in det[:4]]
            conf = float(det[4])
            label = f"Backpack {conf:.2f}"
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB),
             caption="Detected Image", use_container_width=True)
