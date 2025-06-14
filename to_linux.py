import torch
import sys
from pathlib import Path

# Add yolov5 path so Python can resolve the `models` module
yolov5_path = Path(__file__).parent / "yolov5"
sys.path.append(str(yolov5_path))

# Load model
model = torch.load("best.pt", map_location="cpu")

# Save in Linux-compatible format
torch.save(model, "best_linux.pt", _use_new_zipfile_serialization=False)
print("✅ Converted model saved as best_linux.pt")
