# 🎒 Backpack Detection App (YOLOv5 + Streamlit)

This web app uses a custom-trained YOLOv5 model to detect **backpacks** in uploaded images. Built with **Streamlit**, it's deployable locally or on **Render** with full support for model file access and image uploads.

---

## 🚀 Demo

- 🔗 Live URL (if deployed): `https://your-app-name.onrender.com`
- 📸 Upload an image → Get bounding boxes + confidence scores for detected backpacks.

---

## 📁 Folder Structure

```
BackPack-Detection/
├── app.py                      # Streamlit interface
├── yolov5/                     # YOLOv5 repo (unchanged)
│   └── runs/train/debug-test/weights/best.pt  # trained model
├── requirements.txt            # pip dependencies
├── render.yaml                 # Render deployment config
├── README.md                   # this file
```

---

## ⚙️ How It Works

- Loads a YOLOv5 model using `torch.hub.load()`
- Accepts `.jpg`, `.png`, `.jpeg` images via upload
- Inference and visualization happen in real-time
- Annotated image displayed using OpenCV + PIL

---

## ✅ Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/backpack-detection.git
cd backpack-detection
```

### 2. Set Up Environment
```bash
conda create -n BackPack-Detection python=3.10
conda activate BackPack-Detection
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the App
```bash
streamlit run app.py
```

Visit: `http://localhost:8501`

---

## 🌐 Deploy to Render

### 1. Push Code to GitHub
```bash
git init
git remote add origin https://github.com/yourusername/backpack-detection.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 2. Deploy on [https://render.com](https://render.com)

- Click "New Web Service"
- Connect to your GitHub
- It auto-detects `render.yaml`
- Click **Deploy**

---

## 🛠 render.yaml

```yaml
services:
  - type: web
    name: backpack-detection
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port 10000
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
```

---

## 📦 requirements.txt

```txt
streamlit
torch
opencv-python
Pillow
numpy
```

---

## 🧠 Training the Model

YOLOv5 training command (optional):

```bash
cd yolov5
python train.py --img 416 --batch 16 --epochs 3 --data yourdata.yaml --weights yolov5s.pt
```

Ensure trained model is saved to:
```
yolov5/runs/train/debug-test/weights/best.pt
```

---

## 📸 Screenshots

Include a screenshot in your repo as `screenshot.png` and reference it here:

```
![Detection Demo](screenshot.png)
```

---

## 👤 Author

**Amir Etemadi, Ph.D., PE**  
🔗 [LinkedIn](https://linkedin.com/in/amir-etemadi-phd-pe)  
📧 etemadi@gwu.edu

---

## 📜 License

MIT License – free to use and modify.
