from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

# 👉 Create the FastAPI app
app = FastAPI()

# 👉 Allow CORS (frontend -> backend calls allowed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👉 Load your trained YOLO model
model = YOLO("yolov8n.onnx")  # or your own model filename

# 👉 Create the API route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(image)

    # Prepare the output
    output = []
    for box in results[0].boxes:
        output.append({
            "class": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": box.xyxy[0].tolist()
        })

    return output
