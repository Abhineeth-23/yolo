from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load the ONNX model
session = ort.InferenceSession('yolov8n.onnx')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    # Preprocess the image
    img = img.resize((640, 640))  # YOLOv8 input size
    img_data = np.array(img) / 255.0
    img_data = img_data.transpose(2, 0, 1)  # Channels first
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

    # Run inference
    inputs = {session.get_inputs()[0].name: img_data}
    outputs = session.run(None, inputs)

    # Return raw model outputs (later we can improve)
    return JSONResponse(content={"outputs": outputs[0].tolist()})
