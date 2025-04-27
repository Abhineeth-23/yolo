from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import onnxruntime
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or you can specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI()

# Load the ONNX model once
session = onnxruntime.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    img = np.array(image)

    # Preprocess according to YOLOv8 input needs
    img = img.transpose(2, 0, 1)  # Channels first
    img = img[np.newaxis, ...].astype('float32') / 255.0

    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)

    return JSONResponse(content={"output": outputs})
