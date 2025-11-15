
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_SIZE = 256

model_path = r"F:\Deep Learning Projects\Project 4\models\model_1.keras"
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load Keras model.")

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG image files are allowed.")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    try:
        pred = model.predict(img_array)
        
        idx = np.argmax(pred, axis=1)[0]
        label = class_names[idx]
        confidence = float(np.max(pred))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": label,
        "confidence_percentage": round(confidence * 100, 2),
        "confidence_scores": pred[0].tolist()
    })