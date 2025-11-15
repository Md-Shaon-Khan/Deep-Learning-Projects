# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf  # <-- Must import tensorflow
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import io

app = FastAPI()

IMAGE_SIZE = 256

# Load model once
model_path = r"F:\Deep Learning Projects\Project 4\models\model_1.keras"
model = tf.keras.models.load_model(model_path)

# Correct class order from training
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Preprocessing layer exactly as used in training
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255),
])

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = resize_and_rescale(img_array)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG allowed.")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    pred = model.predict(img_array)
    idx = np.argmax(pred, axis=1)[0]
    label = class_names[idx]

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": label,
        "confidence_scores": pred[0].tolist()
    })
