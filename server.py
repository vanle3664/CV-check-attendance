# Load image uploaded by user
# Preprocessing
# Predict
# Return result
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
import numpy as np
from io import BytesIO
from PIL import Image
from retinaface import RetinaFace
import cv2
import joblib
import os

app = FastAPI()

global pca, model
classes = ['Thu', 'Chau', 'LeVan', 'VAnh', 'Linh', 'Thang',
           'Kien', 'Van', 'Tan', 'Quan', 'Tuan', 'Truong',
           'XAnh', 'Hieu', 'Hung', 'HDuc', 'Duc', 'VDuc', 'Unknown']

with open("pca.pkl", 'rb') as f1:
    pca = joblib.load(f1)


with open("model4.pkl", 'rb') as f2:
    model = joblib.load(f2)


@app.post("/predict-svm/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = Image.open(BytesIO(await file.read()))
    image = img_to_array(image)
    saved_image = cv2.imwrite("saved_image.jpg", image)

    res_face = RetinaFace.detect_faces("saved_image.jpg")
    axis = res_face['face_1']['facial_area']
    image = image[axis[1]:axis[3], axis[0]:axis[2]]

    face = cv2.imwrite("face.jpg", image)

    image = load_img("face.jpg")

    image = img_to_array(image.resize((299, 299)))
    image = np.expand_dims(image, 0)
    image = np.mean(image, axis=3)
    x = image.flatten()
    x = x.reshape(1, -1)

    x = pca.transform(x)

    res = model.predict(x)
    os.remove("face.jpg")
    os.remove("saved_image.jpg")

    return classes[res[0]]
