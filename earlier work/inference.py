import cv2 as cv
import numpy as np
# import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib

def get_embedding(face_img, embedder):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

# Load model and encoder
model = joblib.load('face_recognition_model.pkl')
encoder = joblib.load('label_encoder.pkl')
embedder = FaceNet()
detector = MTCNN()

# Load and preprocess test image
t_im = cv.imread("test/person/Photo on 2024-03-12 at 3.35 PM.jpg")
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x, y, w, h = detector.detect_faces(t_im)[0]['box']
t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160, 160))

# Get embedding
test_im = get_embedding(t_im, embedder)
test_im = np.expand_dims(test_im, axis=0)

# Prediction
ypreds = model.predict(test_im)
confidence_threshold = 0.5

if max(model.predict_proba(test_im)[0]) < confidence_threshold:
    print("Person not in dataset")
else:
    ypreds = model.predict(test_im)
    print(encoder.inverse_transform(ypreds))