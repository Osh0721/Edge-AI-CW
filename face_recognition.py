import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib  # For saving model and encoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.detector = MTCNN()
        self.X = []
        self.Y = []

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.detector.detect_faces(img)
        x, y, w, h = results[0]['box']
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        faces = []
        for im_name in os.listdir(dir):
            path = os.path.join(dir, im_name)
            try:
                single_face = self.extract_face(path)
                faces.append(single_face)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return faces

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            faces = self.load_faces(path)
            self.X.extend(faces)
            self.Y.extend([sub_dir] * len(faces))
        return np.asarray(self.X), np.asarray(self.Y)

def get_embedding(face_img, embedder):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

# Load dataset
faceloading = FACELOADING("Dataset")
X, Y = faceloading.load_classes()

embedder = FaceNet()
EMBEDDED_X = [get_embedding(img, embedder) for img in X]
EMBEDDED_X = np.asarray(EMBEDDED_X)

# Encode labels
encoder = LabelEncoder()
Y_enc = encoder.fit_transform(Y)

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_enc, shuffle=True, random_state=17)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

# Save model and encoder
joblib.dump(model, 'face_recognition_model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')
