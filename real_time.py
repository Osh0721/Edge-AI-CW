import cv2 as cv
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib

# Ensure TensorFlow logging is controlled
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the FaceNet model for embeddings
embedder = FaceNet()

# Load the SVM model and label encoder
model = joblib.load('face_recognition_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# Initialize MTCNN for face detection
detector = MTCNN()

# Function to get embeddings using FaceNet
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

# Select webcam;
cap = cv.VideoCapture(0) 

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect faces in the frame
    results = detector.detect_faces(rgb_img)
    for result in results:
        x, y, w, h = result['box']
        face = rgb_img[y:y+h, x:x+w]
        face = cv.resize(face, (160, 160))

        # Generate embedding for the detected face
        embedding = get_embedding(face)
        embedding = np.expand_dims(embedding, axis=0)

        # Predict the identity of the face
        ypred = model.predict(embedding)
        if model.predict_proba(embedding).max() > 0.5:  # Confidence threshold
            final_name = encoder.inverse_transform(ypred)[0]
        else:
            final_name = "Unknown"

        # Display the identity and a bounding box around the face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


