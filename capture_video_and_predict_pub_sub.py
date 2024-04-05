import cv2 as cv
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib
import time
from google.cloud import pubsub_v1  

# Initialize Pub/Sub client
publisher = pubsub_v1.PublisherClient()
project_id = "intelligate-security-system"  
topic_id = "intelligate_pub_sub" 
topic_path = publisher.topic_path(project_id, topic_id)

def publish_message(message):
    """Publishes a message to a Pub/Sub topic."""
    # Data must be a bytestring
    message_data = message.encode("utf-8")
    # Publish a message
    try:
        publish_future = publisher.publish(topic_path, data=message_data)
        publish_future.result()  # Verify the publish succeeded
        print(f"Message published to {topic_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Your existing code
# Start timing the entire script execution
script_start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
embedder = FaceNet()
model = joblib.load('face_recognition_model.pkl')
encoder = joblib.load('label_encoder.pkl')
detector = MTCNN()

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]

def get_frames_from_video(video_path, samples=5):
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    frames_to_sample = np.linspace(0, total_frames-1, samples, endpoint=False).astype(int)
    sampled_frames = []

    for i in frames_to_sample:
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_img)
            for face in faces:
                x, y, w, h = face['box']
                face_img = rgb_img[y:y+h, x:x+w]
                if face_img.size > 0:
                    sampled_frames.append(cv.resize(face_img, (160, 160)))
    cap.release()
    return sampled_frames

def predict_person_from_samples(frames):
    best_prediction = ("Unknown", 0.5)  # (Name, confidence)
    for face in frames:
        if face is not None:
            embedding = get_embedding(face)
            embedding = np.expand_dims(embedding, axis=0)
            prediction = model.predict(embedding)
            confidence = model.predict_proba(embedding).max()
            if confidence > best_prediction[1]:  # Confidence threshold
                person_name = encoder.inverse_transform(prediction)[0]
                best_prediction = (person_name, confidence)
    return best_prediction[0]

# Main
video_path = "captured_video.mp4"  # Specify the path to your video file here

# Process the video and predict person
sampled_frames = get_frames_from_video(video_path, 5)  # Use 5 frames from the video
person = predict_person_from_samples(sampled_frames)

# Print and publish the predicted person
predicted_person_message = f"Predicted person: {person}"
print(predicted_person_message)
publish_message(predicted_person_message)  # Publish to Pub/Sub

# End timing the entire script execution and print duration
script_end_time = time.time()
total_duration = script_end_time - script_start_time
print(f"Total script execution took {total_duration:.2f} seconds.")
