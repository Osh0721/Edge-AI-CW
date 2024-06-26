import cv2 as cv
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib
import time 

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

def capture_video(video_path, capture_duration=1, resize_factor=0.2):
    cap = cv.VideoCapture(0)
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(3) * resize_factor)
    frame_height = int(cap.get(4) * resize_factor)
    total_frames = int(frame_rate * capture_duration)
    
    # Define the codec and create VideoWriter object with lower resolution
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            # Resize frame
            resized_frame = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)
            out.write(resized_frame)
            cv.imshow('Capturing Video', resized_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    out.release()
    cv.destroyAllWindows()

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
video_folder = "video_clip"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)
video_path = os.path.join(video_folder, "captured_video.mp4")

# Capture and save a 2-second video
capture_video(video_path)

# Process captured video and predict person
sampled_frames = get_frames_from_video(video_path, 5)  # Use all frames from the 2-second video
person = predict_person_from_samples(sampled_frames)

# Print the predicted person
print(f"Predicted person: {person}")

# End timing the entire script execution
script_end_time = time.time()

# Calculate and print the total duration
total_duration = script_end_time - script_start_time
print(f"Total script execution took {total_duration:.2f} seconds.")