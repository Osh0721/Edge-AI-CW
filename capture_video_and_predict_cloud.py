import cv2 as cv
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib
import time
import subprocess
import mysql.connector
from datetime import datetime
import pytz
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt


# Start timing the entire script execution
script_start_time = time.time()

# Database configuration
db_config = {
    'host': '34.168.6.246',
    'user': 'IntelligateUser',
    'password': 'Intelligate@123',
    'database': 'IntelliGate'
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
embedder = FaceNet()
model = joblib.load('trained_model/face_recognition_model.pkl')
encoder = joblib.load('trained_model/label_encoder.pkl')
detector = MTCNN()

MQTT_SERVER = "192.168.8.119" 
MQTT_PATH = "person_detected"


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

def get_emp_id_by_name(name):
    # Connect to the database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        sql = "SELECT Emp_ID FROM Emp_data WHERE Name LIKE  %s"
        cursor.execute(sql, ('%' + name + '%',))
        result = cursor.fetchone()
        if result:
            print("emp_id found ") 
            return result[0]  # Return the emp_id if found
        else:
            print("emp_id not found ") 
            return None  # Return None if the name is not found
    except mysql.connector.Error as err:
        print(f"Failed to fetch emp_id: {err}")
    finally:
        cursor.close()
        conn.close()

def insert_into_db(emp_id, date, in_time):
    # Connect to the database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO daily_records (emp_id, Date, `IN-Time`) VALUES (%s, %s, %s)"
        val = (emp_id, date, in_time)
        cursor.execute(sql, val)
        conn.commit()
        print(f"Record inserted for employee ID {emp_id} at {in_time} on {date}")
    except mysql.connector.Error as err:
        print(f"Failed to insert record: {err}")
    finally:
        cursor.close()
        conn.close()


def send_signal_to_pi(person_name):
    message = "Unknown" if person_name == "Unknown" else "Recognized"
    try:
        publish.single(MQTT_PATH, payload=message, hostname=MQTT_SERVER)
    except Exception as e:
        print(f"Failed to send signal to Raspberry Pi: {e}")


def predict_person_from_samples(frames):
    processed_names = set()  # Initialize an empty set to keep track of processed names
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

                # Set timezone to Sri Lanka
                sl_timezone = pytz.timezone('Asia/Colombo')
                now = datetime.now(sl_timezone)
                date = now.strftime('%Y-%m-%d')
                in_time = now.strftime('%H:%M:%S')

                # Check if the person's name has not been processed yet
                if person_name != "Unknown" and person_name not in processed_names:
                    emp_id = get_emp_id_by_name(person_name)
                    if emp_id is not None:
                        print(f"Predicted person: {person_name} (Employee ID: {emp_id}) at {in_time} on {date}")
                        insert_into_db(emp_id, date, in_time)
                        send_signal_to_pi(person_name)  # Send signal for recognized person
                    else:
                        print(f"No matching employee found for {person_name}. Skipping...")
                    processed_names.add(person_name)  # Add the name to the set of processed names
                elif person_name == "Unknown":
                    send_signal_to_pi("Unknown")  # Send signal for unknown person

    return best_prediction[0]



# Main script execution begins here
# Specify the path to your existing video file here
video_path = "video_clip/captured_video.mp4"

# Process the video and predict person
sampled_frames = get_frames_from_video(video_path, 5)
person = predict_person_from_samples(sampled_frames)

# End timing the entire script execution
script_end_time = time.time()

# Calculate and print the total duration
total_duration = script_end_time - script_start_time
print(f"Total script execution took {total_duration:.2f} seconds.")