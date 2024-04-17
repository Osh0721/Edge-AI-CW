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
from google.cloud import pubsub_v1
import json

# Start timing the entire script execution
script_start_time = time.time()

# Configuration for Pub/Sub
project_id = "intelligate-security-system"
topic_id = "intelligate_pub_sub"
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

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

def insert_into_db(emp_id, date, current_time):
    # Connect to the database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        # Check if there's already a record for today for this employee
        check_sql = "SELECT `IN-Time`, `OUT-Time` FROM daily_records WHERE emp_id = %s AND Date = %s"
        cursor.execute(check_sql, (emp_id, date))
        result = cursor.fetchone()

        if result:
            # Record exists, update the OUT-Time
            if result[1] is None or current_time > result[1]:  # Only update if the new OUT-Time is later
                update_sql = "UPDATE daily_records SET `OUT-Time` = %s WHERE emp_id = %s AND Date = %s"
                cursor.execute(update_sql, (current_time, emp_id, date))
                conn.commit()
                print(f"OUT-Time updated for employee ID {emp_id} to {current_time} on {date}")
            else:
                print(f"Existing OUT-Time {result[1]} is later than the current time {current_time}. No update needed.")
        else:
            # No record exists, insert new IN-Time
            insert_sql = "INSERT INTO daily_records (emp_id, Date, `IN-Time`) VALUES (%s, %s, %s)"
            cursor.execute(insert_sql, (emp_id, date, current_time))
            conn.commit()
            print(f"IN-Time recorded for employee ID {emp_id} at {current_time} on {date}")
    except mysql.connector.Error as err:
        print(f"Failed to update record: {err}")
    finally:
        cursor.close()
        conn.close()



def predict_person_from_samples(frames):
    processed_names = set()
    best_prediction = ("Unknown", 0.5)  # Initializes with "Unknown" as the default best prediction
    for face in frames:
        if face is not None:
            embedding = get_embedding(face)
            embedding = np.expand_dims(embedding, axis=0)
            prediction = model.predict(embedding)
            confidence = model.predict_proba(embedding).max()
            
            person_name = encoder.inverse_transform(prediction)[0] if confidence > best_prediction[1] else "Unknown"
            best_prediction = (person_name, confidence)

            sl_timezone = pytz.timezone('Asia/Colombo')
            now = datetime.now(sl_timezone)
            date = now.strftime('%Y-%m-%d')
            in_time = now.strftime('%H:%M:%S')

            if person_name != "Unknown":
                if person_name not in processed_names:
                    emp_id = get_emp_id_by_name(person_name)
                    if emp_id is not None:
                        print(f"Predicted person: {person_name} (Employee ID: {emp_id}) at {in_time} on {date}")
                        insert_into_db(emp_id, date, in_time)
                    else:
                        print(f"No matching employee found for {person_name}. Skipping...")
                    processed_names.add(person_name)
            else:
                print("Predicted person: Unknown")

            # Always send prediction to Raspberry Pi whether known or unknown
            send_prediction_to_pi(person_name)

    return best_prediction[0]
def send_prediction_to_pi(person_name):
    """Publishes the recognized person's name to the Pub/Sub topic."""
    data = {'result': person_name}
    data = json.dumps(data).encode("utf-8")
    try:
        publish_future = publisher.publish(topic_path, data)
        publish_future.result()  # Wait for publish to complete.
        print("Prediction sent to Raspberry Pi via Pub/Sub")
    except Exception as e:
        print("Error sending prediction to Raspberry Pi via Pub/Sub:", e)


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