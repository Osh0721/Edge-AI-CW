import threading
from google.cloud import pubsub_v1
import RPi.GPIO as GPIO
import time
import json
import subprocess
import os

# Set your Google Cloud credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/admin/MQTT/intelligate-security-system-e5236dd4ecca.json"

# GPIO setup
GREEN_LED_PIN = 25
RED_LED_PIN = 7
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.output(RED_LED_PIN, GPIO.HIGH)  # Red LED ON by default

# Set up GPIO mode and pins for ultrasonic sensor
GPIO.setmode(GPIO.BOARD)
TRIG_PIN = 18
ECHO_PIN = 16
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def handle_prediction(prediction_result):
    """Handles the logic to control the LED based on the prediction result."""
    employee = ["Oshan", "Nadun", "Roshen", "Maxi"]
    if prediction_result in employee:
        print("Authorized")
        GPIO.output(RED_LED_PIN, GPIO.LOW)  # Turn off red LED
        GPIO.output(GREEN_LED_PIN, GPIO.HIGH)  # Turn on green LED
        time.sleep(3)  # Keep green LED on for 1 second
        GPIO.output(GREEN_LED_PIN, GPIO.LOW)
        time.sleep(1)   # Turn off green LED
        GPIO.output(RED_LED_PIN, GPIO.HIGH)  # Turn on red LED after a delay
    else:
        print("Unauthorized")
        GPIO.output(RED_LED_PIN, GPIO.HIGH)  # Ensure red LED is on if unauthorized

def callback(message):
    print(f"Received Pub/Sub message: {message.data.decode('utf-8')}")
    try:
        data = json.loads(message.data)
        print(data)
        prediction_result = data['result']
        handle_prediction(prediction_result)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Missing key in data: {e}")
    message.ack()

def measure_distance():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = (pulse_duration * 34000) / 2
    return distance / 100  

def capture_and_push_video():
    try:
        while True:
            # Measure distance in meters
            dist = measure_distance()
            # Check if distance is less than or equal to 1 meter
            if 0.1 < dist <= 1:
                print("Distance:", dist, "m")
                # Execute the shell commands to capture video and push to GitHub
                subprocess.run([
                    "ffmpeg", "-f", "v4l2", "-video_size", "1280x720", "-i", "/dev/video0", "-t", "2",
                    "/home/admin/videos/captured_video.mp4"
                ])
                subprocess.run(["mv", "/home/admin/videos/captured_video.mp4", "/home/admin/Edge-AI-CW/video_clip"])
                os.chdir("/home/admin/Edge-AI-CW/video_clip")
                subprocess.run(["git", "add", "captured_video.mp4"])
                subprocess.run(["git", "commit", "-m", "Added a new video file to test the automate process without authentication"])
                subprocess.run(["git", "config", "credential.helper", "store"])
                subprocess.run(["git", "config", "--global", "user.name", "Osh0721"])
                subprocess.run(["git", "config", "--global", "user.email", "oshanrathnayaka53@gmail.com"])
                subprocess.run(["echo", "https://Osh0721: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@github.com", ">", "~/.git-credentials"])
                subprocess.run(["git", "push", "origin", "main"])
                print("Video captured and pushed to GitHub")
                time.sleep(30)
            else:
                print("Distance:", dist, "m")
                print("Please move towards the camera")
                time.sleep(4)  # Wait for 4 seconds before checking again
    except KeyboardInterrupt:
        pass

# Create and start threads for Pub/Sub subscription and ultrasonic sensor measurement
pubsub_thread = threading.Thread(target=lambda: streaming_pull_future.result())
ultrasonic_thread = threading.Thread(target=capture_and_push_video)

pubsub_thread.start()
ultrasonic_thread.start()

# Join threads to wait for their completion
pubsub_thread.join()
ultrasonic_thread.join()

# Clean up GPIO
GPIO.cleanup()
