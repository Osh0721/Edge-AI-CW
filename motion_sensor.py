import cv2 as cv
import os
import numpy as np
import RPi.GPIO as GPIO
import time

# Set up GPIO pins
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
motion_sensor_pin = 4  # Use GPIO pin 4 for the motion sensor input
GPIO.setup(motion_sensor_pin, GPIO.IN)  # Set GPIO pin 4 as an input

def capture_video(video_path, capture_duration=2, resize_factor=0.2):
    cap = cv.VideoCapture(0)
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(3) * resize_factor)
    frame_height = int(cap.get(4) * resize_factor)
    total_frames = int(frame_rate * capture_duration)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)
        out.write(resized_frame)
    
    cap.release()
    out.release()

def main():
    video_folder = "video_clip"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    print("System armed, waiting for motion.")
    
    try:
        while True:
            if GPIO.input(motion_sensor_pin):
                print("Motion detected! Capturing video...")
                video_path = os.path.join(video_folder, "captured_video_" + time.strftime("%Y%m%d-%H%M%S") + ".mp4")
                capture_video(video_path, capture_duration=2, resize_factor=0.2)
                print(f"Video saved to {video_path}")
                
                # Wait a short period to avoid capturing too many videos in quick succession
                time.sleep(10)  # Adjust the sleep time as needed

    except KeyboardInterrupt:
        print("Program terminated by user")

    finally:
        GPIO.cleanup()  # Clean up GPIO to ensure a clean exit

if __name__ == "__main__":
    main()