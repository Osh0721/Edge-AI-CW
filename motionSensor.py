import cv2
import time
from gpiozero import MotionSensor

# Initialize the motion sensor (connect it to GPIO pin 4)
pir = MotionSensor(4)

# Initialize the webcam
camera = cv2.VideoCapture(0)

# Main loop to detect motion and activate the camera
while True:
    if pir.motion_detected:
        print("Motion detected!")
       
        # Capture frame from the webcam
        ret, frame = camera.read()

        # Generate a unique file name based on current timestamp
        file_name = '/home/admin/sensor' + time.strftime("%Y%m%d-%H%M%S") + '.png'

        # Save the captured frame as an image file
        cv2.imwrite(file_name, frame)

       
    # Add a short delay to avoid continuously checking for motion
    time.sleep(0.5)

# Release the webcam
camera.release()