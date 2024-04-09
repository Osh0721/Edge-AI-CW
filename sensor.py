import RPi.GPIO as GPIO
import time
import cv2

# Set up GPIO mode and pins
GPIO.setmode(GPIO.BCM)
TRIG_PIN = 18
ECHO_PIN = 16
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Define function to measure distance
def measure_distance():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start

    # Calculate distance in cm
    distance = (pulse_duration * 34000) / 2  
    distance = distance / 100  
    return distance

#Function to activate webcam
def activate_webcam():
    # Initialize webcam
    camera = cv2.VideoCapture(0)
    
    # Capture video for 5 seconds
    for i in range(150):  # 30 frames per second * 5 seconds = 150 frames
        ret, frame = camera.read()
        cv2.imshow('Video', frame)
        cv2.waitKey(33)  # Delay for 33ms (30 fps)
    
    # Release webcam
    camera.release()
    cv2.destroyAllWindows()

try:
    while True:
        # Measure distance
        dist = measure_distance()
        
        # Check if distance is 1 meter or below
        if dist <= 1:
            print("Person detected at a distance of", dist, "meters.")
            print("Activating webcam...")
            activate_webcam()
            print("Webcam activated!")
        
        # Add a delay to avoid continuous measurement
        time.sleep(0.1)

except KeyboardInterrupt:
    # Clean up GPIO
    GPIO.cleanup()
