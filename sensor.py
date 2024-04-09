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
    distance = (pulse_duration * 34000) / 2
    return distance / 100  

# Define function to activate webcam
def activate_webcam():
    print("Person Detected. Activating webcam!!!")
    # Initialize webcam
    camera = cv2.VideoCapture(0)
    start_time = time.time()
    
    # Capture video for 5 seconds
    while time.time() - start_time <= 5:
        ret, frame = camera.read()
        cv2.imshow('Webcam', frame)
        cv2.waitKey(1)
    
    # Release webcam
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam deactivated.")

try:
    while True:
        # Measure distance in meters
        dist = measure_distance()
        
        # Check if distance is less than or equal to 1 meter
        if dist <= 1:
            print("Distance:", dist, "m")
            activate_webcam()  # Activate webcam
        else:
            print("Distance:", dist, "m")
            print("Please move bit towrds the camera")
            time.sleep(4)  # Wait for 1 second before checking again
        
except KeyboardInterrupt:
    # Clean up GPIO
    GPIO.cleanup()
