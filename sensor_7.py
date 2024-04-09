import RPi.GPIO as GPIO
import time
import cv2

# Set up GPIO mode and pins
GPIO.setmode(GPIO.BCM)
TRIG_PIN = 17
ECHO_PIN = 18
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Define function to measure distance
def measure_distance():
    # Send a pulse to trigger the sensor
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    
    # Wait for the echo to start
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
    
    # Wait for the echo to end
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
    
    # Calculate the duration of the echo pulse
    pulse_duration = pulse_end - pulse_start
    
    # Calculate distance (speed of sound is approximately 343 m/s)
    distance = pulse_duration * 34300 / 2
    
    return distance

# Define function to activate webcam
def activate_webcam():
    # Initialize webcam
    camera = cv2.VideoCapture(0)
    
    # Capture image
    ret, frame = camera.read()
    
    # Save image
    cv2.imwrite('captured_image.jpg', frame)
    
    # Release webcam
    camera.release()

try:
    while True:
        # Measure distance
        dist = measure_distance()
        
        # Check if distance is less than or equal to 2 meters
        if dist <= 200:  # 2 meters in centimeters
            print("Distance:", dist, "cm")
            print("Activating webcam...")
            activate_webcam()
            print("Webcam activated!")
            
            # Wait for a short time before resuming distance measurement
            time.sleep(2)
        else:
            print("Distance:", dist, "cm")
            print("No need to activate webcam.")
        
except KeyboardInterrupt:
    # Clean up GPIO
    GPIO.cleanup()
