import RPi.GPIO as GPIO
import time

# Set GPIO mode to BCM
GPIO.setmode(GPIO.BCM)

# Define GPIO pins
TRIG_PIN = 17
ECHO_PIN = 18

# Set up GPIO pins
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Function to measure distance
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

try:
    while True:
        # Measure distance
        dist = measure_distance()
        
        # Print distance
        print("Distance:", dist, "cm")
        
        # Wait for a short time before next measurement
        time.sleep(1)

except KeyboardInterrupt:
    # Clean up GPIO
    GPIO.cleanup()
