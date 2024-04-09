import RPi.GPIO as GPIO
import time

# Set up GPIO mode and pins
GPIO.setmode(GPIO.BCM)
TRIG_PIN = 18
ECHO_PIN = 16
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

while True:
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.0001)
    GPIO.output(TRIG_PIN, False)
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = (pulse_duration*34000)/2
    dist = round(distance, 2)
    print("Distance:", dist)
    time.sleep(1)