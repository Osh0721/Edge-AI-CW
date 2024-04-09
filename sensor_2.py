import RPi.GPIO as GPIO
import time

# Set the GPIO mode and HC-SR501 PIR motion sensor pin
GPIO.setmode(GPIO.BCM)
PIR_PIN = 12
GPIO.setup(PIR_PIN, GPIO.IN)

# Initialize variables to keep track of previous and current states
prev_pir_value = GPIO.LOW
pir_value = GPIO.LOW

try:
    while True:
        # Read the HC-SR501 PIR motion sensor value
        pir_value = GPIO.input(PIR_PIN)

        # Check for change event (motion started or stopped)
        if pir_value != prev_pir_value:
            if pir_value == GPIO.HIGH:
                print("Motion started!")
            else:
                print("Motion stopped!")

        # Update the previous value with the current value
        prev_pir_value = pir_value

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")
    GPIO.cleanup()
