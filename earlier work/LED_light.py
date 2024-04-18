# Import time, GPIO, and new Led class
import time
import RPi.GPIO as GPIO
from led import Led

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Create variables to hold the 3 LEDs
redled = Led(18, "RED", 5)
yellowled = Led(8, "YELLOW", 2) 
greenled = Led(20, "GREEN", 5)
# print("")

#start traffic sequence
redled.on()
redled.off()

greenled.on()
greenled.off()

yellowled.on()
yellowled.off()

redled.on()
redled.off()

GPIO.cleanup()