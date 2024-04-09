import RPi.GPIO as GPIO
import time
 
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
PIR_PIN = 23
GPIO.setup(PIR_PIN, GPIO.IN)

print('Starting up the PIR Module ')
time.sleep(1)
print ('Ready')

while True:
  if GPIO.input(PIR_PIN):
    print('Motion Detected')
  time.sleep(1)


