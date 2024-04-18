import time
import RPi.GPIO as GPIO

class Led:
	def __init__(self, pinNumber, name, duration):
		self.pinNumber = pinNumber
		self.name = name
		self.duration = duration
		GPIO.setup(pinNumber,GPIO.OUT)
	
	def on (self):
		print(self.name + " on")
		GPIO.output(self.pinNumber, GPIO.HIGH)

	def off (self):
		time.sleep(self.duration)
		print(self.name + " off")
		GPIO.output(self.pinNumber, GPIO.LOW)