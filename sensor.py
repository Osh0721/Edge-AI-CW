import RPi.GPIO as GPIO
import time
import subprocess
import os

# Set up GPIO mode and pins
GPIO.setmode(GPIO.BOARD)
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

try:
    while True:
        # Measure distance in centimeters
        dist = measure_distance()

        # Check if distance is less than or equal to 1 centimeter
        if 0.1 < dist <= 1:
            print("Distance:", dist, "cm")

            # Execute the shell commands to capture video and push to GitHub
            subprocess.run([
                "ffmpeg", "-f", "v4l2", "-video_size", "1280x720", "-i", "/dev/video0", "-t", "2",
                "/home/admin/videos/captured_video.mp4"
            ])

            subprocess.run(["mv", "/home/admin/videos/captured_video.mp4", "/home/admin/Edge-AI-CW/video_clip"])
            os.chdir("/home/admin/Edge-AI-CW/video_clip")
            subprocess.run(["git", "add", "captured_video.mp4"])
            subprocess.run(["git", "commit", "-m", "Added a new video file to test the automate process without authentication"])
            subprocess.run(["git", "config", "credential.helper", "store"])
            subprocess.run(["git", "config", "--global", "user.name", "Osh0721"])
            subprocess.run(["git", "config", "--global", "user.email", "oshanrathnayaka53@gmail.com"])
            subprocess.run(["echo", "https://Osh0721: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@github.com", ">", "~/.git-credentials"])
            subprocess.run(["git", "push", "origin", "main"])

            print("Video captured and pushed to GitHub")
            time.sleep(30)

        else:
            print("Distance:", dist, "m")
            print("Please move bit towrds the camera")
            time.sleep(4)  

except KeyboardInterrupt:
    # Clean up GPIO
    GPIO.cleanup()