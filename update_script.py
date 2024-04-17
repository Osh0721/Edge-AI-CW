import subprocess
import sys
import os

# Define the path to your repository's root directory
repo_path = '/home/samanerendra/Edge-AI-CW'

# Ensure you're in the correct directory
os.chdir(repo_path)

# Pull the latest changes from the repository
# subprocess.run(['git', 'pull'], check=True)

# Now call the main functionality script
subprocess.run([sys.executable, 'capture_video_and_predict_cloud.py'])