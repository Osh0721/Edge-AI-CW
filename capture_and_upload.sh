#!/bin/bash

cd ..

cd Edge-AI-CW

git fetch 

git merge 

ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video0 -t 2 /home/admin/videos/captured_video.mp4

mv /home/admin/videos/captured_video.mp4 /home/admin/Edge-AI-CW/video_clip

cd /home/admin/Edge-AI-CW/video_clip

git add captured_video.mp4

git commit -m "Added a new video file to test the automate process without authentication"

git config credential.helper store
git config --global user.name "Osh0721"
git config --global user.email "oshanrathnayaka53@gmail.com"
echo "https://Osh0721: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@github.com" > ~/.git-credentials


git push origin main