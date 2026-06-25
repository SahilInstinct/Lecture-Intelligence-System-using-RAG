# Covert the videos to mp3 using ffmpeg
import os
import subprocess

files = os.listdir('videos')
for file in files:
    tutorial_no = file.split('#')[1].split(' ')[0]
    file_name = file.split('_')[0].split('.com')[1].strip()
    subprocess.run(['ffmpeg','-i', f"videos/{file}", f"audios/{tutorial_no}_{file_name}.mp3"])