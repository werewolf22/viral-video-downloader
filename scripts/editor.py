from moviepy.editor import *
import os
import random

VIDEO_DIR = "input/processed"
OUTPUT = "output/final.mp4"

def create_compilation():
    clips = []

    files = os.listdir(VIDEO_DIR)
    random.shuffle(files)

    for f in files[:5]:
        clip = VideoFileClip(f"{VIDEO_DIR}/{f}")

        # Add text overlay
        txt = TextClip("FUNNY MOMENT 😂", fontsize=60, color='white')
        txt = txt.set_position(("center", "top")).set_duration(clip.duration)

        video = CompositeVideoClip([clip, txt])
        clips.append(video)

    final = concatenate_videoclips(clips)
    final.write_videofile(OUTPUT)


if __name__ == "__main__":
    create_compilation()