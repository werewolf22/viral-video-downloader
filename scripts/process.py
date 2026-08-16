from moviepy.editor import VideoFileClip
import os

INPUT_DIR = "input/videos"
OUTPUT_DIR = "input/processed"

def trim_video(file):
    clip = VideoFileClip(file).subclip(0, 8)  # first 8 sec
    filename = os.path.basename(file)
    output_path = os.path.join(OUTPUT_DIR, filename)
    clip.write_videofile(output_path)
    return output_path


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f"{INPUT_DIR}/{f}" for f in os.listdir(INPUT_DIR)]

    for f in files:
        trim_video(f)