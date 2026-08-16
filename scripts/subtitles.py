import whisper
import os

model = whisper.load_model("base")

INPUT_DIR = "input/processed"
OUTPUT_DIR = "input/subtitles"

def generate_subtitles(video):
    result = model.transcribe(video)

    filename = os.path.basename(video).replace(".mp4", ".srt")
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w") as f:
        for i, seg in enumerate(result['segments']):
            f.write(f"{i+1}\n")
            f.write(f"{seg['start']} --> {seg['end']}\n")
            f.write(f"{seg['text']}\n\n")

    return path


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f"{INPUT_DIR}/{f}" for f in os.listdir(INPUT_DIR)]

    for f in files:
        generate_subtitles(f)