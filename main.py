import os

print("Step 1: Fetching videos...")
os.system("python scripts/fetch.py")

print("Step 2: Processing clips...")
os.system("python scripts/process.py")

print("Step 3: Generating subtitles...")
os.system("python scripts/subtitles.py")

print("Step 4: Creating final video...")
os.system("python scripts/editor.py")

print("Done ✅ Check output folder")