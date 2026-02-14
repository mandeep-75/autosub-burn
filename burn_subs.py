import subprocess
import sys
import os

# Points to the portable ffmpeg you downloaded into this folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = os.path.join(BASE_DIR, "tools/ffmpeg")

def burn_subtitles(video_in, srt_in):
    if not os.path.exists(FFMPEG_PATH):
        print(f"❌ Error: Local ffmpeg not found at {FFMPEG_PATH}")
        print("Download it first: curl -O https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip")
        return

    video_out = f"{os.path.splitext(video_in)[0]}_burned.mov"
    
    print(f"🔥 Burning {srt_in} into {video_in}...")

    # Professional Styling
    # PrimaryColour=&H00FFFF (Yellow), Outline=1 (Black border), FontSize=14
    style = "FontSize=14,PrimaryColour=&H00FFFF,OutlineColour=&H000000,BorderStyle=1,Outline=1,MarginV=20"

    # We use a list for subprocess to handle spaces in filenames safely
    filter_string = f"subtitles=filename='{srt_in}':force_style='{style}'"

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_in,
        "-vf", filter_string,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy", # Keeps original audio quality
        video_out
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Success! Created: {video_out}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed. Check if {srt_in} is a valid SRT file.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python burn_only.py <video_file> <srt_file>")
    else:
        burn_subtitles(sys.argv[1], sys.argv[2])