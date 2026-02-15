import argparse
import sys
import os
import whisper
import subprocess
import gc
try:
    import torch
except ImportError:
    torch = None
from utils import generate_srt_content, AVAILABLE_FONTS

# --- CONFIG ---
FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "ffmpeg")

def main():
    parser = argparse.ArgumentParser(description="AI Subtitle Generator CLI")
    
    # Required
    parser.add_argument("video_path", help="Path to input video file")
    
    # Optional
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3", "turbo"], help="Whisper model size")
    parser.add_argument("--roman", action="store_true", default=True, help="Transliterate to Roman (Hinglish/Punjlish)")
    parser.add_argument("--no-roman", action="store_false", dest="roman", help="Disable transliteration")
    parser.add_argument("--karaoke", action="store_true", default=True, help="Enable Karaoke Highlights")
    parser.add_argument("--no-karaoke", action="store_false", dest="karaoke", help="Disable Karaoke Highlights")
    
    # Style
    parser.add_argument("--highlight-color", default="#FFFF00", help="Hex color for highlight (e.g. #FFFF00)")
    parser.add_argument("--base-color", default="#FFFFFF", help="Hex color for base text (e.g. #FFFFFF)")
    parser.add_argument("--max-words", type=int, default=8, help="Max words per line")
    parser.add_argument("--font-size", type=int, default=26, help="Font size in pixels")
    parser.add_argument("--base-font", default="Arial", choices=list(AVAILABLE_FONTS.keys()), help="Base font name")
    parser.add_argument("--highlight-font", default="Arial Black", choices=list(AVAILABLE_FONTS.keys()), help="Highlight font name")
    
    # Randomization
    parser.add_argument("--random-base", action="store_true", help="Randomize base font per line")
    parser.add_argument("--random-highlight", action="store_true", help="Randomize highlight font per word")

    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video not found at {args.video_path}")
        return

    print(f"🎬 Processing: {args.video_path}")
    print(f"🧠 Loading Model: {args.model}...")
    
    model = whisper.load_model(args.model)
    
    print("🎵 Transcribing...")
    # fp16=False for CPU compatibility if needed, though CLI usually runs on whatever torch supports.
    # explicit fp16=False is safer for M1/M2 CPU fallback if GPU isn't picked up correctly, or just to be safe.
    result = model.transcribe(args.video_path, word_timestamps=True, verbose=True, fp16=False)
    
    # Unload Model immediately to free memory for video processing
    del model
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
    gc.collect()
    
    print("✍️ Generating Subtitles...")
    srt_content = generate_srt_content(
        result['segments'],
        use_roman=args.roman,
        use_karaoke=args.karaoke,
        highlight_color=args.highlight_color,
        base_color=args.base_color,
        max_words_per_line=args.max_words,
        font_size=args.font_size,
        base_font=args.base_font,
        highlight_font=args.highlight_font,
        random_base=args.random_base,
        random_highlight=args.random_highlight
    )
    
    srt_path = f"{os.path.splitext(args.video_path)[0]}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"✅ Saved SRT: {srt_path}")
    
    print("🔥 Burning Video...")
    output_video_path = f"{os.path.splitext(args.video_path)[0]}_subbed.mov"
    
    # Style logic matches App.py
    def hex_to_ass(hex_color):
        hex_color = hex_color.lstrip('#')
        # Check length
        if len(hex_color) != 6: return "&HFFFFFF"
        return f"&H{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}"
    
    ass_base = hex_to_ass(args.base_color)
    tech_base_font = AVAILABLE_FONTS.get(args.base_font, 'Arial')
    
    style = f"FontName={tech_base_font},FontSize={args.font_size},PrimaryColour={ass_base},OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=50,Alignment=2"
    filter_str = f"subtitles=filename='{srt_path}':force_style='{style}'"
    
    cmd = [
        FFMPEG_PATH, "-y", 
        "-i", args.video_path, 
        "-vf", filter_str,
        "-c:v", "h264_videotoolbox", "-realtime", "1", "-b:v", "8M",
        "-c:a", "copy",
        output_video_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✨ ALL DONE! Video saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e}")

if __name__ == "__main__":
    main()
