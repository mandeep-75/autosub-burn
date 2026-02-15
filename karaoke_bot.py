import os
import subprocess
import sys
import whisper
import torch
from indic_transliteration import sanscript
from datetime import timedelta

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
FFMPEG_PATH = os.path.join(TOOLS_DIR, "ffmpeg")

# ---------------- CONFIG ----------------
MODEL_NAME = "large-v3" # User preferred large-v3
HIGHLIGHT_COLOR = "#FFFF00" # Yellow hex for karaoke highlight
# ----------------------------------------

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def to_romanized(text):
    if not text.strip(): return text
    
    # Detect Script
    is_hindi = any('\u0900' <= c <= '\u097F' for c in text)
    is_punjabi = any('\u0A00' <= c <= '\u0A7F' for c in text)
    
    if is_hindi:
        roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ISO)
    elif is_punjabi:
        roman = sanscript.transliterate(text, sanscript.GURMUKHI, sanscript.ISO)
    else:
        # Already English or other Roman script
        return text.lower().strip()

    # Clean up common ISO marks for "Hinglish/Punjlish" readability
    mapping = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o',
        'é': 'e', 'ḍ': 'd', 'ṛ': 'r', 'ṣ': 'sh', 'ṭ': 't',
        'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n', 'ṃ': 'm', 'ḥ': 'h',
        'v': 'w', 'b' : 'b' # Optional customizations
    }
    for k, v in mapping.items():
        roman = roman.replace(k, v)
    return roman.lower().strip()

def create_one_word_srt(segments, output_path):
    """Generates SRT where each entry is exactly one word."""
    with open(output_path, "w", encoding="utf-8") as f:
        count = 1
        for seg in segments:
            for word_info in seg.get('words', []):
                start = format_timestamp(word_info['start'])
                end = format_timestamp(word_info['end'])
                text = to_romanized(word_info['word'])
                f.write(f"{count}\n{start} --> {end}\n{text}\n\n")
                count += 1

def create_karaoke_highlight_srt(segments, output_path):
    """Generates SRT where the full sentence is shown, but the current word is colored."""
    with open(output_path, "w", encoding="utf-8") as f:
        count = 1
        for seg in segments:
            words = seg.get('words', [])
            if not words: continue
            
            # Phrase-level transliteration
            phrase_words = [to_romanized(w['word']) for w in words]
            
            for i, current_word in enumerate(words):
                start = format_timestamp(current_word['start'])
                end = format_timestamp(current_word['end'])
                
                # Build the sentence with the current word highlighted
                highlighted_phrase = []
                for j, word_text in enumerate(phrase_words):
                    if i == j:
                        highlighted_phrase.append(f"<font color=\"{HIGHLIGHT_COLOR}\">{word_text}</font>")
                    else:
                        highlighted_phrase.append(word_text)
                
                text = " ".join(highlighted_phrase)
                f.write(f"{count}\n{start} --> {end}\n{text}\n\n")
                count += 1

def main():
    if len(sys.argv) < 2:
        print("Usage: python karaoke_bot.py <video_file>")
        return

    video_in = sys.argv[1]
    if not os.path.exists(video_in):
        print(f"❌ Video not found: {video_in}")
        return

    one_word_srt = "fast_one_word.srt"
    karaoke_srt = "karaoke_highlight.srt"
    video_out = f"{os.path.splitext(video_in)[0]}_karaoke.mov"

    try:
        print(f"🧠 1/4 Loading Whisper model ({MODEL_NAME})...")
        model = whisper.load_model(MODEL_NAME)

        print(f"🎵 2/4 Transcribing (Auto-Detecting Language: Hi/En/Pa)...")
        # Removing language="hi" for auto-detection
        result = model.transcribe(video_in, word_timestamps=True, verbose=True)

        print(f"✍️ 3/4 Creating separate SRT files...")
        # 1. Fast one-word-only version
        create_one_word_srt(result['segments'], one_word_srt)
        # 2. Context version with one-word highlight (karaoke)
        create_karaoke_highlight_srt(result['segments'], karaoke_srt)

        print("🔥 4/4 Burning Karaoke subtitles (Hardware GPU)...")
        # Style optimized for karaoke (larger, centered)
        style = "FontName=Arial Black,FontSize=26,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=50,Alignment=2"
        filter_str = f"subtitles=filename='{karaoke_srt}':force_style='{style}'"
        
        burn_cmd = [
            FFMPEG_PATH, "-y", "-i", video_in, "-vf", filter_str,
            "-c:v", "h264_videotoolbox", "-realtime", "1", "-b:v", "8M", "-c:a", "copy",
            video_out
        ]
        subprocess.run(burn_cmd, check=True)

        print(f"\n✨ DONE!")
        print(f"Created Rapid SRT: {one_word_srt}")
        print(f"Created Karaoke SRT: {karaoke_srt}")
        print(f"Final Video: {video_out}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
