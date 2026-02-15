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

# ---------------- MODEL SELECTION ----------------
# Options: "tiny", "base", "small", "medium", "large-v3", "turbo"
MODEL_NAME = "large-v3" 
# -------------------------------------------------

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def to_hinglish(text):
    if not text.strip():
        return text
    
    # Transliterate Devanagari to Roman
    roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ISO)
    
    # Clean up common ISO marks for "Hinglish" feel
    mapping = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o',
        'é': 'e', 'ḍ': 'd', 'ṛ': 'r', 'ṣ': 'sh', 'ṭ': 't',
        'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n', 'ṃ': 'm', 'ḥ': 'h'
    }
    for k, v in mapping.items():
        roman = roman.replace(k, v)
    return roman.lower().strip()

def split_segment_into_words(segments, max_words=4):
    """Splits whisper segments into smaller chunks of max_words for punchy subs."""
    new_segments = []
    for seg in segments:
        words = seg.get('words', [])
        if not words:
            # Fallback if word_timestamps=False
            text_words = seg['text'].split()
            duration = seg['end'] - seg['start']
            if not text_words: continue
            
            for i in range(0, len(text_words), max_words):
                chunk = text_words[i:i+max_words]
                new_segments.append({
                    'start': seg['start'] + (i / len(text_words)) * duration,
                    'end': seg['start'] + (min(i + max_words, len(text_words)) / len(text_words)) * duration,
                    'text': " ".join(chunk)
                })
        else:
            for i in range(0, len(words), max_words):
                chunk = words[i:i+max_words]
                new_segments.append({
                    'start': chunk[0]['start'],
                    'end': chunk[-1]['end'],
                    'text': " ".join([w['word'] for w in chunk])
                })
    return new_segments

def write_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = to_hinglish(seg['text'])
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python refine_bot.py <video_file>")
        return

    video_in = sys.argv[1]
    if not os.path.exists(video_in):
        print(f"❌ Video not found: {video_in}")
        return

    final_srt = "final_hinglish.srt"
    video_out = f"{os.path.splitext(video_in)[0]}_subbed.mov"

    try:
        print(f"🧠 1/3 Loading Whisper model ({MODEL_NAME})...")
        # Use MPS (Metal) on Mac if available
        device = "mps" if torch_available_and_mps() else "cpu"
        model = whisper.load_model(MODEL_NAME)

        print(f"🎵 2/3 Transcribing & Translating (showing progress)...")
        # verbose=True to show every transcribed segment in real-time
        result = model.transcribe(video_in, language="hi", word_timestamps=True, verbose=True)
        
        # Split into punchy 4-word segments
        short_segments = split_segment_into_words(result['segments'], max_words=4)
        
        write_srt(short_segments, final_srt)

        print("🔥 3/3 Burning subtitles with Hardware GPU (VideoToolbox)...")
        style = "FontName=Arial Black,FontSize=28,PrimaryColour=&H00FFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=40,Alignment=2"
        # Escape path for FFmpeg
        filter_str = f"subtitles=filename='{final_srt}':force_style='{style}'"
        
        burn_cmd = [
            FFMPEG_PATH, "-y", "-i", video_in, "-vf", filter_str,
            "-c:v", "h264_videotoolbox", "-realtime", "1", "-b:v", "8M", "-c:a", "copy",
            video_out
        ]
        subprocess.run(burn_cmd, check=True)

        print(f"\n✨ ALL DONE! Created: {video_out}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if os.path.exists(final_srt):
            os.remove(final_srt)

def torch_available_and_mps():
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False

if __name__ == "__main__":
    main()
