import random
import platform
import shutil
from indic_transliteration import sanscript
from datetime import timedelta

def get_best_ffmpeg_encoder():
    system = platform.system()
    if system == "Darwin":
        return "h264_videotoolbox"
    
    # Check for NVIDIA NVENC
    if shutil.which("nvidia-smi"):
        return "h264_nvenc"
    
    return "libx264" # Fallback to CPU

AVAILABLE_FONTS = {
    'Arial': 'Arial', 
    'Arial Black': 'Arial-Black', 
    'Impact': 'Impact', 
    'Verdana': 'Verdana', 
    'Courier New': 'Courier', 
    'Georgia': 'Georgia', 
    'Trebuchet MS': 'TrebuchetMS', 
    'Comic Sans MS': 'ComicSansMS', 
    'Helvetica': 'Helvetica', 
    'Times New Roman': 'TimesNewRomanPSMT'
}

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
        return text.strip() 

    # Clean up common ISO marks
    mapping = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o',
        'é': 'e', 'ḍ': 'd', 'ṛ': 'r', 'ṣ': 'sh', 'ṭ': 't',
        'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n', 'ṃ': 'm', 'ḥ': 'h',
        'v': 'w', 'b' : 'b'
    }
    for k, v in mapping.items():
        roman = roman.replace(k, v)
    return roman.lower().strip()

def split_long_segment(segment, max_words=10):
    """Splits a single Whisper segment into multiple smaller segments."""
    words = segment.get('words', [])
    if not words: return [segment]
    
    new_segments = []
    
    # Chunk words into smaller groups
    for i in range(0, len(words), max_words):
        chunk = words[i : i + max_words]
        
        new_seg = {
            'start': chunk[0]['start'],
            'end': chunk[-1]['end'],
            'text': " ".join([w['word'] for w in chunk]),
            'words': chunk
        }
        new_segments.append(new_seg)
        
    return new_segments

def parse_srt_to_segments(srt_path):
    """Parse a standard SRT file into our internal segment list, filtering non-speech tags."""
    import pysubs2
    subs = pysubs2.load(srt_path, encoding="utf-8")
    segments = []
    for s in subs:
        text = s.text.replace("\\N", " ").strip()
        if not text: continue
        
        # Filter out [TAGS] like [BLANK_AUDIO]
        words_raw = text.split()
        words_filtered = [w for w in words_raw if not (w.startswith("[") and w.endswith("]"))]
        
        if not words_filtered:
            continue
            
        filtered_text = " ".join(words_filtered)
        start = s.start / 1000.0
        end = s.end / 1000.0
        
        word_list = []
        if words_filtered and end > start:
            duration = end - start
            per_word = duration / len(words_filtered)
            for i, w in enumerate(words_filtered):
                word_list.append({
                    "word": w,
                    "start": start + i * per_word,
                    "end": start + (i + 1) * per_word
                })
                
        segments.append({
            "start": start,
            "end": end,
            "text": filtered_text,
            "words": word_list
        })
    return segments

def merge_segments(segments, max_words=8):
    """
    Groups many small segments (e.g. 1-word segments from whisper.cpp -ml 1)
    into larger segments for display, while preserving word-level timing.
    """
    if not segments: return []
    
    merged = []
    current_words = []
    
    for seg in segments:
        seg_words = seg.get('words', [])
        if not seg_words:
            # Create a word if it's missing (fallback)
            seg_words = [{
                "word": seg.get('text', ''),
                "start": seg['start'],
                "end": seg['end']
            }]
        
        current_words.extend(seg_words)
        
        if len(current_words) >= max_words:
            merged.append({
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": " ".join([w["word"] for w in current_words]),
                "words": current_words
            })
            current_words = []
            
    if current_words:
        merged.append({
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "text": " ".join([w["word"] for w in current_words]),
            "words": current_words
        })
        
    return merged

def parse_whisper_json(data):
    """Convert whisper.cpp Full JSON (-ojf) into our internal segment list with word-level details."""
    segments = []
    
    def to_secs(v):
        if v is None: return 0.0
        if isinstance(v, (int, float)):
            return float(v) / 1000.0 # whisper.cpp JSON offsets are in ms
        return 0.0

    if "transcription" in data:
        for seg in data["transcription"]:
            text = seg.get("text", "").strip()
            if not text: continue
            
            # Use tokens for word-level accuracy if available
            word_list = []
            tokens = seg.get("tokens", [])
            for tok in tokens:
                tok_text = tok.get("text", "").strip()
                # Skip special tokens [...]
                if not tok_text or (tok_text.startswith("[") and tok_text.endswith("]")):
                    continue
                
                off = tok.get("offsets", {})
                word_list.append({
                    "word": tok_text,
                    "start": to_secs(off.get("from")),
                    "end": to_secs(off.get("to"))
                })
            
            # Fallback to segment-level if no words found
            if not word_list:
                off = seg.get("offsets", {})
                start_sec = to_secs(off.get("from"))
                end_sec = to_secs(off.get("to"))
                words_raw = text.split()
                # Filter out [TAGS]
                words_filtered = [w for w in words_raw if not (w.startswith("[") and w.endswith("]"))]
                
                if words_filtered and end_sec >= start_sec:
                    duration = end_sec - start_sec
                    per_word = duration / len(words_filtered)
                    for i, w in enumerate(words_filtered):
                        word_list.append({
                            "word": w,
                            "start": start_sec + i * per_word,
                            "end": start_sec + (i + 1) * per_word
                        })
                    text = " ".join(words_filtered)
            
            if word_list:
                segments.append({
                    "start": word_list[0]["start"],
                    "end": word_list[-1]["end"],
                    "text": text,
                    "words": word_list
                })
    return segments

def generate_srt_content(
    segments, 
    use_roman=True, 
    use_karaoke=True, 
    highlight_color="#FFFF00", 
    base_color="#FFFFFF", 
    max_words_per_line=8, 
    font_size=26,
    base_font="Arial",
    highlight_font="Arial Black",
    random_base_font=False,
    random_highlight_font=False,
    random_base_color=False,
    random_highlight_color=False
):
    srt_content = ""
    count = 1
    
    processed_segments = []
    for seg in segments:
        processed_segments.extend(split_long_segment(seg, max_words=max_words_per_line))
    
    for seg in processed_segments:
        words = seg.get('words', [])
        if not words: continue
        
        # Pre-process words
        processed_words_text = []
        for w in words:
            text = w['word']
            if use_roman:
                text = to_romanized(text)
            processed_words_text.append(text)
            
        # --- FONT LOGIC ---
        # Resolve to technical name using the mapping
        # base_font here comes as the User-Friendly Key (e.g. "Arial Black")
        # We need to map it to 'Arial-Black'
        
        # Helper to get random tech font
        def get_random_tech_font():
            return random.choice(list(AVAILABLE_FONTS.values()))

        # Helper to resolve single font
        def resolve_font(name):
            return AVAILABLE_FONTS.get(name, 'Arial')


        
        # Pre-calculate fonts for each word to ensure consistency across karaoke frames
        word_base_fonts = []
        word_highlight_fonts = []
        
        for _ in words:
            # Base Font for this word
            if random_base_font:
                word_base_fonts.append(get_random_tech_font())
            else:
                word_base_fonts.append(resolve_font(base_font))
                
            # Highlight Font for this word
            if random_highlight_font:
                word_highlight_fonts.append(get_random_tech_font())
            else:
                word_highlight_fonts.append(resolve_font(highlight_font))

        # Pre-calculate colors for each word
        word_base_colors = []
        word_highlight_colors = []
        
        # Helper for random hex color
        def get_random_color():
            return "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()

        # Helper: Convert #RRGGBB to ASS &HBBGGRR&
        def hex_to_ass_tag(hex_c):
            h = hex_c.lstrip('#')
            if len(h) == 6:
                return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&"
            return hex_c

        for _ in words:
            # Base Color
            if random_base_color:
                word_base_colors.append(hex_to_ass_tag(get_random_color()))
            else:
                word_base_colors.append(hex_to_ass_tag(base_color))
            
            # Highlight Color
            if random_highlight_color:
                word_highlight_colors.append(hex_to_ass_tag(get_random_color()))
            else:
                word_highlight_colors.append(hex_to_ass_tag(highlight_color))


        # Determine highlighting
        if use_karaoke:
            for i, current_word in enumerate(words):
                start = format_timestamp(current_word['start'])
                end = format_timestamp(current_word['end'])
                
                line_parts = []
                for j, word_text in enumerate(processed_words_text):
                    if i == j:
                        # Highlighted Word
                        # Increase size slightly for pop effect (1.2x)
                        current_highlight_font = word_highlight_fonts[j]
                        current_highlight_color = word_highlight_colors[j]
                        # Use ASS \c tag instead of <font>
                        line_parts.append(f'{{\\fn{current_highlight_font}}}{{\\fs{int(font_size * 1.2)}}}{{\\c{current_highlight_color}}}{word_text}')
                    else:
                        # Base Word
                        current_base_font = word_base_fonts[j]
                        current_base_color = word_base_colors[j]
                        line_parts.append(f'{{\\fn{current_base_font}}}{{\\fs{font_size}}}{{\\c{current_base_color}}}{word_text}')
                
                text_line = " ".join(line_parts)
                srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
                count += 1
        else:
            # Standard Mode
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])
            
            line_parts = []
            for j, w_text in enumerate(processed_words_text):
                current_base_font = word_base_fonts[j]
                current_base_color = word_base_colors[j]
                line_parts.append(f'{{\\fn{current_base_font}}}{{\\fs{font_size}}}{{\\c{current_base_color}}}{w_text}')
            
            text_line = " ".join(line_parts)
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1
                
    return srt_content
