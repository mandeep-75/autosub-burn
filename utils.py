import random
import platform
import shutil
import pysubs2
from indic_transliteration import sanscript
from datetime import timedelta

# ----------------------------------------------------------------------
# Font mapping (display name -> technical name for ASS)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# FFmpeg encoder selection
# ----------------------------------------------------------------------
def get_best_ffmpeg_encoder():
    """Return the best hardware accelerated H.264 encoder for the current system."""
    system = platform.system()
    if system == "Darwin":
        return "h264_videotoolbox"
    if shutil.which("nvidia-smi"):
        return "h264_nvenc"
    return "libx264"  # software fallback

# ----------------------------------------------------------------------
# whisper.cpp GPU flags
# ----------------------------------------------------------------------
def get_whisper_gpu_flags(device_choice):
    """
    Return whisper.cpp command line flags for GPU based on user choice.
    device_choice: 'auto', 'CPU', or 'CUDA'
    """
    if device_choice == "CPU":
        return ["-ng", "0"]                     # force CPU
    if device_choice == "CUDA":
        if shutil.which("nvidia-smi"):           # simple check for NVIDIA GPU
            return ["-ng", "1"]                  # enable GPU
        # fallback to CPU if CUDA requested but not available
        return ["-ng", "0"]
    # auto: enable GPU if possible (NVIDIA or macOS)
    if shutil.which("nvidia-smi") or platform.system() == "Darwin":
        return ["-ng", "1"]
    return ["-ng", "0"]

# ----------------------------------------------------------------------
# Timestamp formatting for SRT
# ----------------------------------------------------------------------
def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# ----------------------------------------------------------------------
# Romanization (Hindi / Punjabi)
# ----------------------------------------------------------------------
def to_romanized(text):
    """Convert Devanagari or Gurmukhi script to simple romanized form."""
    if not text.strip():
        return text

    is_hindi = any('\u0900' <= c <= '\u097F' for c in text)
    is_punjabi = any('\u0A00' <= c <= '\u0A7F' for c in text)

    if is_hindi:
        roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ISO)
    elif is_punjabi:
        roman = sanscript.transliterate(text, sanscript.GURMUKHI, sanscript.ISO)
    else:
        return text.strip()

    # Clean up common diacritics to simple ascii
    mapping = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o',
        'é': 'e', 'ḍ': 'd', 'ṛ': 'r', 'ṣ': 'sh', 'ṭ': 't',
        'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n', 'ṃ': 'm', 'ḥ': 'h',
        'v': 'w', 'b': 'b'
    }
    for k, v in mapping.items():
        roman = roman.replace(k, v)
    return roman.lower().strip()

# ----------------------------------------------------------------------
# Segment splitting (long lines)
# ----------------------------------------------------------------------
def split_long_segment(segment, max_words=10):
    """Split a segment into smaller chunks based on max_words."""
    words = segment.get('words', [])
    if not words:
        return [segment]

    new_segments = []
    for i in range(0, len(words), max_words):
        chunk = words[i:i + max_words]
        new_segments.append({
            'start': chunk[0]['start'],
            'end': chunk[-1]['end'],
            'text': " ".join(w['word'] for w in chunk),
            'words': chunk
        })
    return new_segments

# ----------------------------------------------------------------------
# Parsing SRT files (fallback)
# ----------------------------------------------------------------------
def parse_srt_to_segments(srt_path):
    """Parse a standard SRT file into our internal segment list, filtering non‑speech tags."""
    subs = pysubs2.load(srt_path, encoding="utf-8")
    segments = []
    for s in subs:
        text = s.text.replace("\\N", " ").strip()
        if not text:
            continue

        # Remove bracketed tags like [BLANK_AUDIO]
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

# ----------------------------------------------------------------------
# Parsing whisper.cpp JSON output
# ----------------------------------------------------------------------
def parse_whisper_json(json_path):
    """Extract word‑level segments from whisper.cpp full JSON (-ojf)."""
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    language = data.get('language', 'unknown')
    segments = []

    def to_secs(v):
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return v / 1000.0  # whisper.cpp JSON offsets are in ms
        return 0.0

    if "transcription" in data:
        for seg in data["transcription"]:
            text = seg.get("text", "").strip()
            if not text:
                continue

            word_list = []
            tokens = seg.get("tokens", [])
            for tok in tokens:
                tok_text = tok.get("text", "").strip()
                if not tok_text or (tok_text.startswith("[") and tok_text.endswith("]")):
                    continue
                off = tok.get("offsets", {})
                word_list.append({
                    "word": tok_text,
                    "start": to_secs(off.get("from")),
                    "end": to_secs(off.get("to"))
                })

            # Fallback if no tokens (older format)
            if not word_list:
                off = seg.get("offsets", {})
                start_sec = to_secs(off.get("from"))
                end_sec = to_secs(off.get("to"))
                words_raw = text.split()
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

    return segments, language

# ----------------------------------------------------------------------
# Merging word‑level segments into display lines
# ----------------------------------------------------------------------
def merge_segments(segments, max_words=8):
    """
    Group many small (e.g. 1‑word) segments into larger display segments,
    preserving word‑level timing.
    """
    if not segments:
        return []

    merged = []
    current_words = []

    for seg in segments:
        seg_words = seg.get('words', [])
        if not seg_words:
            # fallback: create a single word from the whole segment
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
                "text": " ".join(w["word"] for w in current_words),
                "words": current_words[:]   # copy
            })
            current_words = []

    if current_words:
        merged.append({
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "text": " ".join(w["word"] for w in current_words),
            "words": current_words
        })

    return merged

# ----------------------------------------------------------------------
# Generate SRT content with full styling support
# ----------------------------------------------------------------------
def generate_srt_content(
    segments,
    use_roman=True,
    use_karaoke=False,
    show_original_and_roman=False,
    max_words_per_line=8,
    font_size=26,
    base_font="Arial",
    highlight_font="Arial Black",
    random_base_font=False,
    random_highlight_font=False,
    random_base_color=False,
    random_highlight_color=False,
    base_bold=False,
    base_italic=False,
    highlight_bold=False,
    highlight_italic=False,
    include_styling=True,
    highlight_color="#FFFF00",
    base_color="#FFFFFF"
):
    """
    Generate SRT content from segment list.

    If include_styling is False, output is plain text (no ASS tags).
    Otherwise, ASS override tags are inserted for font, size, color, bold, italic.

    For karaoke (use_karaoke=True), each word becomes its own subtitle event
    with the current word highlighted.

    For bilingual (show_original_and_roman=True), each line shows both original
    and romanized text (separated by a newline).
    """
    srt_content = ""
    count = 1

    # First split any long segments according to max_words_per_line
    processed_segments = []
    for seg in segments:
        processed_segments.extend(split_long_segment(seg, max_words_per_line))

    # Helper to resolve font name to technical ASS name
    def resolve_font(name):
        return AVAILABLE_FONTS.get(name, 'Arial')

    def get_random_tech_font():
        return random.choice(list(AVAILABLE_FONTS.values()))

    # Helper to convert hex to ASS color (&HBBGGRR)
    def hex_to_ass_tag(hex_c):
        h = hex_c.lstrip('#')
        if len(h) == 6:
            return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&"
        return hex_c

    def get_random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()

    # Process each segment
    for seg in processed_segments:
        words = seg.get('words', [])
        if not words:
            continue

        # Prepare romanized texts if needed
        word_texts = []
        for w in words:
            txt = w['word']
            if use_roman:
                txt = to_romanized(txt)
            word_texts.append(txt)

        # If bilingual, we also need original texts
        if show_original_and_roman:
            original_texts = [w['word'] for w in words]

        # Pre‑compute fonts and colors for each word (for consistency across karaoke frames)
        word_base_fonts = []
        word_highlight_fonts = []
        word_base_colors = []
        word_highlight_colors = []

        for _ in words:
            # Base font
            if random_base_font:
                word_base_fonts.append(get_random_tech_font())
            else:
                word_base_fonts.append(resolve_font(base_font))

            # Highlight font
            if random_highlight_font:
                word_highlight_fonts.append(get_random_tech_font())
            else:
                word_highlight_fonts.append(resolve_font(highlight_font))

            # Base color
            if random_base_color:
                word_base_colors.append(hex_to_ass_tag(get_random_color()))
            else:
                word_base_colors.append(hex_to_ass_tag(base_color))

            # Highlight color
            if random_highlight_color:
                word_highlight_colors.append(hex_to_ass_tag(get_random_color()))
            else:
                word_highlight_colors.append(hex_to_ass_tag(highlight_color))

        # Helper to build styling tags for a word
        def style_tags(word_idx, is_highlight):
            if not include_styling:
                return ""

            if is_highlight:
                font = word_highlight_fonts[word_idx]
                color = word_highlight_colors[word_idx]
                bold = highlight_bold
                italic = highlight_italic
                size = int(font_size * 1.2) if use_karaoke else font_size
            else:
                font = word_base_fonts[word_idx]
                color = word_base_colors[word_idx]
                bold = base_bold
                italic = base_italic
                size = font_size

            tags = []
            if font:
                tags.append(f"\\fn{font}")
            if color:
                tags.append(f"\\c{color}")
            if size:
                tags.append(f"\\fs{size}")
            if bold:
                tags.append("\\b1")
            if italic:
                tags.append("\\i1")
            # Reset bold/italic after word? We'll rely on resetting at each word.
            # In ASS, style changes are applied from that point onward until changed.
            # So we need to include reset codes after each word if we want to avoid bleed.
            # To keep it simple, we assume each word is separate or we reset at the end of line.
            # We'll add closing tags later if needed. For now, we just open style.
            # The line will be built as "word1 word2 ..." and the last style will persist,
            # but in karaoke each word is a separate event so it's fine.
            # In non‑karaoke, all words share the same base style, so no issue.
            return "{" + "".join(tags) + "}"

        # Generate subtitle events
        if use_karaoke:
            # One event per word, with current word highlighted
            for i, current_word in enumerate(words):
                start = format_timestamp(current_word['start'])
                end = format_timestamp(current_word['end'])

                # Build the line: for each word, apply appropriate style
                line_parts = []
                for j, wtext in enumerate(word_texts):
                    if j == i:
                        # Highlighted word
                        tags = style_tags(j, is_highlight=True)
                        line_parts.append(f"{tags}{wtext}")
                    else:
                        # Base word
                        tags = style_tags(j, is_highlight=False)
                        line_parts.append(f"{tags}{wtext}")

                text_line = " ".join(line_parts)
                srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
                count += 1

        elif show_original_and_roman:
            # Bilingual mode: each line shows original above romanized
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])

            # Build original line with base style
            orig_parts = []
            for j, otext in enumerate(original_texts):
                tags = style_tags(j, is_highlight=False)
                orig_parts.append(f"{tags}{otext}")
            orig_line = " ".join(orig_parts)

            # Build romanized line with base style (maybe different font)
            roman_parts = []
            for j, rtext in enumerate(word_texts):
                tags = style_tags(j, is_highlight=False)
                roman_parts.append(f"{tags}{rtext}")
            roman_line = " ".join(roman_parts)

            # Combine with newline
            text_line = f"{orig_line}\\N{roman_line}"
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1

        else:
            # Standard mode: one subtitle per segment
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])

            line_parts = []
            for j, wtext in enumerate(word_texts):
                tags = style_tags(j, is_highlight=False)
                line_parts.append(f"{tags}{wtext}")

            text_line = " ".join(line_parts)
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1

    return srt_content

# ----------------------------------------------------------------------
# Transform SRT case (uppercase/lowercase) preserving structure
# ----------------------------------------------------------------------
def transform_srt_case(srt_content, case_func):
    """
    Apply case_func (str.upper or str.lower) to the text part of each subtitle block.
    """
    blocks = srt_content.strip().split("\n\n")
    transformed_blocks = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            idx = lines[0]
            times = lines[1]
            text = "\n".join(lines[2:])
            transformed_text = case_func(text)
            transformed_blocks.append(f"{idx}\n{times}\n{transformed_text}")
        else:
            # keep malformed blocks unchanged
            transformed_blocks.append(block)
    return "\n\n".join(transformed_blocks)