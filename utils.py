import random
import platform
import shutil
from indic_transliteration import sanscript
from datetime import timedelta

# Font mapping: display name -> technical name (used in ASS)
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
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def to_romanized(text):
    """Convert Devanagari/Gurmukhi text to a simple romanized form.
       Safely handles non‑string input by converting to string first."""
    # Defensive: ensure we have a string
    if not isinstance(text, str):
        text = str(text)
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
    # Simplify diacritics
    mapping = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ē': 'e', 'ō': 'o',
        'é': 'e', 'ḍ': 'd', 'ṛ': 'r', 'ṣ': 'sh', 'ṭ': 't',
        'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n', 'ṃ': 'm', 'ḥ': 'h',
        'v': 'w', 'b': 'b'
    }
    for k, v in mapping.items():
        roman = roman.replace(k, v)
    return roman.lower().strip()

def split_long_segment(segment, max_words=10):
    """Split a segment into smaller chunks if it has too many words."""
    words = segment.get('words', [])
    if not words:
        return [segment]
    new_segments = []
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

def merge_segments(segments, max_words=8):
    """Merge word‑level segments into lines of at most max_words words."""
    if not segments:
        return []
    merged = []
    current_words = []
    for seg in segments:
        seg_words = seg.get('words', [])
        if not seg_words:
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

def hex_to_ass(hex_color):
    """
    Convert #RRGGBB to ASS color format &HBBGGRR& (opaque).
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return "&H000000&"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{b}{g}{r}&"

def random_ass_color():
    """Generate a random opaque ASS colour."""
    return f"&H{random.randint(0, 0xFFFFFF):06x}&"

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
    random_highlight_color=False,
    base_bold=False,
    base_italic=False,
    highlight_bold=False,
    highlight_italic=False,
    include_styling=True,
    show_original_and_roman=False,
    text_transform=None
):
    """
    Generate SRT content with optional ASS‑style inline formatting.

    New behaviour for randomness:
      - Base (non‑highlighted) text: if random_base_font/color is True,
        the *entire line* gets one random font/color.
      - Highlighted words (when use_karaoke=True): if random_highlight_font/color is True,
        *each* highlighted word gets its own random font/color.
    """
    srt_content = ""
    count = 1

    # Split long segments if needed
    processed_segments = []
    for seg in segments:
        processed_segments.extend(split_long_segment(seg, max_words=max_words_per_line))

    # Pre‑resolve the base font technical name (if not random)
    base_font_tech = AVAILABLE_FONTS.get(base_font, 'Arial')
    highlight_font_tech = AVAILABLE_FONTS.get(highlight_font, 'Arial')

    for seg in processed_segments:
        words = seg.get('words', [])
        if not words:
            continue

        # --- Prepare word texts (original and romanized) ---
        # Defensive: ensure each word text is a string
        original_words = [str(w.get('word', '')) for w in words]
        romanized_words = [to_romanized(w) for w in original_words]

        # Choose which word list to use for display
        if show_original_and_roman:
            # We'll build two lines later, so keep both
            pass
        else:
            if use_roman:
                display_words = romanized_words
            else:
                display_words = original_words

        # Apply text transform (uppercase/lowercase)
        if text_transform == 'upper':
            transform = str.upper
        elif text_transform == 'lower':
            transform = str.lower
        else:
            transform = lambda x: x

        if not show_original_and_roman:
            display_words = [transform(w) for w in display_words]

        # --- Styling preparation (only if include_styling) ---
        if include_styling:
            # Line‑level base font and colour (used for non‑highlighted words)
            if random_base_font:
                line_base_font = random.choice(list(AVAILABLE_FONTS.values()))
            else:
                line_base_font = base_font_tech

            if random_base_color:
                line_base_color = random_ass_color()
            else:
                line_base_color = hex_to_ass(base_color)

        # --- Generate subtitle entries ---
        if show_original_and_roman:
            # Single subtitle with two lines: original (top) and romanized (bottom)
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])
            orig_line = " ".join([transform(w) for w in original_words])
            roman_line = " ".join([transform(w) for w in romanized_words])
            full_text = orig_line + "\\N" + roman_line

            if include_styling:
                # Apply line‑level base style to the whole subtitle
                tags = []
                tags.append(f"\\fn{line_base_font}")
                tags.append(f"\\fs{font_size}")
                tags.append(f"\\c{line_base_color}")
                tags.append("\\b1" if base_bold else "\\b0")
                tags.append("\\i1" if base_italic else "\\i0")
                tag_str = "".join(tags)
                line = f"{{{tag_str}}}{full_text}"
            else:
                line = full_text

            srt_content += f"{count}\n{start} --> {end}\n{line}\n\n"
            count += 1

        elif use_karaoke and include_styling:
            # Karaoke mode: one subtitle per word, with that word highlighted
            for i, current_word in enumerate(words):
                start = format_timestamp(current_word['start'])
                end = format_timestamp(current_word['end'])

                # Build the line with all words, but only the i‑th word highlighted
                line_parts = []
                for j, word_text in enumerate(display_words):
                    if i == j:
                        # Highlighted word: per‑word randomness if enabled
                        if random_highlight_font:
                            word_font = random.choice(list(AVAILABLE_FONTS.values()))
                        else:
                            word_font = highlight_font_tech

                        if random_highlight_color:
                            word_color = random_ass_color()
                        else:
                            word_color = hex_to_ass(highlight_color)

                        size = int(font_size * 1.2)  # larger for highlight
                        bold = highlight_bold
                        italic = highlight_italic
                    else:
                        # Base word: use line‑level base font/color
                        word_font = line_base_font
                        word_color = line_base_color
                        size = font_size
                        bold = base_bold
                        italic = base_italic

                    tags = []
                    tags.append(f"\\fn{word_font}")
                    tags.append(f"\\fs{size}")
                    tags.append(f"\\c{word_color}")
                    tags.append("\\b1" if bold else "\\b0")
                    tags.append("\\i1" if italic else "\\i0")
                    tag_str = "".join(tags)
                    line_parts.append(f"{{{tag_str}}}{word_text}")

                text_line = " ".join(line_parts)
                srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
                count += 1

        elif not include_styling:
            # Plain text, no tags
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])
            text_line = " ".join(display_words)
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1

        else:
            # Standard (non‑karaoke) mode with styling – one subtitle per segment
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])

            line_parts = []
            for j, w_text in enumerate(display_words):
                # All words use line‑level base style
                tags = []
                tags.append(f"\\fn{line_base_font}")
                tags.append(f"\\fs{font_size}")
                tags.append(f"\\c{line_base_color}")
                tags.append("\\b1" if base_bold else "\\b0")
                tags.append("\\i1" if base_italic else "\\i0")
                tag_str = "".join(tags)
                line_parts.append(f"{{{tag_str}}}{w_text}")

            text_line = " ".join(line_parts)
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1

    return srt_content