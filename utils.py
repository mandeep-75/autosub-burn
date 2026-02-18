import random
import platform
import shutil
from indic_transliteration import sanscript
from datetime import timedelta

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
    is_hindi = any('\u0900' <= c <= '\u097F' for c in text)
    is_punjabi = any('\u0A00' <= c <= '\u0A7F' for c in text)
    if is_hindi:
        roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ISO)
    elif is_punjabi:
        roman = sanscript.transliterate(text, sanscript.GURMUKHI, sanscript.ISO)
    else:
        return text.strip()
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
    words = segment.get('words', [])
    if not words: return [segment]
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
    if not segments: return []
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
    # New parameters for additional variants
    show_original_and_roman=False,   # if True, output two lines: original + romanized
    text_transform=None               # 'upper', 'lower', or None
):
    srt_content = ""
    count = 1

    processed_segments = []
    for seg in segments:
        processed_segments.extend(split_long_segment(seg, max_words=max_words_per_line))

    for seg in processed_segments:
        words = seg.get('words', [])
        if not words: continue

        # --- Prepare word texts ---
        original_words_text = [w['word'] for w in words]          # never romanized
        romanized_words_text = [to_romanized(w['word']) for w in words]

        # Decide which text to use for display
        if show_original_and_roman:
            # We will build two lines later; for now keep both
            display_words_text = original_words_text   # placeholder, not used directly
        else:
            # Normal mode: use romanized if requested
            if use_roman:
                display_words_text = romanized_words_text
            else:
                display_words_text = original_words_text

        # Apply text transform if requested
        if text_transform == 'upper':
            transform_func = str.upper
        elif text_transform == 'lower':
            transform_func = str.lower
        else:
            transform_func = lambda x: x

        if not show_original_and_roman:
            display_words_text = [transform_func(w) for w in display_words_text]

        # Pre-calculate fonts and colors if styling is enabled
        if include_styling:
            def get_random_tech_font():
                return random.choice(list(AVAILABLE_FONTS.values()))

            def resolve_font(name):
                return AVAILABLE_FONTS.get(name, 'Arial')

            word_base_fonts = []
            word_highlight_fonts = []
            for _ in words:
                if random_base_font:
                    word_base_fonts.append(get_random_tech_font())
                else:
                    word_base_fonts.append(resolve_font(base_font))
                if random_highlight_font:
                    word_highlight_fonts.append(get_random_tech_font())
                else:
                    word_highlight_fonts.append(resolve_font(highlight_font))

            word_base_colors = []
            word_highlight_colors = []
            def get_random_color():
                return "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()
            def hex_to_ass_tag(hex_c):
                h = hex_c.lstrip('#')
                if len(h) == 6:
                    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&"
                return hex_c
            for _ in words:
                if random_base_color:
                    word_base_colors.append(hex_to_ass_tag(get_random_color()))
                else:
                    word_base_colors.append(hex_to_ass_tag(base_color))
                if random_highlight_color:
                    word_highlight_colors.append(hex_to_ass_tag(get_random_color()))
                else:
                    word_highlight_colors.append(hex_to_ass_tag(highlight_color))

        # --- Generate subtitle entries ---
        if show_original_and_roman:
            # Create one subtitle with two lines: original (top) and romanized (bottom)
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])

            # Build original line (not romanized, but may be transformed)
            orig_line = " ".join([transform_func(w) for w in original_words_text])
            # Build romanized line (romanized and transformed)
            roman_line = " ".join([transform_func(w) for w in romanized_words_text])
            full_text = orig_line + "\\N" + roman_line

            if include_styling:
                # Apply base style to the whole subtitle (both lines)
                tags = []
                tags.append(f"\\fn{resolve_font(base_font)}")
                tags.append(f"\\fs{font_size}")
                tags.append(f"\\c{hex_to_ass_tag(base_color)}")
                tags.append("\\b1" if base_bold else "\\b0")
                tags.append("\\i1" if base_italic else "\\i0")
                tag_str = "".join(tags)
                line = f"{{{tag_str}}}{full_text}"
            else:
                line = full_text

            srt_content += f"{count}\n{start} --> {end}\n{line}\n\n"
            count += 1

        elif use_karaoke and include_styling:
            # Original karaoke mode (word by word)
            for i, current_word in enumerate(words):
                start = format_timestamp(current_word['start'])
                end = format_timestamp(current_word['end'])

                line_parts = []
                for j, word_text in enumerate(display_words_text):
                    if i == j:
                        font = word_highlight_fonts[j]
                        col = word_highlight_colors[j]
                        size = int(font_size * 1.2)
                        bold = highlight_bold
                        italic = highlight_italic
                    else:
                        font = word_base_fonts[j]
                        col = word_base_colors[j]
                        size = font_size
                        bold = base_bold
                        italic = base_italic

                    tags = []
                    tags.append(f"\\fn{font}")
                    tags.append(f"\\fs{size}")
                    tags.append(f"\\c{col}")
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
            text_line = " ".join(display_words_text)
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1

        else:
            # Standard mode with styling (single subtitle per segment)
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])

            line_parts = []
            for j, w_text in enumerate(display_words_text):
                font = word_base_fonts[j]
                col = word_base_colors[j]
                size = font_size
                bold = base_bold
                italic = base_italic

                tags = []
                tags.append(f"\\fn{font}")
                tags.append(f"\\fs{size}")
                tags.append(f"\\c{col}")
                tags.append("\\b1" if bold else "\\b0")
                tags.append("\\i1" if italic else "\\i0")
                tag_str = "".join(tags)
                line_parts.append(f"{{{tag_str}}}{w_text}")

            text_line = " ".join(line_parts)
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1

    return srt_content