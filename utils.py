import random
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
    random_base=False,
    random_highlight=False
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

        current_base_font = get_random_tech_font() if random_base else resolve_font(base_font)
        
        # Determine highlighting
        if use_karaoke:
            for i, current_word in enumerate(words):
                start = format_timestamp(current_word['start'])
                end = format_timestamp(current_word['end'])
                
                line_parts = []
                for j, word_text in enumerate(processed_words_text):
                    current_highlight_font = get_random_tech_font() if random_highlight else resolve_font(highlight_font)
                    
                    if i == j:
                        # Highlighted Word
                        # Increase size slightly for pop effect (1.2x)
                        line_parts.append(f'{{\\fn{current_highlight_font}}}{{\\fs{int(font_size * 1.2)}}}<font color="{highlight_color}">{word_text}</font>')
                    else:
                        # Base Word
                        line_parts.append(f'{{\\fn{current_base_font}}}{{\\fs{font_size}}}<font color="{base_color}">{word_text}</font>')
                
                text_line = " ".join(line_parts)
                srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
                count += 1
        else:
            # Standard Mode
            start = format_timestamp(words[0]['start'])
            end = format_timestamp(words[-1]['end'])
            
            line_parts = []
            for w_text in processed_words_text:
                line_parts.append(f'{{\\fn{current_base_font}}}{{\\fs{font_size}}}<font color="{base_color}">{w_text}</font>')
            
            text_line = " ".join(line_parts)
            srt_content += f"{count}\n{start} --> {end}\n{text_line}\n\n"
            count += 1
                
    return srt_content
