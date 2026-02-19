import streamlit as st
import os
import shutil
import subprocess
import pysubs2
import time
import platform
import gc
import glob
import json
import requests
from tempfile import NamedTemporaryFile
from utils import AVAILABLE_FONTS, generate_srt_content, merge_segments

# Import fontTools (with fallback)
try:
    from fontTools.ttLib import TTFont
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    st.warning("fontTools not installed. Custom font names will be based on filenames.")

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
CUSTOM_FONTS_DIR = os.path.join(BASE_DIR, "custom_fonts")
os.makedirs(CUSTOM_FONTS_DIR, exist_ok=True)

# whisper.cpp paths
WHISPER_CPP_DIR = os.path.join(TOOLS_DIR, "whisper.cpp")
WHISPER_CLI = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")
MODELS_DIR = os.path.join(WHISPER_CPP_DIR, "models")
WHISPER_MODELS = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin"
}

# FFmpeg path
SYSTEM_FFMPEG = shutil.which("ffmpeg")
LOCAL_FFMPEG = os.path.join(TOOLS_DIR, "ffmpeg")
if os.path.exists(LOCAL_FFMPEG):
    FFMPEG_PATH = LOCAL_FFMPEG
elif SYSTEM_FFMPEG:
    FFMPEG_PATH = SYSTEM_FFMPEG
else:
    FFMPEG_PATH = "ffmpeg"

def get_font_family(font_path):
    """Return the font family name using fontTools, or filename if extraction fails."""
    if FONTTOOLS_AVAILABLE:
        try:
            with TTFont(font_path) as font:
                for record in font['name'].names:
                    if record.nameID == 1 and record.platformID == 3 and record.langID == 0x409:
                        return record.toStr()
                for record in font['name'].names:
                    if record.nameID == 16 and record.platformID == 3 and record.langID == 0x409:
                        return record.toStr()
                for record in font['name'].names:
                    if record.nameID in (1, 16) and record.platformID == 1:
                        return record.toStr()
        except Exception:
            pass
    return os.path.splitext(os.path.basename(font_path))[0]

def load_custom_fonts():
    custom_fonts = {}
    font_files = glob.glob(os.path.join(CUSTOM_FONTS_DIR, "*.ttf")) + \
                 glob.glob(os.path.join(CUSTOM_FONTS_DIR, "*.otf")) + \
                 glob.glob(os.path.join(CUSTOM_FONTS_DIR, "*.ttc"))
    for font_file in font_files:
        family = get_font_family(font_file)
        if family and family not in custom_fonts:
            custom_fonts[family] = family
    for display_name, family_name in custom_fonts.items():
        AVAILABLE_FONTS[display_name] = family_name
    return custom_fonts

load_custom_fonts()

def get_best_ffmpeg_encoder():
    system = platform.system()
    if system == "Darwin":
        return "h264_videotoolbox"
    if shutil.which("nvidia-smi"):
        return "h264_nvenc"
    return "libx264"

def get_whisper_gpu_flags(device_choice):
    """Return whisper.cpp flags for GPU based on user choice."""
    if device_choice == "CPU":
        return ["-ng", "0"]  # disable GPU
    if device_choice == "CUDA":
        # Check if CUDA is available (simple nvidia-smi check)
        if shutil.which("nvidia-smi"):
            return ["-ng", "1"]  # enable GPU
        else:
            st.warning("CUDA requested but nvidia-smi not found – falling back to CPU.")
            return ["-ng", "0"]
    # auto: enable GPU if possible
    if shutil.which("nvidia-smi") or platform.system() == "Darwin":
        return ["-ng", "1"]
    return ["-ng", "0"]

def ensure_model_exists(model_name):
    filename = WHISPER_MODELS.get(model_name)
    if not filename:
        return None
    model_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path

    st.info(f"📥 Downloading {model_name} model...")
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return model_path
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return None

def parse_whisper_json(json_path):
    """Extract word-level segments from whisper.cpp full JSON output."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    language = data.get('language', 'unknown')
    raw_segments = []
    for seg in data.get('segments', []):
        for word_info in seg.get('words', []):
            raw_segments.append({
                'start': word_info['start'],
                'end': word_info['end'],
                'text': word_info['word'].strip()
            })
    return raw_segments, language

def parse_srt_to_segments(srt_path):
    """Fallback: read SRT and return line-level segments (crude word splitting)."""
    segments = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    blocks = content.split('\n\n')
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            time_line = lines[1]
            text = ' '.join(lines[2:]).strip()
            # Parse time
            start_end = time_line.split(' --> ')
            if len(start_end) == 2:
                start = pysubs2.time.ms_from_str(start_end[0].replace(',', '.'))
                end = pysubs2.time.ms_from_str(start_end[1].replace(',', '.'))
                segments.append({'start': start/1000.0, 'end': end/1000.0, 'text': text})
    return segments

def transform_srt_case(srt_content, case_func):
    """Transform only the subtitle text lines to uppercase/lowercase."""
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
            transformed_blocks.append(block)
    return "\n\n".join(transformed_blocks)

# --- MAIN APP ---
st.set_page_config(page_title="Subtitle Refiner Bot", page_icon="🎬")

st.title("🎬 AI Subtitle Generator & Refiner")
st.markdown("""Automated captions with **Roman Hindi/Punjabi** transliteration and **Karaoke Highlights**.
Say thx to Mandeep for this amazing tool.
""")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("📁 Upload Custom Fonts", expanded=False):
        st.markdown("Upload `.ttf`, `.otf`, or `.ttc` files. They will be saved to `custom_fonts/` and appear in the font lists below.")
        uploaded_fonts = st.file_uploader(
            "Choose font files",
            type=["ttf", "otf", "ttc"],
            accept_multiple_files=True,
            key="font_uploader"
        )
        if uploaded_fonts:
            for uploaded_file in uploaded_fonts:
                save_path = os.path.join(CUSTOM_FONTS_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            load_custom_fonts()
            st.success(f"Uploaded {len(uploaded_fonts)} font(s). They are now available in the font dropdowns.")

    model_options = {
        "tiny": "tiny (multilingual)",
        "base": "base (multilingual)",
        "small": "small (multilingual)",
        "medium": "medium (multilingual)",
        "large-v3": "large-v3",
        "large-v3-turbo": "large-v3-turbo"
    }
    selected_label = st.selectbox(
        "Whisper Model",
        options=list(model_options.values()),
        index=1
    )
    model_size = [k for k, v in model_options.items() if v == selected_label][0]

    language_options = {
        "Auto (detect)": None,
        "Hindi": "hi",
        "Punjabi": "pa",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Chinese": "zh",
        "Arabic": "ar",
        "Bengali": "bn",
        "Urdu": "ur",
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Oriya": "or",
        "Assamese": "as",
        "Maithili": "mai",
        "Sindhi": "sd",
        "Nepali": "ne",
        "Sinhala": "si",
        "Khmer": "km",
        "Lao": "lo",
        "Thai": "th",
        "Vietnamese": "vi",
        "Indonesian": "id",
        "Malay": "ms",
        "Tagalog": "tl",
        "Burmese": "my",
    }
    selected_lang_label = st.selectbox(
        "Language",
        options=list(language_options.keys()),
        index=0,
        help="Select the language of the audio, or leave as Auto for detection."
    )
    selected_language_code = language_options[selected_lang_label]

    device_choice = st.radio(
        "Processing device",
        ["auto", "CPU", "CUDA"],
        index=0,
        help="auto = use GPU if available, else CPU"
    )

    st.divider()
    st.subheader("📏 Line & Font")
    max_words = st.slider("Max Words per Line", 3, 20, 8)
    font_size = st.slider("Font Size (px)", 10, 100, 16)

    st.divider()
    st.subheader("🎨 Colors")
    highlight_color = st.color_picker("Karaoke Highlight Color", "#FFFF00")
    base_color = st.color_picker("Base Text Color", "#FFFFFF")

    st.divider()
    st.subheader("🖼️ Border")
    border_enabled = st.checkbox("Enable border", value=True)
    if border_enabled:
        border_width = st.slider("Border width", 0.5, 5.0, 2.0, step=0.5)
        border_color = st.color_picker("Border color", "#000000")
    else:
        border_width = 0
        border_color = "#000000"

    st.divider()
    st.subheader("💡 Shadow")
    shadow_enabled = st.checkbox("Enable shadow", value=False)
    if shadow_enabled:
        shadow_color = st.color_picker("Shadow color", "#000000")
        shadow_distance = st.slider("Shadow distance", 0.5, 10.0, 2.0, step=0.5)
    else:
        shadow_color = "#000000"
        shadow_distance = 0

    st.divider()
    st.subheader("🔤 Fonts")
    font_keys = list(AVAILABLE_FONTS.keys())
    base_font = st.selectbox("Base Font", font_keys, index=0)
    highlight_font = st.selectbox("Highlight Font", font_keys, index=min(1, len(font_keys)-1))

    st.divider()
    st.subheader("🖋️ Text Style")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Base Text**")
        base_bold = st.checkbox("Bold")
        base_italic = st.checkbox("Italic")
    with col4:
        st.markdown("**Highlight Text**")
        highlight_bold = st.checkbox("Bold", key="highlight_bold")
        highlight_italic = st.checkbox("Italic", key="highlight_italic")

    st.divider()
    st.subheader("📍 Position")
    y_padding = st.slider("Vertical Padding (Y)", 0, 1000, 50)
    alignment_option = st.selectbox("Alignment", ["Center", "Left", "Right"], index=0)
    x_padding = st.slider("Horizontal Padding (X)", 0, 500, 20)

    st.divider()
    st.subheader("🎲 Randomization (Experimental)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fonts**")
        random_base_font = st.checkbox("Random Base Font")
        random_highlight_font = st.checkbox("Random Highlight Font")
    with col2:
        st.markdown("**Colors**")
        random_base_color = st.checkbox("Random Base Color")
        random_highlight_color = st.checkbox("Random Highlight Color")

    st.divider()
    with st.expander("✏️ Text Case Transformation", expanded=True):
        st.markdown("Apply uppercase/lowercase to the subtitle text currently shown in Step 2.")

        if st.button("⬆️ UPPERCASE", use_container_width=True):
            if "srt_editor_editable" in st.session_state:
                current = st.session_state["srt_editor_editable"]
                transformed = transform_srt_case(current, str.upper)
                variant = st.session_state.get("edit_choice", "Romanised (Editable)")
                st.session_state["srt_editor_editable"] = transformed
                st.session_state.edited_srt[variant] = transformed
                st.success(f"Converted to UPPERCASE for {variant}. Click Save Edits to keep changes.")

        if st.button("⬇️ lowercase", use_container_width=True):
            if "srt_editor_editable" in st.session_state:
                current = st.session_state["srt_editor_editable"]
                transformed = transform_srt_case(current, str.lower)
                variant = st.session_state.get("edit_choice", "Romanised (Editable)")
                st.session_state["srt_editor_editable"] = transformed
                st.session_state.edited_srt[variant] = transformed
                st.success(f"Converted to lowercase for {variant}. Click Save Edits to keep changes.")

    st.divider()
    if st.button("🧹 Clear Memory & Reset"):
        keys_to_clear = ['transcription_result', 'video_path', 'output_dir', 'video_name', 'last_uploaded_filename', 'edited_srt', 'srt_editor_editable', 'last_edit_choice']
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        gc.collect()
        st.rerun()

# 1. Upload Video
uploaded_file = st.file_uploader("Upload a Video (MOV, MP4)", type=["mov", "mp4", "mkv"])

if uploaded_file is not None:
    if 'last_uploaded_filename' not in st.session_state or st.session_state.get('last_uploaded_filename') != uploaded_file.name:
        video_name = os.path.splitext(uploaded_file.name)[0]
        output_dir = os.path.join(BASE_DIR, "outputs", video_name)
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, "original" + os.path.splitext(uploaded_file.name)[1])
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state['video_path'] = video_path
        st.session_state['output_dir'] = output_dir
        st.session_state['video_name'] = video_name
        st.session_state['last_uploaded_filename'] = uploaded_file.name

        if 'transcription_result' in st.session_state:
            del st.session_state['transcription_result']
        if 'edited_srt' in st.session_state:
            del st.session_state['edited_srt']
        st.rerun()

    video_path = st.session_state.get('video_path')
    output_dir = st.session_state.get('output_dir')
    video_name = st.session_state.get('video_name')
    st.video(video_path)

    # 2. Transcribe with whisper.cpp
    if st.button("✨ Step 1: Generate Subtitles", type="primary"):
        with st.spinner(f"Transcribing with {model_size} model... (Using whisper.cpp)"):
            try:
                # Check for whisper-cli binary
                if not os.path.exists(WHISPER_CLI):
                    st.error(f"❌ whisper-cli not found at {WHISPER_CLI}. Please build whisper.cpp first.")
                    st.stop()

                # Ensure model exists
                model_path = ensure_model_exists(model_size)
                if not model_path:
                    st.stop()

                # Extract audio
                temp_audio = os.path.join(output_dir, "audio.wav")
                subprocess.run(
                    [FFMPEG_PATH, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", temp_audio],
                    check=True, capture_output=True
                )

                # Build whisper.cpp command
                output_base = os.path.join(output_dir, "whisper_out")
                cmd = [
                    WHISPER_CLI,
                    "-m", model_path,
                    "-f", temp_audio,
                    "-ojf",        # full JSON (with word timestamps)
                    "-osrt",       # SRT fallback
                    "-of", output_base,
                    "-l", selected_language_code if selected_language_code else "auto",
                    "-t", "8",
                    "-bs", "5",
                    "-ml", "1",    # max line length 1 -> word-level
                    "-sow"         # split on word
                ]
                # Add GPU flags based on device choice
                cmd.extend(get_whisper_gpu_flags(device_choice))

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"Transcription failed: {result.stderr}")
                    st.stop()

                # Parse output
                json_path = output_base + ".json"
                srt_path = output_base + ".srt"

                raw_segments = []
                language = "unknown"
                if os.path.exists(json_path):
                    raw_segments, language = parse_whisper_json(json_path)
                elif os.path.exists(srt_path):
                    # Fallback: use SRT and crude word splitting
                    line_segments = parse_srt_to_segments(srt_path)
                    # Heuristically split lines into words (rough)
                    for seg in line_segments:
                        words = seg['text'].split()
                        duration = seg['end'] - seg['start']
                        per_word = duration / len(words) if words else 0
                        for i, w in enumerate(words):
                            raw_segments.append({
                                'start': seg['start'] + i * per_word,
                                'end': seg['start'] + (i + 1) * per_word,
                                'text': w
                            })
                else:
                    st.error("❌ Failed to generate subtitle output.")
                    if result.stderr:
                        st.info(f"Debug Info (stderr): {result.stderr}")
                    if result.stdout:
                        st.info(f"Debug Info (stdout): {result.stdout}")
                    st.stop()

                # Merge word-level segments into lines based on max_words
                merged_segments = merge_segments(raw_segments, max_words=max_words)

                st.session_state['transcription_result'] = {
                    'raw_segments': raw_segments,
                    'segments': merged_segments,
                    'language': language
                }

                st.success(f"Transcription Complete! Language: {language.upper()}")

            except Exception as e:
                st.error(f"Transcription Failed: {e}")

    # 3. Step 2: Edit Plain Text Subtitles (Editable variants only)
    if 'transcription_result' in st.session_state:
        st.divider()
        st.subheader("Step 2: Edit Plain Text Subtitles")

        st.markdown("""
        Choose the subtitle variant you want to edit, then modify the text below.
        **Do not change timestamps or numbers.** Your edits are saved when you click the *Save Edits* button.
        """)

        if 'language' in st.session_state['transcription_result']:
            lang = st.session_state['transcription_result']['language']
            st.info(f"🌍 Auto-Detected Language: **{lang.upper()}**")

        segments = st.session_state['transcription_result']['segments']
        raw_segments = st.session_state['transcription_result']['raw_segments']

        # Generate the two editable variants (plain text, no styling)
        roman_editable = generate_srt_content(
            segments, use_roman=True, use_karaoke=False,
            max_words_per_line=max_words,
            include_styling=False
        )
        orig_editable = generate_srt_content(
            segments, use_roman=False, use_karaoke=False,
            max_words_per_line=max_words,
            include_styling=False
        )

        # Initialize edited_srt if needed
        if 'edited_srt' not in st.session_state:
            st.session_state.edited_srt = {
                "Romanised (Editable)": roman_editable,
                "Original (Editable)": orig_editable
            }
        else:
            if "Romanised (Editable)" not in st.session_state.edited_srt:
                st.session_state.edited_srt["Romanised (Editable)"] = roman_editable
            if "Original (Editable)" not in st.session_state.edited_srt:
                st.session_state.edited_srt["Original (Editable)"] = orig_editable

        edit_choice = st.radio(
            "Select variant to edit:",
            ["Romanised (Editable)", "Original (Editable)"],
            horizontal=True,
            key="edit_choice"
        )

        if "last_edit_choice" not in st.session_state:
            st.session_state.last_edit_choice = edit_choice

        if st.session_state.last_edit_choice != edit_choice:
            st.session_state["srt_editor_editable"] = st.session_state.edited_srt[edit_choice]
            st.session_state.last_edit_choice = edit_choice

        if "srt_editor_editable" not in st.session_state:
            st.session_state["srt_editor_editable"] = st.session_state.edited_srt[edit_choice]

        edited_text = st.text_area(
            f"Edit {edit_choice} (plain text only)",
            key="srt_editor_editable",
            height=300
        )

        if st.button("💾 Save Edits", key="save_edits"):
            st.session_state.edited_srt[edit_choice] = st.session_state["srt_editor_editable"]
            st.success(f"Edits saved for {edit_choice}!")

        # Helper to apply Romanised edits to segment list
        def apply_edits_to_segments(edited_srt, original_segments):
            blocks = edited_srt.strip().split("\n\n")
            if len(blocks) != len(original_segments):
                return original_segments
            new_segments = []
            for i, block in enumerate(blocks):
                lines = block.split("\n")
                if len(lines) >= 3:
                    text = " ".join(lines[2:]).strip()
                else:
                    text = ""
                new_segments.append({
                    "start": original_segments[i]["start"],
                    "end": original_segments[i]["end"],
                    "text": text,
                })
            return new_segments

        # Step 3: Choose final style and burn
        st.divider()
        st.subheader("Step 3: Choose Subtitle Style & Burn to Video")

        st.markdown("""
        **Note:** If you select *Romanised (Editable)* or *Original (Editable)*, the burned subtitles will use the text you edited in Step 2.  
        For *Standard* and *Bilingual* styles, if you edited the Romanised version, those edits will be used (timings preserved).  
        *Karaoke* and *Lip Sync* always use the original transcription to maintain word‑level highlighting.
        """)

        variant_names = [
            "Romanised (Editable)",
            "Original (Editable)",
            "Styled (Romanized)",
            "Karaoke (Highlighted)",
            "Bilingual (Original + Romanized)",
            "Lip Sync (Word‑level)"
        ]

        selected_variant = st.selectbox("Choose Subtitle Type for Burning:", variant_names)

        def get_final_srt(variant):
            use_edited = False
            edited_segments = None
            if "Romanised (Editable)" in st.session_state.edited_srt:
                edited_text = st.session_state.edited_srt["Romanised (Editable)"]
                original_roman = generate_srt_content(
                    segments, use_roman=True, use_karaoke=False,
                    max_words_per_line=max_words, include_styling=False
                )
                if edited_text != original_roman:
                    edited_segments = apply_edits_to_segments(edited_text, segments)
                    if edited_segments is not None:
                        use_edited = True

            if variant == "Romanised (Editable)":
                return st.session_state.edited_srt.get("Romanised (Editable)", roman_editable)
            elif variant == "Original (Editable)":
                return st.session_state.edited_srt.get("Original (Editable)", orig_editable)
            elif variant == "Styled (Romanized)":
                base_segs = edited_segments if use_edited else segments
                return generate_srt_content(
                    base_segs, use_roman=True, use_karaoke=False,
                    max_words_per_line=max_words, font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font, random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color, random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            elif variant == "Karaoke (Highlighted)":
                return generate_srt_content(
                    raw_segments, use_roman=True, use_karaoke=True,
                    max_words_per_line=max_words, font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font, random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color, random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            elif variant == "Bilingual (Original + Romanized)":
                base_segs = edited_segments if use_edited else segments
                return generate_srt_content(
                    base_segs, use_roman=True,
                    show_original_and_roman=True,
                    max_words_per_line=max_words, font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font, random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color, random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            elif variant == "Lip Sync (Word‑level)":
                return generate_srt_content(
                    raw_segments, use_roman=True, use_karaoke=False,
                    max_words_per_line=1,
                    font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font, random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color, random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            else:
                return ""

        srt_text = get_final_srt(selected_variant)

        with st.expander(f"📄 Preview of selected SRT", expanded=False):
            st.text_area("SRT Content", srt_text, height=200, disabled=True)

        st.download_button("⬇️ Download Selected SRT", srt_text, file_name="subtitles.srt")

        srt_path = video_path + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)

        # 4. Burn Video
        if st.button("🔥 Create Final Video"):
            output_video_path = os.path.join(output_dir, "final_video.mov")
            dialogue_srt_path = os.path.join(output_dir, "dialogue.srt")
            karaoke_srt_path = os.path.join(output_dir, "lip_sync.srt")

            with st.spinner("Burning subtitles & Saving to Folder..."):
                try:
                    with open(dialogue_srt_path, "w", encoding="utf-8") as f:
                        f.write(get_final_srt("Styled (Romanized)"))
                    with open(karaoke_srt_path, "w", encoding="utf-8") as f:
                        f.write(get_final_srt("Karaoke (Highlighted)"))

                    temp_srt_path = os.path.join(output_dir, "temp_burn.srt")
                    with open(temp_srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_text)

                    subs = pysubs2.load(temp_srt_path, encoding="utf-8")

                    def hex_to_ass(hex_color):
                        hex_color = hex_color.lstrip('#')
                        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                        return f"&H00{b}{g}{r}"

                    ass_base_color = hex_to_ass(base_color)

                    if border_enabled:
                        ass_border_color = hex_to_ass(border_color)
                        outline_width = border_width
                    else:
                        ass_border_color = "&H00000000"
                        outline_width = 0

                    if shadow_enabled:
                        ass_shadow_color = hex_to_ass(shadow_color)
                        shadow_dist = int(round(shadow_distance))
                    else:
                        ass_shadow_color = "&H00000000"
                        shadow_dist = 0

                    tech_base_font = AVAILABLE_FONTS.get(base_font, 'Arial')
                    align_map = {"Center": 2, "Left": 1, "Right": 3}

                    style = pysubs2.SSAStyle(
                        fontname=tech_base_font,
                        fontsize=font_size,
                        primarycolor=ass_base_color,
                        outlinecolor=ass_border_color,
                        backcolor=ass_shadow_color,
                        borderstyle=1,
                        outline=outline_width,
                        shadow=shadow_dist,
                        alignment=align_map[alignment_option],
                        marginv=y_padding,
                        marginl=x_padding,
                        marginr=x_padding
                    )
                    subs.styles["Default"] = style
                    ass_path = temp_srt_path + ".ass"
                    subs.save(ass_path)

                    escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
                    escaped_fonts_dir = CUSTOM_FONTS_DIR.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
                    filter_str = f"ass='{escaped_ass_path}':fontsdir='{escaped_fonts_dir}'"

                    encoder = get_best_ffmpeg_encoder()

                    def build_cmd(enc):
                        cmd = [
                            FFMPEG_PATH, "-y",
                            "-i", video_path,
                            "-vf", filter_str,
                            "-c:v", enc, "-b:v", "8M",
                            "-c:a", "copy",
                            output_video_path
                        ]
                        if enc == "h264_videotoolbox":
                            cmd.insert(10, "-realtime")
                            cmd.insert(11, "1")
                        return cmd

                    try:
                        cmd = build_cmd(encoder)
                        subprocess.run(cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        if encoder in ["h264_nvenc", "h264_videotoolbox"]:
                            st.warning(f"{encoder} failed, falling back to software encoder (libx264). This may be slower.")
                            cmd = build_cmd("libx264")
                            subprocess.run(cmd, check=True, capture_output=True)
                        else:
                            raise e

                    if os.path.exists(temp_srt_path): os.remove(temp_srt_path)
                    if os.path.exists(ass_path): os.remove(ass_path)

                    st.success(f"Done! Files saved in: outputs/{video_name}/")
                    st.video(output_video_path)

                    with open(output_video_path, "rb") as v_file:
                        st.download_button("⬇️ Download Final Video", v_file.read(), file_name="final_video.mov")

                except subprocess.CalledProcessError as e:
                    st.error(f"FFmpeg Error: {e.stderr.decode() if e.stderr else str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")