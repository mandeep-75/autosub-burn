import streamlit as st
import os
import shutil
import subprocess
import pysubs2
import torch
import faster_whisper
import platform
import gc
import glob
import random
from typing import List, Dict, Optional, Tuple
from utils import AVAILABLE_FONTS, generate_srt_content, merge_segments
# Optional: memory debugging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional: fontTools for font family extraction
try:
    from fontTools.ttLib import TTFont
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
CUSTOM_FONTS_DIR = os.path.join(BASE_DIR, "custom_fonts")
os.makedirs(CUSTOM_FONTS_DIR, exist_ok=True)

# FFmpeg detection
SYSTEM_FFMPEG = shutil.which("ffmpeg")
LOCAL_FFMPEG = os.path.join(TOOLS_DIR, "ffmpeg")
if os.path.exists(LOCAL_FFMPEG):
    FFMPEG_PATH = LOCAL_FFMPEG
elif SYSTEM_FFMPEG:
    FFMPEG_PATH = SYSTEM_FFMPEG
else:
    FFMPEG_PATH = "ffmpeg"

# --- HELPER FUNCTIONS (local) ---
def get_font_family(font_path: str) -> str:
    """Extract font family name using fontTools, fallback to filename."""
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

def get_best_ffmpeg_encoder() -> str:
    system = platform.system()
    if system == "Darwin":
        return "h264_videotoolbox"
    if shutil.which("nvidia-smi"):
        return "h264_nvenc"
    return "libx264"

def hex_to_ass(hex_color: str) -> str:
    """Convert #RRGGBB to ASS color format &H00BBGGRR (opaque)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H00{b}{g}{r}"

def transform_srt_case(srt_content: str, case_func) -> str:
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

def apply_edits_to_segments(edited_srt: str, original_segments: List[Dict]) -> Optional[List[Dict]]:
    """
    Replace text in original_segments with the edited text from an SRT string.
    Returns new segments list if block count matches, otherwise None.
    """
    blocks = edited_srt.strip().split("\n\n")
    if len(blocks) != len(original_segments):
        return None
    new_segments = []
    for i, block in enumerate(blocks):
        lines = block.split("\n")
        if len(lines) >= 3:
            text = " ".join(lines[2:]).strip()
        else:
            text = ""
        new_seg = original_segments[i].copy()
        new_seg["text"] = text
        new_segments.append(new_seg)
    return new_segments

# --- STREAMLIT APP ---
st.set_page_config(page_title="Subtitle Refiner Bot", page_icon="🎬")

st.title("🎬 AI Subtitle Generator & Refiner")
st.markdown("""Automated captions with **Roman Hindi/Punjabi** transliteration and **Karaoke Highlights**.
Say thx to Mandeep for this amazing tool.
""")

# Initialize session state keys
if 'edited_srt' not in st.session_state:
    st.session_state.edited_srt = {}
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None
if 'video_name' not in st.session_state:
    st.session_state.video_name = None
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'srt_editor_editable' not in st.session_state:
    st.session_state.srt_editor_editable = ""
if 'edit_choice' not in st.session_state:
    st.session_state.edit_choice = "Romanised (Editable)"
if 'last_edit_choice' not in st.session_state:
    st.session_state.last_edit_choice = st.session_state.edit_choice
if 'last_model_params' not in st.session_state:
    st.session_state.last_model_params = None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("📁 Upload Custom Fonts", expanded=False):
        st.markdown("Upload `.ttf`, `.otf`, or `.ttc` files. They will be saved to `custom_fonts/` and appear in font lists.")
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
            # Reload fonts
            st.cache_data.clear()
            load_custom_fonts()
            st.success(f"Uploaded {len(uploaded_fonts)} font(s). They are now available.")

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

    # Clear resource cache if model parameters change
    current_params = (model_size, device_choice)
    if st.session_state.last_model_params != current_params:
        st.cache_resource.clear()
        st.session_state.last_model_params = current_params

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

    # Output bitrate control
    st.divider()
    st.subheader("📦 Output Quality")
    bitrate = st.select_slider(
        "Video Bitrate",
        options=["1M", "2M", "4M", "8M", "16M"],
        value="4M",
        help="Higher bitrate = better quality but larger file."
    )

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

    # Memory debug
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        st.caption(f"📊 RAM: {mem:.1f} MB")

    if st.button("🧹 Clear Memory & Reset"):
        keys_to_clear = ['transcription_result', 'video_path', 'output_dir', 'video_name',
                         'last_uploaded_filename', 'edited_srt', 'srt_editor_editable',
                         'edit_choice', 'last_edit_choice', 'last_model_params']
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- MAIN AREA ---

# 1. Upload Video
uploaded_file = st.file_uploader("Upload a Video (MOV, MP4, MKV)", type=["mov", "mp4", "mkv"])

if uploaded_file is not None:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        # Clear previous video data
        for key in ['transcription_result', 'edited_srt', 'srt_editor_editable']:
            if key in st.session_state:
                del st.session_state[key]

        video_name = os.path.splitext(uploaded_file.name)[0]
        output_dir = os.path.join(BASE_DIR, "outputs", video_name)
        os.makedirs(output_dir, exist_ok=True)

        # Stream write to file (avoid loading entire file into memory)
        video_ext = os.path.splitext(uploaded_file.name)[1]
        video_path = os.path.join(output_dir, "original" + video_ext)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        st.session_state.video_path = video_path
        st.session_state.output_dir = output_dir
        st.session_state.video_name = video_name
        st.session_state.last_uploaded_filename = uploaded_file.name

        st.rerun()

    video_path = st.session_state.video_path
    output_dir = st.session_state.output_dir
    video_name = st.session_state.video_name

    st.video(video_path)

    # 2. Transcribe
    if st.button("✨ Step 1: Generate Subtitles", type="primary"):
        with st.spinner(f"Transcribing with {model_size} model... (Using faster-whisper)"):
            try:
                # Extract audio using FFmpeg (discard output to avoid memory)
                temp_audio = os.path.join(output_dir, "audio.wav")
                subprocess.run(
                    [FFMPEG_PATH, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", temp_audio],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )

                if device_choice == "auto":
                    use_cuda = torch.cuda.is_available()
                elif device_choice == "CUDA":
                    use_cuda = torch.cuda.is_available()
                    if not use_cuda:
                        st.warning("CUDA requested but not available – falling back to CPU.")
                else:
                    use_cuda = False

                device = "cuda" if use_cuda else "cpu"
                compute_type = "float16" if use_cuda else "int8"

                @st.cache_resource
                def get_whisper_model(model_size, device, compute_type):
                    return faster_whisper.WhisperModel(
                        model_size,
                        device=device,
                        compute_type=compute_type,
                        download_root=None,
                        cpu_threads=8 if device == "cpu" else 0,
                        num_workers=1
                    )

                model = get_whisper_model(model_size, device, compute_type)

                segments_generator, info = model.transcribe(
                    temp_audio,
                    word_timestamps=True,
                    language=selected_language_code,
                    task="transcribe",
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    vad_filter=False,
                )

                word_level_segments = []
                for segment in segments_generator:
                    if segment.words:
                        for word in segment.words:
                            word_level_segments.append({
                                "start": word.start,
                                "end": word.end,
                                "text": word.word,
                                "words": [{
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end
                                }]
                            })
                    else:
                        words = segment.text.split()
                        duration = segment.end - segment.start
                        per_word = duration / len(words) if words else 0
                        for i, w in enumerate(words):
                            start = segment.start + i * per_word
                            end = segment.start + (i + 1) * per_word
                            word_level_segments.append({
                                "start": start,
                                "end": end,
                                "text": w,
                                "words": [{"word": w, "start": start, "end": end}]
                            })

                # Use merge_segments from utils (assumed to work correctly)
                merged_segments = merge_segments(word_level_segments, max_words=max_words)

                st.session_state.transcription_result = {
                    'raw_segments': word_level_segments,
                    'segments': merged_segments,
                    'language': info.language
                }

                # Generate editable SRTs (plain, no styling)
                roman_editable = generate_srt_content(
                    merged_segments, use_roman=True, use_karaoke=False,
                    max_words_per_line=max_words,
                    include_styling=False
                )
                orig_editable = generate_srt_content(
                    merged_segments, use_roman=False, use_karaoke=False,
                    max_words_per_line=max_words,
                    include_styling=False
                )

                st.session_state.edited_srt = {
                    "Romanised (Editable)": roman_editable,
                    "Original (Editable)": orig_editable
                }
                st.session_state.srt_editor_editable = roman_editable
                st.session_state.edit_choice = "Romanised (Editable)"
                st.session_state.last_edit_choice = "Romanised (Editable)"

                st.success(f"Transcription Complete! Language: {info.language.upper()}")

                # Cleanup
                del segments_generator, info, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)

            except subprocess.CalledProcessError as e:
                st.error(f"FFmpeg error: {e}")
            except Exception as e:
                st.error(f"Transcription Failed: {e}")

    # 3. Step 2: Edit Plain Text Subtitles
    if st.session_state.transcription_result:
        st.divider()
        st.subheader("Step 2: Edit Plain Text Subtitles")

        st.markdown("""
        Choose the subtitle variant you want to edit, then modify the text below.
        **Do not change timestamps or numbers.** Your edits are saved when you click the *Save Edits* button.
        """)

        lang = st.session_state.transcription_result.get('language', 'unknown')
        st.info(f"🌍 Auto-Detected Language: **{lang.upper()}**")

        segments = st.session_state.transcription_result['segments']
        raw_segments = st.session_state.transcription_result['raw_segments']

        edit_choice = st.radio(
            "Select variant to edit:",
            ["Romanised (Editable)", "Original (Editable)"],
            horizontal=True,
            key="edit_choice"
        )

        if st.session_state.last_edit_choice != edit_choice:
            st.session_state.srt_editor_editable = st.session_state.edited_srt.get(edit_choice, "")
            st.session_state.last_edit_choice = edit_choice

        edited_text = st.text_area(
            f"Edit {edit_choice} (plain text only)",
            key="srt_editor_editable",
            height=300
        )

        if st.button("💾 Save Edits", key="save_edits"):
            st.session_state.edited_srt[edit_choice] = st.session_state.srt_editor_editable
            st.success(f"Edits saved for {edit_choice}!")

        # Case transformation with callbacks
        def to_uppercase():
            if st.session_state.srt_editor_editable:
                transformed = transform_srt_case(st.session_state.srt_editor_editable, str.upper)
                st.session_state.srt_editor_editable = transformed
                st.session_state.edited_srt[st.session_state.edit_choice] = transformed

        def to_lowercase():
            if st.session_state.srt_editor_editable:
                transformed = transform_srt_case(st.session_state.srt_editor_editable, str.lower)
                st.session_state.srt_editor_editable = transformed
                st.session_state.edited_srt[st.session_state.edit_choice] = transformed

        def reset_to_original():
            if st.session_state.edit_choice == "Romanised (Editable)":
                original = generate_srt_content(segments, use_roman=True, use_karaoke=False,
                                                max_words_per_line=max_words, include_styling=False)
            else:
                original = generate_srt_content(segments, use_roman=False, use_karaoke=False,
                                                max_words_per_line=max_words, include_styling=False)
            st.session_state.srt_editor_editable = original
            st.session_state.edited_srt[st.session_state.edit_choice] = original

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.button("⬆️ UPPERCASE", on_click=to_uppercase, use_container_width=True)
        with col_b:
            st.button("⬇️ lowercase", on_click=to_lowercase, use_container_width=True)
        with col_c:
            st.button("↩️ Reset to Original", on_click=reset_to_original, use_container_width=True)

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

        # Helper to get final SRT and a descriptive caption
        def get_final_srt(variant: str) -> Tuple[str, str]:
            # For variants that may use edited Romanised text
            if variant in ["Styled (Romanized)", "Bilingual (Original + Romanized)"]:
                use_edited = False
                edited_segments = None
                if "Romanised (Editable)" in st.session_state.edited_srt:
                    edited_text = st.session_state.edited_srt["Romanised (Editable)"]
                    original_roman = generate_srt_content(segments, use_roman=True, use_karaoke=False,
                                                          max_words_per_line=max_words, include_styling=False)
                    if edited_text != original_roman:
                        edited_segments = apply_edits_to_segments(edited_text, segments)
                        if edited_segments is not None:
                            use_edited = True
                segs = edited_segments if use_edited else segments
                caption = "✅ Using your edited Romanised text." if use_edited else "ℹ️ Using original transcription (no edits or edits don't apply)."
            elif variant == "Romanised (Editable)":
                srt = st.session_state.edited_srt.get("Romanised (Editable)", "")
                original = generate_srt_content(segments, use_roman=True, use_karaoke=False,
                                                max_words_per_line=max_words, include_styling=False)
                caption = "✏️ Using your edited Romanised text." if srt != original else "ℹ️ No edits applied."
                return srt, caption
            elif variant == "Original (Editable)":
                srt = st.session_state.edited_srt.get("Original (Editable)", "")
                original = generate_srt_content(segments, use_roman=False, use_karaoke=False,
                                                max_words_per_line=max_words, include_styling=False)
                caption = "✏️ Using your edited Original text." if srt != original else "ℹ️ No edits applied."
                return srt, caption
            elif variant == "Karaoke (Highlighted)":
                segs = segments
                caption = "🎤 Using original word-level transcription (karaoke)."
            elif variant == "Lip Sync (Word‑level)":
                segs = raw_segments
                caption = "🎤 Using original word-level transcription (lip sync)."
            else:
                segs = segments
                caption = "ℹ️ Using original transcription."

            # Generate SRT with appropriate parameters, passing the original flags
            if variant == "Styled (Romanized)":
                srt = generate_srt_content(
                    segs, use_roman=True, use_karaoke=False,
                    max_words_per_line=max_words, font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font,
                    random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color,
                    random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            elif variant == "Karaoke (Highlighted)":
                srt = generate_srt_content(
                    segs, use_roman=True, use_karaoke=True,
                    max_words_per_line=max_words, font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font,
                    random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color,
                    random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            elif variant == "Bilingual (Original + Romanized)":
                srt = generate_srt_content(
                    segs, use_roman=True,
                    show_original_and_roman=True,
                    max_words_per_line=max_words, font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font,
                    random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color,
                    random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            elif variant == "Lip Sync (Word‑level)":
                srt = generate_srt_content(
                    segs, use_roman=True, use_karaoke=False,
                    max_words_per_line=1,
                    font_size=font_size,
                    base_font=base_font, highlight_font=highlight_font,
                    random_base_font=random_base_font,
                    random_highlight_font=random_highlight_font,
                    random_base_color=random_base_color,
                    random_highlight_color=random_highlight_color,
                    base_bold=base_bold, base_italic=base_italic,
                    highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                    include_styling=True,
                    highlight_color=highlight_color, base_color=base_color
                )
            else:
                srt = ""

            return srt, caption

        srt_text, caption = get_final_srt(selected_variant)
        st.caption(caption)

        # Preview expander (expanded=True)
        with st.expander(f"📄 Preview of selected SRT", expanded=True):
            st.text_area("SRT Content", srt_text, height=200, disabled=True, key=f"preview_{selected_variant}")

        st.download_button("⬇️ Download Selected SRT", srt_text, file_name="subtitles.srt")

        # Save current SRT to file for burning
        srt_path = os.path.join(output_dir, "current.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)

        # Burn button
        if st.button("🔥 Create Final Video"):
            output_video_path = os.path.join(output_dir, "final_video.mov")
            # Also save fixed variants for reference
            dialogue_srt_path = os.path.join(output_dir, "dialogue.srt")
            karaoke_srt_path = os.path.join(output_dir, "lip_sync.srt")
            with open(dialogue_srt_path, "w", encoding="utf-8") as f:
                f.write(get_final_srt("Styled (Romanized)")[0])
            with open(karaoke_srt_path, "w", encoding="utf-8") as f:
                f.write(get_final_srt("Karaoke (Highlighted)")[0])

            ass_path = None
            with st.spinner("Burning subtitles & Saving to Folder..."):
                try:
                    subs = pysubs2.load(srt_path, encoding="utf-8")

                    ass_base_color = hex_to_ass(base_color)
                    ass_highlight_color = hex_to_ass(highlight_color)  # for karaoke secondary color

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
                        secondarycolor=ass_highlight_color,  # crucial for karaoke
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
                    ass_path = srt_path + ".ass"
                    subs.save(ass_path)

                    # Use subtitles filter for better path handling
                    filter_str = f"subtitles='{ass_path}'"
                    if CUSTOM_FONTS_DIR and os.path.exists(CUSTOM_FONTS_DIR):
                        filter_str += f":fontsdir='{CUSTOM_FONTS_DIR}'"

                    encoder = get_best_ffmpeg_encoder()

                    cmd = [
                        FFMPEG_PATH, "-y",
                        "-i", video_path,
                        "-vf", filter_str,
                        "-c:v", encoder, "-b:v", bitrate,
                        "-c:a", "copy",
                        output_video_path
                    ]
                    if encoder == "h264_videotoolbox":
                        cmd.insert(10, "-realtime")
                        cmd.insert(11, "1")

                    # Run ffmpeg, discarding output
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                    st.success(f"Done! Files saved in: outputs/{video_name}/")
                    st.video(output_video_path)

                    with open(output_video_path, "rb") as v_file:
                        st.download_button("⬇️ Download Final Video", v_file.read(), file_name="final_video.mov")

                except subprocess.CalledProcessError as e:
                    st.error(f"FFmpeg Error: {e}")
                    if encoder != "libx264":
                        st.warning("Trying fallback software encoder (libx264).")
                        try:
                            cmd = [
                                FFMPEG_PATH, "-y",
                                "-i", video_path,
                                "-vf", filter_str,
                                "-c:v", "libx264", "-b:v", bitrate,
                                "-c:a", "copy",
                                output_video_path
                            ]
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                            st.success("Fallback successful!")
                        except Exception as e2:
                            st.error(f"Fallback also failed: {e2}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    if ass_path and os.path.exists(ass_path):
                        try:
                            os.remove(ass_path)
                        except:
                            pass
                    if os.path.exists(srt_path):
                        try:
                            os.remove(srt_path)
                        except:
                            pass
                    # Flush CUDA cache after video processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()