import streamlit as st
import os
import shutil
import subprocess
import pysubs2
import time
import torch
import faster_whisper
import platform
import gc
import glob
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

def transform_srt_case(srt_content, case_func):
    """Transform only the subtitle text lines to uppercase/lowercase."""
    blocks = srt_content.strip().split("\n\n")
    transformed_blocks = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            idx = lines[0]
            times = lines[1]
            # Text may be multi-line (e.g., if user added line breaks)
            text = "\n".join(lines[2:])
            transformed_text = case_func(text)
            transformed_blocks.append(f"{idx}\n{times}\n{transformed_text}")
        else:
            # Keep malformed blocks unchanged
            transformed_blocks.append(block)
    return "\n\n".join(transformed_blocks)

# --- MAIN APP ---
st.set_page_config(page_title="Subtitle Refiner Bot", page_icon="🎬")

st.title("🎬 AI Subtitle Generator & Refiner")
st.markdown("""Automated captions with **Roman Hindi/Punjabi** transliteration and **Karaoke Highlights**.
If using on Streamlit Cloud, please use 'auto' device and 'tiny' model only.
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
    # --- Text Case Transformation (updates widget key directly) ---
    with st.expander("✏️ Text Case Transformation", expanded=True):
        st.markdown("Apply uppercase/lowercase to the subtitle text currently shown in Step 2.")

        if st.button("⬆️ UPPERCASE", use_container_width=True):
            if "srt_editor_editable" in st.session_state:
                current = st.session_state["srt_editor_editable"]
                transformed = transform_srt_case(current, str.upper)
                variant = st.session_state.get("edit_choice", "Romanised (Editable)")
                # Update both the widget state and the persistent storage
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

    # 2. Transcribe
    if st.button("✨ Step 1: Generate Subtitles", type="primary"):
        with st.spinner(f"Transcribing with {model_size} model... (Using faster-whisper)"):
            try:
                temp_audio = os.path.join(output_dir, "audio.wav")
                subprocess.run(
                    [FFMPEG_PATH, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", temp_audio],
                    check=True, capture_output=True
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

                model = faster_whisper.WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=None,
                    cpu_threads=8 if device == "cpu" else 0,
                    num_workers=1
                )

                segments_generator, info = model.transcribe(
                    temp_audio,
                    word_timestamps=True,
                    language=None,
                    task="transcribe",
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    initial_prompt=None,
                    vad_filter=False,
                    vad_parameters=None
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

                merged_segments = merge_segments(word_level_segments, max_words=max_words)

                st.session_state['transcription_result'] = {
                    'raw_segments': word_level_segments,
                    'segments': merged_segments,
                    'language': info.language
                }

                st.success(f"Transcription Complete! Language: {info.language.upper()}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

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
            # Ensure both keys exist (in case of new session)
            if "Romanised (Editable)" not in st.session_state.edited_srt:
                st.session_state.edited_srt["Romanised (Editable)"] = roman_editable
            if "Original (Editable)" not in st.session_state.edited_srt:
                st.session_state.edited_srt["Original (Editable)"] = orig_editable

        # Radio to choose which variant to edit
        edit_choice = st.radio(
            "Select variant to edit:",
            ["Romanised (Editable)", "Original (Editable)"],
            horizontal=True,
            key="edit_choice"
        )

        # Manage the editor's session state when variant changes
        if "last_edit_choice" not in st.session_state:
            st.session_state.last_edit_choice = edit_choice

        if st.session_state.last_edit_choice != edit_choice:
            # Variant changed, update editor content to the saved text for the new variant
            st.session_state["srt_editor_editable"] = st.session_state.edited_srt[edit_choice]
            st.session_state.last_edit_choice = edit_choice

        # Initialize editor if not present
        if "srt_editor_editable" not in st.session_state:
            st.session_state["srt_editor_editable"] = st.session_state.edited_srt[edit_choice]

        # Text area for editing
        edited_text = st.text_area(
            f"Edit {edit_choice} (plain text only)",
            key="srt_editor_editable",
            height=300
        )

        # Save button
        if st.button("💾 Save Edits", key="save_edits"):
            st.session_state.edited_srt[edit_choice] = st.session_state["srt_editor_editable"]
            st.success(f"Edits saved for {edit_choice}!")

        # ------------------------------------------------------------
        # Helper to apply Romanised edits to segment list
        def apply_edits_to_segments(edited_srt, original_segments):
            """
            Replaces the text in original_segments with the text from edited_srt.
            Assumes the same number of subtitle blocks and same timings.
            Returns a new list of segments (each segment has 'start', 'end', and 'text').
            """
            blocks = edited_srt.strip().split("\n\n")
            if len(blocks) != len(original_segments):
                # Mismatch – fall back to original
                return original_segments

            new_segments = []
            for i, block in enumerate(blocks):
                lines = block.split("\n")
                if len(lines) >= 3:
                    # Text may be multi-line; join with spaces for a single line
                    text = " ".join(lines[2:]).strip()
                else:
                    text = ""
                # Copy timing from original
                new_segments.append({
                    "start": original_segments[i]["start"],
                    "end": original_segments[i]["end"],
                    "text": text,
                    # For simplicity, we don't add word-level info here
                })
            return new_segments
        # ------------------------------------------------------------

        # Step 3: Choose final style and burn
        st.divider()
        st.subheader("Step 3: Choose Subtitle Style & Burn to Video")

        st.markdown("""
        **Note:** If you select *Romanised (Editable)* or *Original (Editable)*, the burned subtitles will use the text you edited in Step 2.  
        For *Standard* and *Bilingual* styles, if you edited the Romanised version, those edits will be used (timings preserved).  
        *Karaoke* and *Lip Sync* always use the original transcription to maintain word‑level highlighting.
        """)

        # List of all possible subtitle types
        variant_names = [
            "Romanised (Editable)",
            "Original (Editable)",
            "Styled (Romanized)",
            "Karaoke (Highlighted)",
            "Bilingual (Original + Romanized)",
            "Lip Sync (Word‑level)"
        ]

        selected_variant = st.selectbox("Choose Subtitle Type for Burning:", variant_names)

        # Helper to get the final SRT content based on selection
        def get_final_srt(variant):
            # Determine if we should use edited Romanised text for certain variants
            use_edited = False
            edited_segments = None
            if "Romanised (Editable)" in st.session_state.edited_srt:
                edited_text = st.session_state.edited_srt["Romanised (Editable)"]
                # Check if it differs from the original romanised text
                original_roman = generate_srt_content(
                    segments, use_roman=True, use_karaoke=False,
                    max_words_per_line=max_words, include_styling=False
                )
                if edited_text != original_roman:
                    # Try to apply edits
                    edited_segments = apply_edits_to_segments(edited_text, segments)
                    if edited_segments is not None:
                        use_edited = True

            if variant == "Romanised (Editable)":
                # Return edited version if available, else generate fresh
                if "Romanised (Editable)" in st.session_state.edited_srt:
                    return st.session_state.edited_srt["Romanised (Editable)"]
                else:
                    return generate_srt_content(
                        segments, use_roman=True, use_karaoke=False,
                        max_words_per_line=max_words, include_styling=False
                    )
            elif variant == "Original (Editable)":
                if "Original (Editable)" in st.session_state.edited_srt:
                    return st.session_state.edited_srt["Original (Editable)"]
                else:
                    return generate_srt_content(
                        segments, use_roman=False, use_karaoke=False,
                        max_words_per_line=max_words, include_styling=False
                    )
            elif variant == "Standard (Romanized)":
                if use_edited:
                    # Use edited segments (which have only basic text, no words)
                    return generate_srt_content(
                        edited_segments, use_roman=True, use_karaoke=False,
                        max_words_per_line=max_words, font_size=font_size,
                        base_font=base_font, highlight_font=highlight_font,
                        random_base_font=random_base_font, random_highlight_font=random_highlight_font,
                        random_base_color=random_base_color, random_highlight_color=random_highlight_color,
                        base_bold=base_bold, base_italic=base_italic,
                        highlight_bold=highlight_bold, highlight_italic=highlight_italic,
                        include_styling=True,
                        highlight_color=highlight_color, base_color=base_color
                    )
                else:
                    return generate_srt_content(
                        segments, use_roman=True, use_karaoke=False,
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
                # Always use original segments (word-level required)
                return generate_srt_content(
                    segments, use_roman=True, use_karaoke=True,
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
                if use_edited:
                    return generate_srt_content(
                        edited_segments, use_roman=True,
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
                else:
                    return generate_srt_content(
                        segments, use_roman=True,
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
                # Always use original raw_segments (word-level required)
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

        # Optional preview
        with st.expander(f"📄 Preview of selected SRT", expanded=False):
            st.text_area("SRT Content", srt_text, height=200, disabled=True)

        # Download button
        st.download_button("⬇️ Download Selected SRT", srt_text, file_name="subtitles.srt")

        # Save the current SRT to a file for burning
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
                    # Save the two fixed variants (used for reference, not burning)
                    with open(dialogue_srt_path, "w", encoding="utf-8") as f:
                        f.write(get_final_srt("Standard (Romanized)"))
                    with open(karaoke_srt_path, "w", encoding="utf-8") as f:
                        f.write(get_final_srt("Karaoke (Highlighted)"))

                    # Use the current srt_text (which may be edited) for burning
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