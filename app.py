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
from tempfile import NamedTemporaryFile
from utils import AVAILABLE_FONTS, generate_srt_content, merge_segments

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")

SYSTEM_FFMPEG = shutil.which("ffmpeg")
LOCAL_FFMPEG = os.path.join(TOOLS_DIR, "ffmpeg")

if os.path.exists(LOCAL_FFMPEG):
    FFMPEG_PATH = LOCAL_FFMPEG
elif SYSTEM_FFMPEG:
    FFMPEG_PATH = SYSTEM_FFMPEG
else:
    FFMPEG_PATH = "ffmpeg"


def get_best_ffmpeg_encoder():
    system = platform.system()
    if system == "Darwin":
        return "h264_videotoolbox"
    if shutil.which("nvidia-smi"):
        return "h264_nvenc"
    return "libx264"


# --- MAIN APP ---
st.set_page_config(page_title="Subtitle Refiner Bot", page_icon="🎬")

st.title("🎬 AI Subtitle Generator & Refiner")
st.markdown("Automated captions with **Roman Hindi/Punjabi** transliteration and **Karaoke Highlights**.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
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
    highlight_font = st.selectbox("Highlight Font", font_keys, index=1)

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
    if st.button("🧹 Clear Memory & Reset"):
        keys_to_clear = ['transcription_result', 'video_path', 'output_dir', 'video_name', 'last_uploaded_filename']
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
                    cpu_threads=4 if device == "cpu" else 0,
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

                # Store both raw and merged segments
                st.session_state['transcription_result'] = {
                    'raw_segments': word_level_segments,
                    'segments': merged_segments,
                    'language': info.language
                }

                st.success(f"Transcription Complete! Language: {info.language.upper()}")

                # Clean up model from memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                st.error(f"Transcription Failed: {e}")

    # 3. Refine & Preview
    if 'transcription_result' in st.session_state:
        st.divider()
        st.subheader("Step 2: Select Subtitle Style")

        if 'language' in st.session_state['transcription_result']:
            lang = st.session_state['transcription_result']['language']
            st.info(f"🌍 Auto-Detected Language: **{lang.upper()}**")

        segments = st.session_state['transcription_result']['segments']
        raw_segments = st.session_state['transcription_result']['raw_segments']

        variants = {}

        # Variant 1: Original (no roman, no karaoke) – styled
        variants["Original (Auto-Detected)"] = generate_srt_content(
            segments, use_roman=False, use_karaoke=False,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base_font=random_base_font, random_highlight_font=random_highlight_font,
            random_base_color=random_base_color, random_highlight_color=random_highlight_color,
            base_bold=base_bold, base_italic=base_italic,
            highlight_bold=highlight_bold, highlight_italic=highlight_italic
        )

        # Variant 2: Standard (Romanized) – styled
        variants["Standard (Romanized)"] = generate_srt_content(
            segments, use_roman=True, use_karaoke=False,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base_font=random_base_font, random_highlight_font=random_highlight_font,
            random_base_color=random_base_color, random_highlight_color=random_highlight_color,
            base_bold=base_bold, base_italic=base_italic,
            highlight_bold=highlight_bold, highlight_italic=highlight_italic
        )

        # Variant 3: Karaoke (Highlighted) – styled
        variants["Karaoke (Highlighted)"] = generate_srt_content(
            segments, use_roman=True, use_karaoke=True,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base_font=random_base_font, random_highlight_font=random_highlight_font,
            random_base_color=random_base_color, random_highlight_color=random_highlight_color,
            base_bold=base_bold, base_italic=base_italic,
            highlight_bold=highlight_bold, highlight_italic=highlight_italic
        )

        # Variant 4: Simple (No Styling) – romanized, plain
        variants["Simple (No Styling)"] = generate_srt_content(
            segments, use_roman=True, use_karaoke=False,
            max_words_per_line=max_words,
            include_styling=False
        )

        # Variant 5: Lip Sync (Word‑level) – plain
        variants["Lip Sync (Word‑level)"] = generate_srt_content(
            raw_segments, use_roman=True, use_karaoke=False,
            max_words_per_line=1,
            include_styling=False
        )

        # Variant 6: Lip Sync (Word‑level) Styled – styled
        variants["Lip Sync (Word‑level) Styled"] = generate_srt_content(
            raw_segments, use_roman=True, use_karaoke=False,
            max_words_per_line=1,
            highlight_color=highlight_color, base_color=base_color,
            font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base_font=random_base_font, random_highlight_font=random_highlight_font,
            random_base_color=random_base_color, random_highlight_color=random_highlight_color,
            base_bold=base_bold, base_italic=base_italic,
            highlight_bold=highlight_bold, highlight_italic=highlight_italic,
            include_styling=True
        )

        selected_option = st.selectbox("Choose Subtitle Type:", list(variants.keys()))
        srt_text = variants[selected_option]

        with st.expander(f"📄 View Content: {selected_option}", expanded=True):
            st.text_area("SRT Content", srt_text, height=300, label_visibility="collapsed")

        srt_path = video_path + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)

        st.download_button("⬇️ Download Selected SRT", srt_text, file_name="subtitles.srt")

        # 4. Burn Video
        st.divider()
        st.subheader("Step 3: Burn Subtitles to Video")

        if st.button("🔥 Create Final Video"):
            output_video_path = os.path.join(output_dir, "final_video.mov")
            dialogue_srt_path = os.path.join(output_dir, "dialogue.srt")
            karaoke_srt_path = os.path.join(output_dir, "lip_sync.srt")

            with st.spinner("Burning subtitles & Saving to Folder..."):
                try:
                    with open(dialogue_srt_path, "w", encoding="utf-8") as f:
                        f.write(variants["Standard (Romanized)"])
                    with open(karaoke_srt_path, "w", encoding="utf-8") as f:
                        f.write(variants["Karaoke (Highlighted)"])

                    temp_srt_path = os.path.join(output_dir, "temp_burn.srt")
                    with open(temp_srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_text)

                    subs = pysubs2.load(temp_srt_path, encoding="utf-8")

                    def hex_to_ass(hex_color):
                        hex_color = hex_color.lstrip('#')
                        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                        return f"&H00{b}{g}{r}"

                    ass_base_color = hex_to_ass(base_color)

                    # Border
                    if border_enabled:
                        ass_border_color = hex_to_ass(border_color)
                        outline_width = border_width
                    else:
                        ass_border_color = "&H00000000"
                        outline_width = 0

                    # Shadow
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

                    encoder = get_best_ffmpeg_encoder()

                    def build_cmd(enc):
                        cmd = [
                            FFMPEG_PATH, "-y",
                            "-i", video_path,
                            "-vf", f"ass='{escaped_ass_path}'",
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