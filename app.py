import streamlit as st
import os
import shutil
import subprocess
import pysubs2
import time
from tempfile import NamedTemporaryFile
from utils import AVAILABLE_FONTS, generate_srt_content, parse_whisper_json, parse_srt_to_segments, merge_segments

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
WHISPER_CPP_DIR = os.path.join(TOOLS_DIR, "whisper.cpp")
WHISPER_CLI = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")
MODELS_DIR = os.path.join(WHISPER_CPP_DIR, "models")

# Ensure tools dir exists in path
os.environ["PATH"] = f"{TOOLS_DIR}:{os.environ.get('PATH', '')}"

# Robust FFmpeg finding
SYSTEM_FFMPEG = shutil.which("ffmpeg")
LOCAL_FFMPEG = os.path.join(TOOLS_DIR, "ffmpeg")

# Prefer local ffmpeg if it exists (it's often more capable), then system
if os.path.exists(LOCAL_FFMPEG):
    FFMPEG_PATH = LOCAL_FFMPEG
elif SYSTEM_FFMPEG:
    FFMPEG_PATH = SYSTEM_FFMPEG
else:
    FFMPEG_PATH = "ffmpeg" # Fallback hope

WHISPER_MODELS = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin"
}

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
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return model_path
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return None

# --- MAIN APP ---
st.set_page_config(page_title="Subtitle Refiner Bot", page_icon="🎬")

st.title("🎬 AI Subtitle Generator & Refiner")
st.markdown("Automated captions with **Roman Hindi/Punjabi** transliteration and **Karaoke Highlights**.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    model_size = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large", "large-v3", "large-v3-turbo"], index=1)
    st.divider()
    
    st.subheader("📏 Line & Font")
    max_words = st.slider("Max Words per Line", 3, 20, 8)
    font_size = st.slider("Font Size (px)", 10, 100, 16)
    
    st.divider()
    st.subheader("🎨 Colors")
    highlight_color = st.color_picker("Karaoke Highlight Color", "#FFFF00")
    base_color = st.color_picker("Base Text Color", "#FFFFFF")
    
    st.divider()
    st.subheader("🔤 Fonts")
    font_keys = list(AVAILABLE_FONTS.keys())
    base_font = st.selectbox("Base Font", font_keys, index=0)
    highlight_font = st.selectbox("Highlight Font", font_keys, index=1)
    
    st.divider()
    st.subheader("📍 Position")
    y_padding = st.slider("Vertical Padding (Y)", 0, 1000, 50, help="Distance from bottom edge")
    alignment_option = st.selectbox("Alignment", ["Center", "Left", "Right"], index=0)
    x_padding = st.slider("Horizontal Padding (X)", 0, 500, 20, help="Distance from left/right edge")
    
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

# 1. Upload Video
# 1. Upload Video
uploaded_file = st.file_uploader("Upload a Video (MOV, MP4)", type=["mov", "mp4", "mkv"])

if uploaded_file is not None:
    # Retrieve or create temp file
    # We check if the filename matches what we have in state to avoid re-saving on every button click
    if 'last_uploaded_filename' not in st.session_state or st.session_state.get('last_uploaded_filename') != uploaded_file.name:
        # New file detected
        suffix = os.path.splitext(uploaded_file.name)[1]
        if not suffix: suffix = ".mov"
        
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            temp_video.write(uploaded_file.read())
            st.session_state['temp_video_path'] = temp_video.name
            st.session_state['last_uploaded_filename'] = uploaded_file.name
            
        # Reset transcription state for new file
        if 'transcription_result' in st.session_state:
            del st.session_state['transcription_result']

    video_path = st.session_state['temp_video_path']
    st.video(video_path)

    # 2. Transcribe
    if st.button("✨ Step 1: Generate Subtitles", type="primary"):
        with st.spinner(f"Transcribing with {model_size} model... (Using local whisper.cpp)"):
            try:
                # 1. Ensure binary exists
                if not os.path.exists(WHISPER_CLI):
                    st.error(f"❌ whisper-cli not found at {WHISPER_CLI}. Please build it.")
                    st.stop()
                
                # 2. Ensure model exists
                model_path = ensure_model_exists(model_size)
                if not model_path:
                    st.stop()

                # 3. Extract Audio (16kHz mono WAV is required by whisper.cpp)
                temp_audio = video_path + ".wav"
                subprocess.run([FFMPEG_PATH, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", temp_audio], check=True, capture_output=True)

                # 4. Transcribe using whisper-cli
                output_base = video_path + "_out"
                # Use char limit 1 to get word-level segments (Lip Sync)
                char_limit = 1
                
                cmd = [
                    WHISPER_CLI,
                    "-m", model_path,
                    "-f", temp_audio,
                    "-ojf", # Full JSON for word details
                    "-osrt", # SRT for perfect segment sync
                    "-of", output_base,
                    "-l", "auto",
                    "-t", "8",
                    "-bs", "5",
                    "-ml", str(char_limit),
                    "-sow"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"Transcription failed: {result.stderr}")
                    st.stop()

                # 5. Parse the resulting files
                srt_out_path = output_base + ".srt"
                json_out_path = output_base + ".json"

                if os.path.exists(srt_out_path):
                    # Use SRT as the primary source for segment timing to fix sync
                    segments = parse_srt_to_segments(srt_out_path)
                elif os.path.exists(json_out_path):
                    import json
                    with open(json_out_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    segments = parse_whisper_json(data)
                else:
                    st.error("❌ Failed to generate subtitle output.")
                    st.stop()
                
                # Merge individual word segments back into lines based on max_words setting
                segments = merge_segments(segments, max_words=max_words)
                
                st.session_state['transcription_result'] = {'segments': segments}
                st.success("Transcription Complete!")
                
                # Cleanup
                if os.path.exists(temp_audio): os.remove(temp_audio)
                if os.path.exists(srt_out_path): os.remove(srt_out_path)
                if os.path.exists(json_out_path): os.remove(json_out_path)
                
            except Exception as e:
                st.error(f"Transcription Failed: {e}")

    # 3. Refine & Preview
    if 'transcription_result' in st.session_state:
        st.divider()
        st.subheader("Step 2: Select Subtitle Style")
        
        # Show detected language if available
        if 'language' in st.session_state['transcription_result']:
            lang = st.session_state['transcription_result']['language']
            st.info(f"🌍 Auto-Detected Language: **{lang.upper()}**")
        
        # Generate Variants on the fly based on current settings
        segments = st.session_state['transcription_result']['segments']
        
        variants = {}
        
        # Variant 1: Original (Auto-Detected) - Priority
        variants["Original (Auto-Detected)"] = generate_srt_content(
            segments, use_roman=False, use_karaoke=False,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base_font=random_base_font, random_highlight_font=random_highlight_font,
            random_base_color=random_base_color, random_highlight_color=random_highlight_color
        )

        # Variant 2: Standard (Romanized)
        variants["Standard (Romanized)"] = generate_srt_content(
            segments, use_roman=True, use_karaoke=False,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base_font=random_base_font, random_highlight_font=random_highlight_font,
            random_base_color=random_base_color, random_highlight_color=random_highlight_color
        )
        
        # Variant 3: Karaoke (Highlighted)
        variants["Karaoke (Highlighted)"] = generate_srt_content(
            segments, use_roman=True, use_karaoke=True,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base_font=random_base_font, random_highlight_font=random_highlight_font,
            random_base_color=random_base_color, random_highlight_color=random_highlight_color
        )
        
        # User Selection
        selected_option = st.selectbox("Choose Subtitle Type:", list(variants.keys()))
        srt_text = variants[selected_option]

        # Display SRT Preview
        with st.expander(f"📄 View Content: {selected_option}", expanded=True):
            st.text_area("SRT Content", srt_text, height=300, label_visibility="collapsed")

        # Save SRT to buttons
        srt_path = video_path + ".srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
            
        st.download_button("⬇️ Download Selected SRT", srt_text, file_name="subtitles.srt")

        # 4. Burn Video
        st.divider()
        st.subheader("Step 3: Burn Subtitles to Video")
        
        if st.button("🔥 Create Final Video"):
            base, ext = os.path.splitext(video_path)
            output_video_path = f"{base}_subbed{ext}"
            
            with st.spinner("Burning subtitles using Hardware Acceleration..."):
                try:
                    # 1. Convert SRT to ASS using pysubs2 for better styling control
                    subs = pysubs2.load(srt_path, encoding="utf-8")
                    
                    # 2. Define the style
                    def hex_to_ass(hex_color):
                        # Streamlit returns #RRGGBB, ASS wants &H00BBGGRR (ABGR)
                        hex_color = hex_color.lstrip('#')
                        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                        return f"&H00{b}{g}{r}"

                    ass_base_color = hex_to_ass(base_color)
                    tech_base_font = AVAILABLE_FONTS.get(base_font, 'Arial')
                    align_map = {"Center": 2, "Left": 1, "Right": 3}
                    
                    style = pysubs2.SSAStyle(
                        fontname=tech_base_font,
                        fontsize=font_size,
                        primarycolor=ass_base_color,
                        outlinecolor="&H00000000",
                        backcolor="&H00000000",
                        borderstyle=1,
                        outline=2,
                        shadow=0,
                        alignment=align_map[alignment_option],
                        marginv=y_padding,
                        marginl=x_padding,
                        marginr=x_padding
                    )
                    
                    # Apply style to all events? No, just set as Default
                    subs.styles["Default"] = style
                    
                    # Save as ASS
                    ass_path = video_path + ".ass"
                    subs.save(ass_path)

                    # 3. Burn with FFmpeg
                    # Escape the path for the 'ass' filter (standard ffmpeg filter escaping)
                    escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
                    
                    cmd = [
                        FFMPEG_PATH, "-y", 
                        "-i", video_path, 
                        "-vf", f"ass='{escaped_ass_path}'",
                        "-c:v", "h264_videotoolbox", "-b:v", "8M", "-realtime", "1",
                        "-c:a", "copy",
                        output_video_path
                    ]
                    
                    # Run and capture output
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    st.success("Video Created Successfully!")
                    st.video(output_video_path)
                    
                    # 4. Download
                    with open(output_video_path, "rb") as v_file:
                        st.download_button("⬇️ Download Final Video", v_file.read(), file_name="final_video.mov")
                        
                except subprocess.CalledProcessError as e:
                    st.error(f"FFmpeg Error: {e.stderr.decode() if e.stderr else str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
