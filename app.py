import streamlit as st
import whisper
import os
import shutil
import subprocess
import pysubs2
from tempfile import NamedTemporaryFile
from utils import generate_srt_content, AVAILABLE_FONTS

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")

# Ensure tools dir exists in path
os.environ["PATH"] = f"{TOOLS_DIR}:{os.environ.get('PATH', '')}"

# Robust FFmpeg finding
SYSTEM_FFMPEG = shutil.which("ffmpeg")
LOCAL_FFMPEG = os.path.join(TOOLS_DIR, "ffmpeg")

# Prefer system ffmpeg if available, otherwise local
if SYSTEM_FFMPEG:
    FFMPEG_PATH = SYSTEM_FFMPEG
elif os.path.exists(LOCAL_FFMPEG):
    FFMPEG_PATH = LOCAL_FFMPEG
else:
    FFMPEG_PATH = "ffmpeg" # Fallback hope


@st.cache_resource
def load_whisper_model(model_name="base"):
    return whisper.load_model(model_name)

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
    
    st.markdown("**Randomization (Experimental)**")
    random_base = st.checkbox("Randomize Base Font (Chaotic!)")
    random_highlight = st.checkbox("Randomize Highlight Font (Fun!)")

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
        with st.spinner(f"Transcribing with {model_size} model... (This may take a moment)"):
            try:
                # Extract Audio first to be robust (uses local ffmpeg)
                temp_audio = video_path + ".wav"
                subprocess.run([FFMPEG_PATH, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", temp_audio], check=True, capture_output=True)

                model = load_whisper_model(model_size)
                result = model.transcribe(temp_audio, word_timestamps=True, fp16=False)
                st.session_state['transcription_result'] = result
                st.success("Transcription Complete!")
                
                # Cleanup temp audio
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
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
            random_base=random_base, random_highlight=random_highlight
        )

        # Variant 2: Standard (Romanized)
        variants["Standard (Romanized)"] = generate_srt_content(
            segments, use_roman=True, use_karaoke=False,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base=random_base, random_highlight=random_highlight
        )
        
        # Variant 3: Karaoke (Highlighted)
        variants["Karaoke (Highlighted)"] = generate_srt_content(
            segments, use_roman=True, use_karaoke=True,
            highlight_color=highlight_color, base_color=base_color,
            max_words_per_line=max_words, font_size=font_size,
            base_font=base_font, highlight_font=highlight_font,
            random_base=random_base, random_highlight=random_highlight
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
            output_video_path = video_path.replace(".mov", "_subbed.mov")
            
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
                    cmd = [
                        FFMPEG_PATH, "-y", 
                        "-i", video_path, 
                        "-vf", f"ass='{ass_path}'",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
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
