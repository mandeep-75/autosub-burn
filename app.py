import streamlit as st
import whisper
import os
import subprocess
from tempfile import NamedTemporaryFile
from utils import generate_srt_content, AVAILABLE_FONTS

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
FFMPEG_PATH = os.path.join(TOOLS_DIR, "ffmpeg")

# Add local FFmpeg to PATH so Whisper can use it
os.environ["PATH"] = f"{TOOLS_DIR}:{os.environ.get('PATH', '')}"

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
    model_size = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large-v3", "turbo"], index=4)
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
                # Style logic
                def hex_to_ass(hex_color):
                    hex_color = hex_color.lstrip('#')
                    return f"&H{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}"
                
                ass_base = hex_to_ass(base_color)
                # Resolve technical font name for the global style
                tech_base_font = AVAILABLE_FONTS.get(base_font, 'Arial')
                
                style = f"FontName={tech_base_font},FontSize={font_size},PrimaryColour={ass_base},OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=50,Alignment=2"
                
                filter_str = f"subtitles=filename='{srt_path}':force_style='{style}'"
                
                cmd = [
                    FFMPEG_PATH, "-y", 
                    "-i", video_path, 
                    "-vf", filter_str,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-c:a", "copy",
                    output_video_path
                ]
                
                try:
                    # Run and capture output
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    st.success("Video Created Successfully!")
                    st.video(output_video_path)
                    
                    with open(output_video_path, "rb") as v_file:
                        st.download_button("⬇️ Download Final Video", v_file, file_name="final_video.mov")
                        
                except subprocess.CalledProcessError as e:
                    st.error("FFmpeg Error: " + e.stderr.decode())
