import streamlit as st
import os, tempfile, cv2, torch, subprocess
from PIL import Image
import whisper
import yt_dlp
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI 연산 및 녹음 방송 분석", layout="wide")
st.title("🎬 AI 연산 및 녹음 방송 분석 시스템")

prompt_text = st.text_area("호출할 방식을 입력해주세요", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def describe_image_with_blip(pil_image):
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def extract_keyframes(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(original_fps * fps)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % interval == 0:
            path = os.path.join(tempfile.gettempdir(), f"frame_{count}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
        count += 1
    cap.release()
    return frames

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

def download_youtube_audio(url):
    base_name = "youtube_audio"
    out_path = base_name
    
    if os.path.exists(out_path + ".wav"):
        os.remove(out_path + ".wav")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'prefer_ffmpeg': True,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    actual_file = out_path + ".wav"
    return actual_file

def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions (1s):\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

uploaded_video = st.file_uploader("영상 파일 업로드", type=["mp4", "mov", "mkv"], key="video")
uploaded_image = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"], key="image")
uploaded_audio = st.file_uploader("음성 파일 업로드", type=["wav", "mp3"], key="audio")

if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_container_width=True)
    if st.button("이미지 분석 시작"):
        desc = describe_image_with_blip(image_obj)
        result = analyze_with_ollama(f"Image:\n{desc}\n\n{prompt_text}")
        st.subheader(":mag: 이미지 분석 결과")
        st.write(result)

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

        if st.button("영상 분석 시작"):
            with st.spinner(":scissors: 프레임 분석 중..."):
                frames = extract_keyframes(video_path)
                descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

            with st.spinner(":sound: Whisper를 통한 음성 분석 중..."):
                audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                transcript = transcribe_audio_whisper(audio_path)

            with st.spinner(":robot_face: Ollama 분석 중..."):
                final_prompt = summarize_all_inputs(descriptions, transcript, os.path.basename(video_path), prompt_text)
                result = analyze_with_ollama(final_prompt)
                st.subheader(":chart_with_upwards_trend: 영상 분석 결과")
                st.write(result)

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
        if st.button("음성 파일 분석 시작"):
            transcript = transcribe_audio_whisper(audio_path)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.subheader(":studio_microphone: 음성 분석 결과")
            st.write(":page_facing_up: 전체 텍스트:")
            st.code(transcript)
            st.write(":bulb: 요약 결과:")
            st.write(result)

st.markdown("---")
st.subheader(":globe_with_meridians: 유튜브 링크 또는 로컴 음성 분석")
col1, col2 = st.columns([1, 3])
with col1:
    mode = st.radio(":round_pushpin: 방식", ["유튜브 링크", "로컴 음성 파일"], horizontal=True)
with col2:
    user_input = st.text_input("유튜브 URL" if mode == "유튜브 링크" else "음성 파일 경로")

if st.button(":arrow_forward: 음성 분석 시작"):
    try:
        audio_path = None
        if mode == "유튜브 링크":
            audio_path = download_youtube_audio(user_input)
        else:
            audio_path = user_input

        transcript = transcribe_audio_whisper(audio_path)
        result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
        st.success(":white_check_mark: 음성 분석 완료")
        st.write(":page_facing_up: 전체 텍스트:")
        st.code(transcript)
        st.write(":bulb: 요약 결과:")
        st.write(result)
    except Exception as e:
        st.error(f":x: 오류 발생: {str(e)}")

st.caption("© 2025 시온링트 | 개발자 홍석표")
