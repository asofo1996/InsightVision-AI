import streamlit as st
import os, tempfile, subprocess, cv2, torch
from PIL import Image
import whisper
import yt_dlp
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 설정
st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("🎬 AI 콘텐츠 분석 시스템")
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# Whisper 텍스트 변환
def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

# BLIP 이미지 설명
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

# 프레임 추출
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

# Ollama 분석
def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

# 유튜브 오디오 다운로드
def download_youtube_audio(url):
    audio_file = "youtube_audio.wav"
    if os.path.exists(audio_file):
        os.remove(audio_file)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'youtube_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'prefer_ffmpeg': True,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return audio_file

# 전체 요약
def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions:\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

# 파일 업로드
uploaded_video = st.file_uploader("🎥 영상 파일 업로드", type=["mp4", "mov", "mkv"], key="video")
uploaded_image = st.file_uploader("🖼 이미지 파일 업로드", type=["jpg", "jpeg", "png"], key="image")
uploaded_audio = st.file_uploader("🎤 음성 파일 업로드", type=["wav", "mp3"], key="audio")

# 이미지 처리
if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_container_width=True)
    if st.button("이미지 분석 시작"):
        desc = describe_image_with_blip(image_obj)
        result = analyze_with_ollama(f"Image:\n{desc}\n\n{prompt_text}")
        st.subheader("이미지 분석 결과")
        st.write(result)

# 영상 처리
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

        if st.button("영상 분석 시작"):
            with st.spinner("🎞 프레임 분석 중..."):
                frames = extract_keyframes(video_path)
                descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

            with st.spinner("🔊 Whisper 음성 분석 중..."):
                audio_path = os.path.join(tempfile.gettempdir(), "video_audio.wav")
                subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                transcript = transcribe_audio_whisper(audio_path)

            with st.spinner("🧠 Ollama 종합 분석 중..."):
                final_prompt = summarize_all_inputs(descriptions, transcript, os.path.basename(video_path), prompt_text)
                result = analyze_with_ollama(final_prompt)
                st.subheader("🎬 영상 분석 결과")
                st.write(result)

# 오디오 처리
if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
        if st.button("음성 파일 분석 시작"):
            transcript = transcribe_audio_whisper(audio_path)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.subheader("🎧 음성 분석 결과")
            st.write("전체 텍스트:")
            st.code(transcript)
            st.write("요약 결과:")
            st.write(result)

# 유튜브 분석 모듈
st.markdown("---")
st.subheader("📡 유튜브 또는 로컬 오디오 분석")
col1, col2 = st.columns([1, 3])
with col1:
    mode = st.radio("분석 대상", ["유튜브 링크", "로컬 경로 입력"], horizontal=True)
with col2:
    user_input = st.text_input("유튜브 링크 또는 로컬 경로를 입력하세요")

if st.button("오디오 요약 분석 시작"):
    try:
        audio_path = None
        if mode == "유튜브 링크":
            audio_path = download_youtube_audio(user_input)
        else:
            audio_path = user_input

        transcript = transcribe_audio_whisper(audio_path)
        result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
        st.success("📝 오디오 분석 완료")
        st.write("전체 텍스트:")
        st.code(transcript)
        st.write("요약 결과:")
        st.write(result)
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
