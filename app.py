import streamlit as st
import os, tempfile, cv2, torch, subprocess
from PIL import Image
import whisper
import yt_dlp
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI 타입 및 컨텍스 분석", layout="wide")
st.title("텍스트 + 영상 + 음성 디템 자본 분석")
prompt_text = st.text_area("분석 요청 (Prompt)", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# Whisper 터뷰트
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

def transcribe_audio_whisper(audio_path):
    model = load_whisper()
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

# BLIP 터뷰트
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

# Ollama 텍스트 분석
@st.cache_resource
def load_ollama():
    return Ollama(model="llama3")

def analyze_with_ollama(prompt):
    template = PromptTemplate.from_template("""{prompt_text}""")
    chain = LLMChain(prompt=template, llm=load_ollama())
    return chain.run(prompt_text=prompt)

# 프레임 찾기
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

# 요약 포맷
def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions (1s):\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

# 영상 업로드 구성
uploaded_video = st.file_uploader("영상 파일 업로드", type=["mp4", "mov", "mkv"])
uploaded_image = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("음성 파일 업로드", type=["wav", "mp3"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)
        if st.button("영상 분석 시작"):
            with st.spinner("프레임 분석 중..."):
                frames = extract_keyframes(video_path)
                descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

            with st.spinner("음성 분석 중..."):
                audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                transcript = transcribe_audio_whisper(audio_path)

            with st.spinner("AI 분석 중..."):
                prompt = summarize_all_inputs(descriptions, transcript, os.path.basename(video_path), prompt_text)
                result = analyze_with_ollama(prompt)
                st.subheader("영상 분석 결과")
                st.write("전체 텍스트:")
                st.code(transcript)
                st.write("요약 결과:")
                st.write(result)

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, use_container_width=True)
    if st.button("이미지 분석 시작"):
        description = describe_image_with_blip(image)
        result = analyze_with_ollama(f"Image:\n{description}\n\n{prompt_text}")
        st.subheader("이미지 분석 결과")
        st.write(result)

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
        if st.button("음성 분석 시작"):
            transcript = transcribe_audio_whisper(audio_path)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.subheader("음성 분석 결과")
            st.write("전체 텍스트:")
            st.code(transcript)
            st.write("요약 결과:")
            st.write(result)

# 유튜브 다운로드
st.markdown("---")
st.subheader("유튜브 링크 또는 로컬 오디오 분석")
mode = st.radio("분석 방식 선택", ["유튜브 링크", "로컬 경로"], horizontal=True)
user_input = st.text_input("유튜브 링크 또는 파일 경로 입력")

if st.button("오디오 요약 분석 시작"):
    if mode == "유튜브 링크":
        try:
            audio_wav = os.path.join(tempfile.gettempdir(), "youtube_audio.wav")
            if os.path.exists(audio_wav):
                os.remove(audio_wav)
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': audio_wav.replace(".wav", ""),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav'
                }],
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([user_input])
            transcript = transcribe_audio_whisper(audio_wav)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.success("유튜브 분석 완료")
            st.code(transcript)
            st.write(result)
        except Exception as e:
            st.error(str(e))
    else:
        try:
            transcript = transcribe_audio_whisper(user_input)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.success("로컬 분석 완료")
            st.code(transcript)
            st.write(result)
        except Exception as e:
            st.error(str(e))

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
