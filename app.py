import streamlit as st
import os
import tempfile
import glob
import cv2
import subprocess
import torch
import whisper
import yt_dlp
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 설정 ---
st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("🎬 AI 콘텐츠 분석 시스템")
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# --- BLIP 모델 로딩 ---
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# --- 이미지 설명 생성 ---
def describe_image_with_blip(pil_image):
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# --- 가장 최신 .wav 파일 찾기 ---
def get_latest_wav_file():
    temp_dir = tempfile.gettempdir()
    wav_files = sorted(
        glob.glob(os.path.join(temp_dir, "*.wav")),
        key=os.path.getmtime,
        reverse=True
    )
    return wav_files[0] if wav_files else None

# --- Whisper 변환 ---
def transcribe_audio_whisper():
    audio_path = get_latest_wav_file()
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(".wav 파일이 없습니다.")
    st.info(f"Whisper 대상: {audio_path}")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

# --- 영상 프레임 추출 ---
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

# --- Ollama 분석 ---
def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

# --- 전체 요약 구성 ---
def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions:\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

# --- 유튜브 오디오 다운로드 ---
def download_youtube_audio(url):
    out_base = os.path.join(tempfile.gettempdir(), "youtube_audio")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_base + ".%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# --- 업로드 영역 ---
uploaded_video = st.file_uploader("영상 업로드", type=["mp4", "mov", "mkv"])
uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("음성 업로드", type=["wav", "mp3"])

# --- 이미지 분석 ---
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="업로드한 이미지")
    if st.button("이미지 분석 시작"):
        desc = describe_image_with_blip(image)
        result = analyze_with_ollama(f"Image:\n{desc}\n\n{prompt_text}")
        st.subheader("이미지 분석 결과")
        st.write(result)

# --- 영상 분석 ---
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    st.video(video_path)
    if st.button("영상 분석 시작"):
        frames = extract_keyframes(video_path)
        descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]
        audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        transcript = transcribe_audio_whisper()
        final_prompt = summarize_all_inputs(descriptions, transcript, os.path.basename(video_path), prompt_text)
        result = analyze_with_ollama(final_prompt)
        st.subheader("영상 분석 결과")
        st.write(result)

# --- 음성 분석 ---
if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
    if st.button("음성 분석 시작"):
        transcript = transcribe_audio_whisper()
        result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
        st.subheader("음성 분석 결과")
        st.code(transcript)
        st.write(result)

# --- 유튜브 분석 ---
st.markdown("---")
st.subheader("유튜브 링크 또는 로컬 오디오 분석")
mode = st.radio("분석 모드 선택", ["유튜브 링크", "로컬 경로"], horizontal=True)
user_input = st.text_input("링크 또는 경로 입력")
if st.button("오디오 분석 시작"):
    try:
        if mode == "유튜브 링크":
            download_youtube_audio(user_input)
        else:
            if not os.path.exists(user_input):
                raise FileNotFoundError("❌ 로컬 파일이 존재하지 않습니다.")
            temp_copy = os.path.join(tempfile.gettempdir(), os.path.basename(user_input))
            with open(user_input, 'rb') as src, open(temp_copy, 'wb') as dst:
                dst.write(src.read())
        transcript = transcribe_audio_whisper()
        result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
        st.success("✅ 분석 완료")
        st.code(transcript)
        st.write(result)
    except Exception as e:
        st.error(str(e))

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
