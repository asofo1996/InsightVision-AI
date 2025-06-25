# ✅ 최종 통합 버전: 영상/음성/YouTube 링크 업로드 → Whisper + BLIP + 요약 분석 포함

import streamlit as st
import os, tempfile, subprocess, torch
from PIL import Image
import whisper
import yt_dlp
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("AI 분석 시스템")

# Whisper 및 요약기 모델 로딩
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# YouTube 다운로드
def download_youtube_audio(url):
    temp_dir = tempfile.mkdtemp()
    mp3_path = os.path.join(temp_dir, "youtube_audio.mp3")
    wav_path = os.path.join(temp_dir, "youtube_audio.wav")
    for path in [mp3_path, wav_path]:
        if os.path.exists(path): os.remove(path)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': mp3_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "16000", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

# 영상 프레임 추출
def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(original_fps * fps)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            temp_file = os.path.join(tempfile.gettempdir(), f"frame_{count}.jpg")
            cv2.imwrite(temp_file, frame)
            frames.append(temp_file)
        count += 1
    cap.release()
    return frames

# 이미지 설명
def describe_image(pil_image):
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Whisper로 음성 텍스트 변환
def transcribe_audio(path):
    model = load_whisper_model()
    result = model.transcribe(path)
    return result['text']

# 요약 생성
def summarize_text(text):
    summarizer = load_summarizer()
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

# Ollama 분석 (옵션)
def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

# 분석 지시문
prompt_text = st.text_area("분석 프롬프트 (선택)", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# 입력 방식 선택
st.subheader("🎬 영상 / 음성 / YouTube 분석")
mode = st.radio("입력 방식", ["유튜브 링크", "로컬 음성 파일", "로컬 영상 파일"])

if mode == "유튜브 링크":
    url = st.text_input("유튜브 링크")
    if st.button("YouTube 분석 시작") and url:
        with st.spinner("YouTube 오디오 다운로드 및 분석 중..."):
            try:
                audio_path = download_youtube_audio(url)
                transcript = transcribe_audio(audio_path)
                summary = summarize_text(transcript)
                st.success("✅ 분석 완료")
                st.text_area("전체 텍스트", transcript, height=250)
                st.subheader("요약 결과")
                st.info(summary)
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

elif mode == "로컬 음성 파일":
    audio_file = st.file_uploader("음성(mp3/wav) 업로드", type=["mp3", "wav"])
    if st.button("음성 분석 시작") and audio_file:
        with st.spinner("음성 파일 분석 중..."):
            try:
                temp_audio = os.path.join(tempfile.gettempdir(), audio_file.name)
                with open(temp_audio, "wb") as f:
                    f.write(audio_file.read())
                transcript = transcribe_audio(temp_audio)
                summary = summarize_text(transcript)
                st.success("✅ 분석 완료")
                st.text_area("전체 텍스트", transcript, height=250)
                st.subheader("요약 결과")
                st.info(summary)
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

elif mode == "로컬 영상 파일":
    video_file = st.file_uploader("영상(mp4) 업로드", type=["mp4"])
    if st.button("영상 분석 시작") and video_file:
        with st.spinner("영상 분석 중 (프레임 + 음성)..."):
            try:
                temp_video = os.path.join(tempfile.gettempdir(), video_file.name)
                with open(temp_video, "wb") as f:
                    f.write(video_file.read())

                # 프레임 분석
                frames = extract_frames(temp_video)
                descs = [describe_image(Image.open(p)) for p in frames[:5]]
                st.subheader("영상 프레임 설명 요약")
                st.markdown("\n".join([f"- {d}" for d in descs]))

                # 음성 추출
                audio_path = os.path.join(tempfile.gettempdir(), "video_audio.wav")
                subprocess.run(["ffmpeg", "-y", "-i", temp_video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                transcript = transcribe_audio(audio_path)
                summary = summarize_text(transcript)
                st.subheader("음성 텍스트")
                st.text_area("전체 텍스트", transcript, height=250)
                st.subheader("요약 결과")
                st.info(summary)
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
