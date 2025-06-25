import streamlit as st
import os, tempfile, shutil, subprocess
import torch
import whisper
import yt_dlp
from transformers import pipeline

st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("AI 분석 시스템")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def download_youtube_audio(url):
    temp_dir = tempfile.mkdtemp()
    mp3_path = os.path.join(temp_dir, "youtube_audio.mp3")
    wav_path = os.path.join(temp_dir, "youtube_audio.wav")

    if os.path.exists(mp3_path): os.remove(mp3_path)
    if os.path.exists(wav_path): os.remove(wav_path)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': mp3_path,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "16000", wav_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return wav_path

def transcribe_audio(path):
    model = load_whisper_model()
    result = model.transcribe(path)
    return result["text"]

def summarize_text(text):
    summarizer = load_summarizer()
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def analyze_audio(input_audio_path):
    full_text = transcribe_audio(input_audio_path)
    summary = summarize_text(full_text)
    return full_text, summary

# --- 인터페이스 구성 ---
st.subheader("분석 해석 (선택)")
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

st.divider()
st.subheader("유튜브 또는 오디오 파일 업로드")

mode = st.radio("입력 방식", ["유튜브 링크", "로컬 음성 파일"])

if mode == "유튜브 링크":
    url = st.text_input("YouTube 링크")
    if st.button("오디오 요약 분석 시작") and url:
        with st.spinner("🔄 유튜브에서 오디오 추출 및 분석 중..."):
            try:
                audio_path = download_youtube_audio(url)
                full, summ = analyze_audio(audio_path)
                st.success("✅ 분석 완료")
                st.subheader("전체 녹취본 텍스트")
                st.text_area("📝 전체 텍스트", full, height=300)
                st.subheader("요약 결과")
                st.info(summ)
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

elif mode == "로컬 음성 파일":
    uploaded_file = st.file_uploader("오디오 파일 업로드 (mp3 또는 wav)", type=["mp3", "wav"])
    if st.button("오디오 요약 분석 시작") and uploaded_file:
        with st.spinner("📂 오디오 분석 중..."):
            try:
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                full, summ = analyze_audio(file_path)
                st.success("✅ 분석 완료")
                st.subheader("전체 녹취본 텍스트")
                st.text_area("📝 전체 텍스트", full, height=300)
                st.subheader("요약 결과")
                st.info(summ)
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
