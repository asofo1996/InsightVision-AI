# app.py
import os
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from PIL import Image
import torch
import whisper
import yt_dlp

from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI 분석 시스템", layout="wide")
st.title("AI 분석 시스템")
st.markdown("---")

# Whisper 모델 로딩
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# BLIP 모델 로딩
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# 오디오 추출 함수
def extract_audio_ffmpeg(video_path, output_wav):
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_wav]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 유튜브 링크 다운로드
def download_audio_from_youtube(url, save_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(save_path),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# 텍스트 요약 프롬프트 생성
def build_prompt(transcript, title):
    return f"""Title: {title}\n\nTranscript:\n{transcript}\n\nPlease analyze the video script above. Provide the following:\n1. Video type\n2. Main audience\n3. Tone\n4. Any issues in copywriting\n5. Suggestions for better ad performance (hook, targeting, CTA)."""

# 텍스트 요약 실행
def run_llm_summary(text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    chain = LLMChain(prompt=template, llm=Ollama(model="llama3"))
    return chain.run(prompt_text=text)

# Whisper 변환

def transcribe_audio_whisper(audio_path):
    model = load_whisper()
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

st.subheader("분석 해석")
custom_prompt = st.text_area("분석 프롬프트 입력", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

st.subheader("영상 또는 음성 업로드")
uploaded_file = st.file_uploader("파일 업로드 (MP4, MP3, WAV)", type=["mp4", "mp3", "wav"])

st.subheader("또는 유튜브 링크 분석")
col1, col2 = st.columns(2)
with col1:
    use_youtube = st.radio("분석 소스 선택", ["유튜브 링크", "로컬 음성파일"], horizontal=True)

youtube_link = st.text_input("유튜브 링크를 입력하세요")

if st.button("오디오 요약 분석 시작"):
    with st.spinner("분석 중입니다..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "input.wav")

            # 유튜브에서 다운로드하거나 파일 저장
            if use_youtube == "유튜브 링크" and youtube_link:
                try:
                    download_audio_from_youtube(youtube_link, os.path.join(tmpdir, "youtube_audio"))
                    audio_path = os.path.join(tmpdir, "youtube_audio.wav")
                except Exception as e:
                    st.error(f"유튜브 다운로드 실패: {e}")
                    st.stop()
            elif uploaded_file is not None:
                tmpfile = os.path.join(tmpdir, uploaded_file.name)
                with open(tmpfile, "wb") as f:
                    f.write(uploaded_file.read())
                if uploaded_file.name.endswith(".wav"):
                    audio_path = tmpfile
                else:
                    audio_path = os.path.join(tmpdir, "converted.wav")
                    extract_audio_ffmpeg(tmpfile, audio_path)
            else:
                st.warning("유튜브 링크 또는 파일을 업로드하세요.")
                st.stop()

            try:
                transcript = transcribe_audio_whisper(audio_path)
                summary = run_llm_summary(build_prompt(transcript, uploaded_file.name if uploaded_file else "YouTube Video"))

                st.success("✅ 분석 완료")
                st.subheader("전체 녹취본")
                st.text_area("Transcript", transcript, height=250)
                st.subheader("요약 결과")
                st.write(summary)
            except Exception as e:
                st.error(f"분석 실패: {e}")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
