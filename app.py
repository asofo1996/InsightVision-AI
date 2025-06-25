import streamlit as st
import os
import tempfile
import subprocess
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import whisper
import yt_dlp

# Streamlit 설정
st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("AI 분석 시스템")

# 입력 프롬프트
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# 모델 캐싱
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

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

def convert_to_wav(input_path, output_path):
    command = ["ffmpeg", "-y", "-i", input_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]

def download_youtube_audio(url, output_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path),
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def process_audio_source(uploaded_file=None, youtube_link=None):
    temp_dir = tempfile.gettempdir()
    input_path = Path(temp_dir) / "youtube_audio.mp3"
    wav_path = Path(temp_dir) / "converted_audio.wav"

    # 파일 저장 또는 다운로드
    if youtube_link:
        download_youtube_audio(youtube_link, input_path)
    elif uploaded_file:
        input_path.write_bytes(uploaded_file.read())
    else:
        raise ValueError("No valid audio source")

    convert_to_wav(input_path, wav_path)
    transcript = transcribe_audio(str(wav_path))
    summary = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
    return transcript, summary

# 선택 탭
mode = st.radio("분석 방식 선택", ["유튜브 링크", "영상/음성 업로드", "이미지 업로드"])

if mode == "유튜브 링크":
    yt_url = st.text_input("유튜브 링크를 입력하세요")
    if st.button("오디오 요약 분석 시작"):
        try:
            with st.spinner("유튜브 음성 분석 중..."):
                transcript, summary = process_audio_source(youtube_link=yt_url)
            st.success("✅ 분석 완료")
            st.subheader("🎧 전체 텍스트")
            st.write(transcript)
            st.subheader("🧠 요약 결과")
            st.write(summary)
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")

elif mode == "영상/음성 업로드":
    uploaded_audio = st.file_uploader("🎧 영상 또는 음성 파일 업로드", type=["mp3", "wav", "mp4"])
    if uploaded_audio and st.button("업로드 분석 시작"):
        try:
            with st.spinner("업로드한 파일 분석 중..."):
                transcript, summary = process_audio_source(uploaded_file=uploaded_audio)
            st.success("✅ 분석 완료")
            st.subheader("🎧 전체 텍스트")
            st.write(transcript)
            st.subheader("🧠 요약 결과")
            st.write(summary)
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")

elif mode == "이미지 업로드":
    uploaded_image = st.file_uploader("🖼️ 이미지 파일 업로드", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("이미지 설명 및 분석 시작"):
        with st.spinner("이미지 설명 중..."):
            pil = Image.open(uploaded_image).convert("RGB")
            description = describe_image_with_blip(pil)
        with st.spinner("설명 기반 분석 중..."):
            result = analyze_with_ollama(f"Image Description:\n{description}\n\n{prompt_text}")
        st.success("✅ 분석 완료")
        st.subheader("📄 분석 결과")
        st.write(result)

# 하단
st.caption("© 2025 시온마케팅 | 개발자 홍석표")
