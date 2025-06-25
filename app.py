import os
import tempfile
import streamlit as st
from PIL import Image
import whisper
import torch
import yt_dlp
import ffmpeg
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydub import AudioSegment

# ---- SETUP ----
st.set_page_config(page_title="AI 분석 시스템", layout="wide")
st.title("AI 분석 시스템")

ASSEMBLY_AI_KEY = st.secrets.get("ASSEMBLYAI_API_KEY", "")  # 향후 사용 예정

# ---- LOAD BLIP ----
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# ---- IMAGE DESCRIPTION ----
def describe_image(image):
    processor, model = load_blip()
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# ---- AUDIO TRANSCRIPTION ----
def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

# ---- OLLAMA ----
def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

# ---- DOWNLOAD YOUTUBE ----
def download_youtube_audio(youtube_url):
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "downloaded_audio.mp3")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return audio_path

# ---- AUDIO NORMALIZATION ----
def convert_audio_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
    return wav_path

# ---- UI: 분석 프롬프트 ----
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# ---- 유튜브 or 파일 ----
st.subheader("🎧 오디오 요약 분석")
option = st.radio("", ["유튜브 링크", "로컬 음성 파일"])
youtube_url = ""
local_audio = None

if option == "유튜브 링크":
    youtube_url = st.text_input("유튜브 링크를 입력하세요")
else:
    local_audio = st.file_uploader("로컬 음성/영상 파일 업로드 (mp3/mp4)", type=["mp3", "mp4"])

if st.button("🧠 오디오 요약 분석 시작"):
    with st.spinner("🔊 오디오 다운로드 및 처리 중..."):
        try:
            audio_path = ""
            if youtube_url:
                mp3_path = download_youtube_audio(youtube_url)
                audio_path = convert_audio_to_wav(mp3_path)
            elif local_audio:
                suffix = ".mp4" if local_audio.name.endswith("mp4") else ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(local_audio.read())
                    tmp_path = tmp.name
                audio_path = convert_audio_to_wav(tmp_path) if suffix == ".mp3" else tmp_path
            else:
                st.warning("파일 또는 링크를 올바르게 입력하세요.")
                st.stop()

            st.info("🔎 Whisper로 전체 텍스트 변환 중...")
            transcript = transcribe_audio_whisper(audio_path)
            st.text_area("전체 텍스트", transcript, height=200)

            st.info("✍️ 요약 및 인사이트 분석 중...")
            result = analyze_with_ollama(f"Please summarize and analyze the following transcript:\n{transcript}\n\n{prompt_text}")
            st.subheader("요약 및 분석 결과")
            st.write(result)

        except Exception as e:
            st.error(f"오류 발생: {e}")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
