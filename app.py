import streamlit as st
import os, tempfile, subprocess, torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import whisper
import yt_dlp
import cv2

st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("AI 분석 시스템")

# Prompt 입력
prompt_text = st.text_area("분석할 내용", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

def extract_audio_ffmpeg(video_path):
    audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps * fps)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            path = os.path.join(tempfile.gettempdir(), f"frame_{count}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
        count += 1
    cap.release()
    return frames

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def describe_image(pil_image):
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

def summarize_all(frames_desc, transcript, title, prompt):
    combined = f"Title: {title}\n\nFrame Descriptions:\n" + "\n".join(frames_desc)
    combined += f"\n\nTranscript:\n{transcript}\n\n{prompt.strip()}"
    return combined

def download_audio_from_youtube(url):
    output_path = os.path.join(tempfile.gettempdir(), "downloaded_audio.wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

st.markdown("### 영상 파일 업로드")
video_path = None
uploaded_video = st.file_uploader("Drag and drop file here", type=["mp4", "mpeg4"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

st.markdown("### 이미지 파일 업로드")
uploaded_image = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])
image_obj = None
if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_column_width=True)

col1, col2 = st.columns(2)

with col1:
    if image_obj:
        if st.button("이미지 분석 시작"):
            desc = describe_image(image_obj)
            result = analyze_with_ollama(f"Image Description:\n{desc}\n\n{prompt_text}")
            st.subheader("분석 결과")
            st.write(result)

with col2:
    if video_path:
        if st.button("영상 분석 시작"):
            st.info("프레임 분석 중...")
            frames = extract_frames(video_path)
            frame_descs = [describe_image(Image.open(f)) for f in frames]

            st.info("Whisper를 통한 음성 텍스트 변환 중...")
            audio_path = extract_audio_ffmpeg(video_path)
            transcript = transcribe_audio_whisper(audio_path)

            st.info("Ollama 종합 분석 중...")
            title = os.path.basename(video_path)
            final_prompt = summarize_all(frame_descs, transcript, title, prompt_text)
            result = analyze_with_ollama(final_prompt)
            st.subheader("분석 결과")
            st.write(result)

st.markdown("### 음성 요약 분석 (유튜브 링크 또는 음성 파일)")
audio_source_type = st.radio("입력 방식 선택", ["유튜브 링크", "로컬 음성 파일 업로드"], horizontal=True)
youtube_url = st.text_input("유튜브 링크") if audio_source_type == "유튜브 링크" else ""
audio_file = st.file_uploader("음성 파일 업로드", type=["mp3", "wav"]) if audio_source_type == "로컬 음성 파일 업로드" else None

if st.button("음성 분석 시작"):
    audio_path = None
    if audio_source_type == "유튜브 링크" and youtube_url:
        audio_path = download_audio_from_youtube(youtube_url)
    elif audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name
    if audio_path:
        st.info("음성 텍스트 추출 중...")
        transcript = transcribe_audio_whisper(audio_path)
        st.info("요약 및 분석 중...")
        result = analyze_with_ollama(f"음성 텍스트:\n{transcript}\n\n{prompt_text}")
        st.subheader("요약 분석 결과")
        st.write(result)
    else:
        st.warning("링크 또는 파일을 확인해 주세요")

if not video_path and not image_obj:
    st.warning("영상 또는 이미지를 업로드해 주세요.")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
