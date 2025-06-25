import streamlit as st
import os
import shutil
import glob
import cv2
import tempfile
import subprocess
import torch
import pytesseract
import whisper
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("🎬 AI 콘텐츠 분석 시스템")
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tesseract OCR 경로 설정
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

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

def extract_text_from_image(pil_image):
    return pytesseract.image_to_string(pil_image, lang='kor+eng')

def get_latest_wav_file():
    wav_files = sorted(
        glob.glob(os.path.join(UPLOAD_DIR, "*.wav")),
        key=os.path.getmtime,
        reverse=True
    )
    return wav_files[0] if wav_files else None

def safe_transcribe():
    filepath = get_latest_wav_file()
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(".wav 파일을 찾을 수 없습니다.")
    model = whisper.load_model("base")
    result = model.transcribe(filepath, fp16=torch.cuda.is_available(), language='ko')
    return result['text']

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
            path = os.path.join(UPLOAD_DIR, f"frame_{count}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
        count += 1
    cap.release()
    return frames

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions:\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

uploaded_video = st.file_uploader("영상 업로드", type=["mp4", "mov", "mkv"])
uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("음성 업로드", type=["wav", "mp3"])

if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_container_width=True)
    if st.button("이미지 분석 시작"):
        with st.spinner("이미지 설명 및 텍스트 추출 중..."):
            desc = describe_image_with_blip(image_obj)
            extracted_text = extract_text_from_image(image_obj)

        image_name = uploaded_image.name

        refined_prompt = f"""다음은 광고 이미지 분석을 위한 정보입니다:

[파일명]
{image_name}

[BLIP 이미지 설명]
{desc}

[OCR로 추출된 텍스트]
{extracted_text}

🔍 아래 항목을 반드시 포함해 광고 전문가로서 분석 및 개선안을 제시해 주세요:

1. 파일명으로 추정 가능한 업종, 브랜드, 서비스, 타겟 등
2. 이미지의 색상, 글꼴, 레이아웃 구성, 텍스트 크기/배치의 전략적 의미
3. 주목성과 CTA 효과 (ex. 혜택 받기, 제한 조건, 유도 화살표 등)
4. 톤앤매너 (신뢰, 건강, 활기, 감성 등) 및 감정 유도 요소
5. 광고 심사 규정 위반 가능성 (과장, 비의료인 사용, 표현 등)
6. 시각 흐름(상단 강조 → 하단 클릭 유도 등)의 설계 여부
7. 전체적으로 시청자가 어떤 인식을 하게 되는지 예측

→ 마지막으로, 해당 광고를 보완/개선하기 위한 실질적인 실행 제안을 3가지 해주세요.
"""

        with st.spinner("광고 전문가 관점 분석 중..."):
            result = analyze_with_ollama(refined_prompt)

        st.success("✅ 이미지 광고 분석 완료")
        st.subheader("🧠 분석 결과")
        st.write(result)

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    st.video(video_path)
    if st.button("영상 분석 시작"):
        frames = extract_keyframes(video_path)
        descs = [describe_image_with_blip(Image.open(f)) for f in frames]
        audio_path = os.path.join(UPLOAD_DIR, "extracted_audio.wav")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        transcript = safe_transcribe()
        final_prompt = summarize_all_inputs(descs, transcript, os.path.basename(video_path), prompt_text)
        result = analyze_with_ollama(final_prompt)
        st.subheader("영상 분석 결과")
        st.write(result)

if uploaded_audio:
    suffix = os.path.splitext(uploaded_audio.name)[1]
    saved_path = os.path.join(UPLOAD_DIR, f"uploaded_audio{suffix}")
    with open(saved_path, "wb") as f:
        f.write(uploaded_audio.read())
    if st.button("음성 분석 시작"):
        transcript = safe_transcribe()
        result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
        st.subheader("음성 분석 결과")
        st.code(transcript)
        st.write(result)

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
