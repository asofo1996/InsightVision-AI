import streamlit as st
import os
import shutil
import glob
import cv2
import tempfile
import subprocess
import torch
import torchaudio
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

    st.info(f"🧠 분석 대상 파일: {filepath}")
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    audio = waveform.squeeze().numpy()
    model = whisper.load_model("base")
    result = model.transcribe(audio, fp16=torch.cuda.is_available(), language='ko')
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

# 업로드
uploaded_video = st.file_uploader("영상 업로드", type=["mp4", "mov", "mkv"])
uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("음성 업로드", type=["wav", "mp3"])

if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_container_width=True)
    if st.button("이미지 분석 시작"):
        with st.spinner("이미지 설명 생성 중..."):
            desc = describe_image_with_blip(image_obj)

        image_name = uploaded_image.name
        refined_prompt = f"""아래는 BLIP 모델이 생성한 이미지 설명입니다:

{desc}

이 이미지는 광고 이미지로 사용될 수 있습니다.

다음 항목을 바탕으로 광고 전문가로서 이미지 분석을 수행해 주세요:
1. 이미지의 파일명: {image_name}
2. 이미지에 사용된 주요 색상, 배경, 텍스트, 구성 요소
3. 이미지 내 문구(텍스트)의 전달력과 브랜드 전달 효과
4. 레이아웃 구성의 시각적 완성도
5. 타겟 소비자층과의 적합성
6. SNS, 디스플레이 광고, 배너 광고 등의 활용 적합성
7. 광고 심사를 통과하기 위한 문구/구성 개선 포인트
8. 전반적인 Tone & Manner에 대한 평가

위 분석 후, 개선 아이디어 3가지를 제안해 주세요.
"""

        with st.spinner("Ollama 광고 분석 중..."):
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
