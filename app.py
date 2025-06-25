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
import yt_dlp
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 페이지 기본 설정
st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("🎬 AI 콘텐츠 분석 시스템")
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# 업로드 디렉토리 설정
UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 모델 로드 (캐시 사용) ---
@st.cache_resource
def load_blip():
    """BLIP 이미지 캡셔닝 모델 로드"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# --- 핵심 기능 함수 ---
def describe_image_with_blip(pil_image):
    """BLIP 모델을 사용하여 이미지 설명 생성"""
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def get_latest_wav_file():
    """업로드 디렉토리에서 가장 최근에 수정된 .wav 파일 경로 반환"""
    wav_files = sorted(
        glob.glob(os.path.join(UPLOAD_DIR, "*.wav")),
        key=os.path.getmtime,
        reverse=True
    )
    return wav_files[0] if wav_files else None

def safe_transcribe():
    """오디오 파일을 안전하게 텍스트로 변환 (Whisper 사용)"""
    filepath = get_latest_wav_file()
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(".wav 파일을 찾을 수 없습니다. 파일이 정상적으로 생성되었는지 확인하세요.")

    st.info(f"🧠 분석 대상 오디오 파일: {filepath}")
    
    # torchaudio로 오디오 로드 및 리샘플링
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        
    audio = waveform.squeeze().numpy()
    
    # Whisper 모델로 텍스트 추출
    model = whisper.load_model("base")
    result = model.transcribe(audio, fp16=torch.cuda.is_available(), language='ko')
    return result['text']

def extract_keyframes(video_path, fps=1):
    """비디오에서 초당 1프레임씩 키프레임 추출"""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(original_fps / fps) if original_fps > 0 else 1
    frames = []
    count = 0
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            path = os.path.join(UPLOAD_DIR, f"frame_{frame_number}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
            frame_number += 1
        count += 1
    cap.release()
    return frames

def analyze_with_ollama(prompt_text):
    """Ollama를 사용하여 프롬프트 기반 분석 실행"""
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

def summarize_all_inputs(frames_desc, transcript, title, prompt):
    """모든 입력(프레임 설명, 대본, 제목, 프롬프트)을 하나의 요약 텍스트로 결합"""
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions:\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

def download_youtube_audio(url):
    """yt-dlp를 사용하여 유튜브 오디오를 다운로드하고 wav로 변환"""
    # 출력 파일 경로 (확장자 제외)
    out_base = os.path.join(UPLOAD_DIR, "youtube_audio")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        # --- 💥 핵심 수정 부분 💥 ---
        # yt-dlp가 후처리 과정에서 올바른 확장자(.wav)를 붙이도록 확장자를 제거합니다.
        'outtmpl': out_base,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# --- Streamlit UI 구성 ---

# 1. 파일 업로드 섹션
st.subheader("파일 직접 업로드하여 분석")
uploaded_video = st.file_uploader("영상 업로드", type=["mp4", "mov", "mkv"])
uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("음성 업로드", type=["wav", "mp3"])

# 이미지 분석
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="업로드 이미지")
    if st.button("이미지 분석 시작"):
        with st.spinner("이미지를 분석 중입니다..."):
            desc = describe_image_with_blip(image)
            final_prompt = f"Analyze the following image description:\n{desc}\n\nUser's Request:\n{prompt_text}"
            result = analyze_with_ollama(final_prompt)
            st.subheader("🖼️ 이미지 분석 결과")
            st.write(result)

# 영상 분석
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    st.video(video_path)
    if st.button("영상 분석 시작"):
        with st.spinner("영상을 분석 중입니다. 키프레임을 추출하고 오디오를 변환합니다..."):
            frames = extract_keyframes(video_path)
            descs = [describe_image_with_blip(Image.open(f)) for f in frames]
            
            # 비디오에서 오디오 추출
            audio_path = os.path.join(UPLOAD_DIR, "extracted_audio.wav")
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-vn", 
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            transcript = safe_transcribe()
            final_prompt = summarize_all_inputs(descs, transcript, os.path.basename(uploaded_video.name), prompt_text)
            
            st.info("LLM 모델로 종합 분석을 시작합니다...")
            result = analyze_with_ollama(final_prompt)
            st.subheader("🎥 영상 분석 결과")
            st.write(result)

# 음성 분석
if uploaded_audio:
    suffix = os.path.splitext(uploaded_audio.name)[1]
    saved_path = os.path.join(UPLOAD_DIR, f"uploaded_audio{suffix}")
    with open(saved_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.audio(saved_path)
    if st.button("음성 분석 시작"):
        with st.spinner("음성을 텍스트로 변환하고 분석 중입니다..."):
            transcript = safe_transcribe()
            final_prompt = f"Analyze the following transcript:\n{transcript}\n\nUser's Request:\n{prompt_text}"
            result = analyze_with_ollama(final_prompt)
            st.subheader("🔊 음성 분석 결과")
            st.code(transcript)
            st.write(result)

st.markdown("---")

# 2. 유튜브 링크 또는 로컬 경로 분석 섹션
st.subheader("유튜브 링크 또는 로컬 오디오 파일 경로로 분석")
mode = st.radio("방식 선택", ["유튜브 링크", "로컬 파일"], horizontal=True)
user_input = st.text_input("링크 또는 전체 파일 경로 입력")

if st.button("오디오 요약 분석 시작"):
    try:
        with st.spinner("오디오를 처리하고 분석을 시작합니다..."):
            if mode == "유튜브 링크":
                if not user_input.startswith("http"):
                    st.error("❌ 유효한 유튜브 링크를 입력해주세요 (예: https://www.youtube.com/watch?v=...)")
                else:
                    download_youtube_audio(user_input)
            else: # 로컬 파일 모드
                if not os.path.exists(user_input):
                    raise FileNotFoundError(f"❌ 해당 경로에 파일이 없습니다: {user_input}")
                dst = os.path.join(UPLOAD_DIR, os.path.basename(user_input))
                shutil.copyfile(user_input, dst)

            # 성공적으로 다운로드 또는 복사 후 텍스트 변환 및 분석
            transcript = safe_transcribe()
            final_prompt = f"Analyze the following transcript:\n{transcript}\n\nUser's Request:\n{prompt_text}"
            result = analyze_with_ollama(final_prompt)
            
            st.success("✅ 분석 완료")
            st.subheader("🎧 오디오 분석 결과")
            st.code(transcript)
            st.write(result)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
