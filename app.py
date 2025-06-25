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
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# --- 핵심 기능 함수 ---
def describe_image_with_blip(pil_image):
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# --- 💥 1. 핵심 변경: 다양한 오디오 확장자를 찾는 함수 ---
def get_latest_audio_file():
    """업로드 디렉토리에서 가장 최근의 오디오 파일을 찾아 경로를 반환 (.wav, .m4a, .webm, .mp3 등)"""
    supported_extensions = ["*.wav", "*.mp3", "*.m4a", "*.webm", "*.mp4"]
    all_audio_files = []
    for ext in supported_extensions:
        all_audio_files.extend(glob.glob(os.path.join(UPLOAD_DIR, ext)))
    
    if not all_audio_files:
        return None
        
    latest_file = max(all_audio_files, key=os.path.getmtime)
    return latest_file

# --- 💥 2. 핵심 변경: 새로 만든 함수를 사용하도록 수정 ---
def safe_transcribe():
    """오디오 파일을 안전하게 텍스트로 변환 (Whisper 사용)"""
    filepath = get_latest_audio_file() # 새로운 함수 호출
    if not filepath or not os.path.exists(filepath):
        st.error(f"❌ 분석할 오디오 파일을 'uploaded' 폴더에서 찾지 못했습니다. 지원 형식: .wav, .mp3, .m4a, .webm 등")
        st.warning(f"'uploaded' 폴더 내용물: {os.listdir(UPLOAD_DIR)}")
        raise FileNotFoundError("분석할 오디오 파일을 찾을 수 없습니다.")

    st.info(f"🧠 분석 대상 오디오 파일: {filepath}")
    
    try:
        model = whisper.load_model("base")
        # Whisper의 transcribe 함수에 파일 경로를 직접 넘겨주면 ffmpeg을 통해 자동으로 처리됩니다.
        result = model.transcribe(filepath, fp16=torch.cuda.is_available(), language='ko')
        return result['text']
    except Exception as e:
        st.error(f"오디오 파일 처리 중 오류 발생: {e}")
        raise RuntimeError(f"Whisper에서 '{filepath}' 파일을 처리하는 데 실패했습니다.") from e

def extract_keyframes(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(original_fps / fps) if original_fps > 0 else 1
    frames = []
    count = 0
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % interval == 0:
            path = os.path.join(UPLOAD_DIR, f"frame_{frame_number}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
            frame_number += 1
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

# --- 💥 3. 핵심 변경: postprocessors 제거 ---
def download_youtube_audio(url):
    """yt-dlp를 사용하여 유튜브 오디오를 원본 형식 그대로 다운로드"""
    out_path_template = os.path.join(UPLOAD_DIR, "youtube_audio.%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path_template,
        # 'postprocessors' 옵션을 완전히 제거하여 WAV 변환 과정을 생략합니다.
        'quiet': False,
        'noplaylist': True
    }
    st.info("yt-dlp로 오디오 다운로드를 시작합니다 (원본 형식 유지)...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.success("오디오 다운로드 완료.")
    st.info(f"다운로드 완료 후 'uploaded' 폴더 내용: {os.listdir(UPLOAD_DIR)}")


# --- Streamlit UI 구성 (이하 코드는 이전과 거의 동일) ---
st.subheader("파일 직접 업로드하여 분석")
uploaded_video = st.file_uploader("영상 업로드", type=["mp4", "mov", "mkv"])
# (이하 이미지, 영상, 음성 업로드 및 분석 UI는 생략)

st.markdown("---")
st.subheader("유튜브 링크 또는 로컬 오디오 파일 경로로 분석")
mode = st.radio("방식 선택", ["유튜브 링크", "로컬 파일"], horizontal=True)
user_input = st.text_input("링크 또는 전체 파일 경로 입력")

if st.button("오디오 요약 분석 시작"):
    try:
        with st.spinner("오디오 처리 및 분석을 시작합니다..."):
            if mode == "유튜브 링크":
                if not user_input.startswith("http"):
                    st.error("❌ 유효한 유튜브 링크를 입력해주세요.")
                else:
                    download_youtube_audio(user_input)
            else: # 로컬 파일 모드
                if not os.path.exists(user_input):
                    raise FileNotFoundError(f"❌ 해당 경로에 파일이 없습니다: {user_input}")
                dst = os.path.join(UPLOAD_DIR, os.path.basename(user_input))
                shutil.copyfile(user_input, dst)
                st.info(f"로컬 파일을 'uploaded' 폴더로 복사했습니다: {dst}")

            transcript = safe_transcribe()
            final_prompt = f"Analyze the following transcript:\n{transcript}\n\nUser's Request:\n{prompt_text}"
            
            st.info("LLM 모델로 종합 분석을 시작합니다...")
            result = analyze_with_ollama(final_prompt)
            
            st.success("✅ 분석 완료")
            st.subheader("🎧 오디오 분석 결과")
            st.code(transcript)
            st.write(result)

    except Exception as e:
        st.error(f"최상위 오류 발생: {e}")
        st.exception(e)

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
