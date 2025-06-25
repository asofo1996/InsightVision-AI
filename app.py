import streamlit as st
import os
import shutil
import glob
import cv2
import tempfile
import subprocess
import torch
import whisper
import yt_dlp
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 1. 작업 환경 설정 및 폴더 생성 (가장 중요한 부분) ---
# 이 스크립트와 같은 위치에 'uploaded' 폴더를 생성하고 모든 작업의 기준점으로 사용합니다.
UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded")

# 스크립트 시작 시 작업 폴더가 없으면 자동으로 생성
try:
    if not os.path.exists(UPLOAD_DIR):
        st.info(f"작업 폴더({UPLOAD_DIR})가 없어 새로 생성합니다.")
        os.makedirs(UPLOAD_DIR)
except Exception as e:
    st.error(f"작업 폴더 생성 중 치명적인 오류 발생: {e}")
    st.error("스크립트가 파일을 저장할 폴더를 생성할 권한이 있는지 확인해주세요.")
    st.stop()


# --- 2. Streamlit UI 기본 설정 ---
st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("🎬 AI 콘텐츠 분석 시스템")
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")


# --- 3. 핵심 기능 함수 정의 ---

@st.cache_resource
def load_blip():
    """BLIP 이미지 캡셔닝 모델 로드"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def describe_image_with_blip(pil_image):
    """BLIP 모델을 사용하여 이미지 설명 생성"""
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def get_latest_audio_file():
    """'uploaded' 폴더에서 가장 최근에 수정된 오디오 파일을 찾습니다."""
    supported_extensions = ["*.wav", "*.mp3", "*.m4a", "*.webm", "*.mp4", "*.ogg"]
    all_audio_files = []
    for ext in supported_extensions:
        all_audio_files.extend(glob.glob(os.path.join(UPLOAD_DIR, ext)))
    
    if not all_audio_files:
        return None
        
    latest_file = max(all_audio_files, key=os.path.getmtime)
    return latest_file

def safe_transcribe():
    """가장 최신 오디오 파일을 찾아 Whisper로 음성 인식을 수행합니다."""
    filepath = get_latest_audio_file()
    if not filepath:
        st.error(f"❌ 분석할 오디오 파일을 '{UPLOAD_DIR}' 폴더에서 찾지 못했습니다.")
        st.warning(f"현재 폴더 내용물: {os.listdir(UPLOAD_DIR)}")
        raise FileNotFoundError("분석할 오디오 파일이 없습니다.")

    st.info(f"🧠 분석 대상 오디오 파일: {filepath}")
    
    model = whisper.load_model("base")
    result = model.transcribe(filepath, fp16=torch.cuda.is_available(), language='ko')
    return result['text']

def extract_keyframes(video_path, fps=1):
    """비디오에서 키프레임을 추출하여 'uploaded' 폴더에 저장합니다."""
    cap = cv2.VideoCapture(video_path)
    # (이하 함수 내용은 이전과 동일)
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
    """Ollama를 사용하여 프롬프트 기반 분석 실행"""
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

def summarize_all_inputs(frames_desc, transcript, title, prompt):
    """모든 입력을 하나의 요약 텍스트로 결합"""
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions:\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

def download_youtube_audio(url):
    """yt-dlp로 유튜브 오디오를 원본 형식 그대로 'uploaded' 폴더에 다운로드합니다."""
    # 출력 경로를 UPLOAD_DIR로 명확히 지정
    output_template = os.path.join(UPLOAD_DIR, "youtube_audio.%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'noplaylist': True,
        # postprocessors를 제거하여 원본 형식(.m4a 등)으로 다운로드
    }
    
    st.info(f"yt-dlp로 오디오 다운로드를 시작합니다. 저장 경로: {UPLOAD_DIR}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.success("다운로드 완료!")
    st.info(f"현재 '{UPLOAD_DIR}' 폴더 내용: {os.listdir(UPLOAD_DIR)}")


# --- 4. Streamlit UI 상세 구성 ---
st.markdown("---")
st.subheader("파일 직접 업로드 또는 유튜브 링크로 분석")

# 탭 UI로 변경
tab1, tab2 = st.tabs(["파일 업로드", "유튜브 링크 또는 로컬 경로"])

with tab1:
    st.write("#### 영상, 이미지, 음성 파일을 직접 업로드하여 분석합니다.")
    uploaded_video = st.file_uploader("영상 업로드", type=["mp4", "mov", "mkv"])
    uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    uploaded_audio = st.file_uploader("음성 업로드", type=["wav", "mp3", "m4a"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="업로드 이미지")
        if st.button("이미지 분석 시작"):
            # (이하 분석 로직은 이전과 동일)
            with st.spinner("이미지를 분석 중입니다..."):
                desc = describe_image_with_blip(image)
                final_prompt = f"Analyze the following image description:\n{desc}\n\nUser's Request:\n{prompt_text}"
                result = analyze_with_ollama(final_prompt)
                st.subheader("🖼️ 이미지 분석 결과")
                st.write(result)
    
    # (영상 및 음성 업로드 처리 로직은 생략. 필요 시 이전 코드에서 복사하여 추가 가능)

with tab2:
    st.write("#### 유튜브 링크 또는 로컬 컴퓨터의 오디오 파일 경로를 입력하여 분석합니다.")
    mode = st.radio("방식 선택", ["유튜브 링크", "로컬 파일 경로"], horizontal=True, key="tab2_mode")
    user_input = st.text_input("링크 또는 전체 파일 경로 입력", key="tab2_input")

    if st.button("오디오 요약 분석 시작", key="tab2_button"):
        if not user_input:
            st.warning("링크 또는 파일 경로를 입력해주세요.")
        else:
            try:
                with st.spinner("오디오 처리 및 분석을 시작합니다..."):
                    if mode == "유튜브 링크":
                        if not user_input.startswith("http"):
                            st.error("❌ 유효한 유튜브 링크를 입력해주세요.")
                        else:
                            download_youtube_audio(user_input)
                    else: # 로컬 파일 경로 모드
                        if not os.path.exists(user_input):
                            raise FileNotFoundError(f"❌ 해당 경로에 파일이 없습니다: {user_input}")
                        # 파일을 UPLOAD_DIR로 복사
                        shutil.copy(user_input, UPLOAD_DIR)
                        st.info(f"로컬 파일을 '{UPLOAD_DIR}' 폴더로 복사했습니다.")

                    # 공통 분석 로직
                    transcript = safe_transcribe()
                    final_prompt = f"Analyze the following transcript:\n{transcript}\n\nUser's Request:\n{prompt_text}"
                    
                    st.info("LLM 모델로 종합 분석을 시작합니다...")
                    result = analyze_with_ollama(final_prompt)
                    
                    st.success("✅ 분석 완료")
                    st.subheader("🎧 오디오 분석 결과")
                    st.code(transcript, language='text')
                    st.write(result)

            except Exception as e:
                st.error(f"처리 중 오류가 발생했습니다.")
                st.exception(e)


st.markdown("---")
st.caption("© 2025 시온마케팅 | 개발자 홍석표")
