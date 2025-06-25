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

def get_latest_wav_file():
    wav_files = sorted(
        glob.glob(os.path.join(UPLOAD_DIR, "*.wav")),
        key=os.path.getmtime,
        reverse=True
    )
    return wav_files[0] if wav_files else None

def safe_transcribe():
    filepath = get_latest_wav_file()
    
    # 디버깅: 찾은 파일 경로 확인
    if filepath:
        st.info(f"✅ Whisper가 분석할 오디오 파일을 찾았습니다: {filepath}")
    else:
        st.error("❌ Whisper가 분석할 .wav 파일을 'uploaded' 폴더에서 찾지 못했습니다.")
        # 디버깅: 현재 폴더 내용물 확인
        st.warning(f"'uploaded' 폴더 내용물: {os.listdir(UPLOAD_DIR)}")
        raise FileNotFoundError(".wav 파일을 찾을 수 없습니다.")

    if not os.path.exists(filepath):
         raise FileNotFoundError(f"파일 경로가 존재하지 않습니다: {filepath}")

    try:
        waveform, sample_rate = torchaudio.load(filepath)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        audio = waveform.squeeze().numpy()
        
        model = whisper.load_model("base")
        result = model.transcribe(audio, fp16=torch.cuda.is_available(), language='ko')
        return result['text']
    except Exception as e:
        st.error(f"오디오 파일 처리 중 오류 발생: {e}")
        # 오류 발생 시 더 상세한 정보 제공
        raise RuntimeError(f"torchaudio 또는 Whisper에서 '{filepath}' 파일을 처리하는 데 실패했습니다.") from e


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

def download_youtube_audio(url):
    out_base = os.path.join(UPLOAD_DIR, "youtube_audio")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_base, # 확장자 없이 경로 지정
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': False, # 디버깅을 위해 상세 로그 출력
        'noplaylist': True
    }
    st.info("yt-dlp로 오디오 다운로드를 시작합니다...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    st.success("오디오 다운로드 및 변환 완료.")
    
    # 디버깅: 다운로드 후 폴더 내용 확인
    st.info(f"다운로드 완료 후 'uploaded' 폴더 내용: {os.listdir(UPLOAD_DIR)}")


# --- Streamlit UI 구성 ---
st.subheader("파일 직접 업로드하여 분석")
# (이전과 동일한 파일 업로드 UI 코드는 생략)

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
        st.exception(e) # 전체 Traceback을 화면에 출력

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
