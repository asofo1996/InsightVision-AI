import streamlit as st
import os, tempfile, cv2, subprocess, torch, time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("AI 콘텐츠 분석 시스템")

prompt_text = st.text_area("분석 요청 문장", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")  # base → tiny 로 전환 (속도 향상)

def describe_image_with_blip(pil_image):
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def extract_audio_ffmpeg(video_path):
    audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    print("📤 ffmpeg 명령어:", ' '.join(command))
    print("📥 ffmpeg stdout:", result.stdout)
    print("📥 ffmpeg stderr:", result.stderr)

    if result.returncode != 0:
        st.error("❌ ffmpeg 실행 실패! stderr를 확인하세요.")
        return None
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        st.error("❌ 오디오 파일이 생성되지 않았습니다 (audio.wav 없음 또는 1KB 이하).")
        return None
    return audio_path

def transcribe_audio_whisper(audio_path):
    model = load_whisper()
    st.info("🟡 Whisper 전사 시작됨 - 기다려 주세요...")
    start = time.time()
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    end = time.time()
    st.success(f"🟢 Whisper 전사 완료 (소요 시간: {int(end - start)}초)")
    return result['text']

def extract_all_keyframes(video_path, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps * interval_sec)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            path = os.path.join(tempfile.gettempdir(), f"frame_{frame_count}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
        frame_count += 1
    cap.release()
    return frames

def summarize_video_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions:\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

video_path = None
image_obj = None

uploaded_video = st.file_uploader("영상 파일 업로드", type=["mp4"], key="upload_video")
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

uploaded_image = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"], key="upload_image")
if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지 미리보기", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    if image_obj is not None:
        if st.button("이미지 분석 시작"):
            with st.spinner("이미지 설명 생성 중..."):
                desc = describe_image_with_blip(image_obj)
            with st.spinner("Ollama 분석 중..."):
                result = analyze_with_ollama(f"Image Description:\n{desc}\n\n{prompt_text}")
            st.success("이미지 분석 완료")
            st.subheader("분석 결과")
            st.write(result)

with col2:
    if video_path is not None:
        if st.button("영상 분석 시작"):
            total_steps = 3
            overall_progress = st.progress(0)
            status_text = st.empty()

            t0 = time.perf_counter()
            with st.spinner("프레임 추출 중 (2초 간격)..."):
                frames = extract_all_keyframes(video_path)
                descriptions = []
                for i, f in enumerate(frames):
                    desc = describe_image_with_blip(Image.open(f))
                    descriptions.append(desc)
                    progress = (i + 1) / len(frames) * (1 / total_steps)
                    overall_progress.progress(progress)
                    status_text.write(f"프레임 분석 진행률: {int(progress * 100)}%")

            status_text.write("프레임 분석 완료")
            overall_progress.progress(1 / total_steps)

            with st.spinner("Whisper를 통한 음성 텍스트 변환 중..."):
                audio_path = extract_audio_ffmpeg(video_path)
                if audio_path is None:
                    st.stop()
                st.audio(audio_path)
                transcript = transcribe_audio_whisper(audio_path)
                overall_progress.progress(2 / total_steps)
                status_text.write("음성 전사 완료")

            with st.spinner("Ollama 종합 분석 중..."):
                title = os.path.basename(video_path)
                final_prompt = summarize_video_inputs(descriptions, transcript, title, prompt_text)
                result = analyze_with_ollama(final_prompt)

            overall_progress.progress(1.0)
            status_text.write("분석 완료")
            t1 = time.perf_counter()

            st.success(f"영상 분석 완료 (총 소요 시간: {int(t1 - t0)}초)")
            st.subheader("분석 결과")
            st.write(result)

if image_obj is None and video_path is None:
    st.warning("영상 또는 이미지를 업로드해 주세요.")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
