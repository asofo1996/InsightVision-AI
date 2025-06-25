import streamlit as st
import os, tempfile, cv2, torch, subprocess, glob
from PIL import Image
import whisper
import yt_dlp
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 기본 세팅
st.set_page_config(page_title="AI 콘텐츠 분석 시스템", layout="wide")
st.title("🎬 AI 콘텐츠 분석 시스템")
prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

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

def transcribe_audio_whisper(auto_detect_path):
    if not os.path.exists(auto_detect_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {auto_detect_path}")
    model = whisper.load_model("base")
    result = model.transcribe(auto_detect_path, fp16=torch.cuda.is_available())
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
            path = os.path.join(tempfile.gettempdir(), f"frame_{count}.jpg")
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

# ✅ 다운로드 후 확장자 자동 감지
def download_youtube_audio(url):
    out_base = os.path.join(tempfile.gettempdir(), "youtube_audio")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_base + ".%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # 확장자가 무엇이 되었든 .wav만 찾아냄
    matched_files = glob.glob(out_base + "*.wav")
    if not matched_files:
        raise FileNotFoundError("다운로드된 .wav 파일을 찾을 수 없습니다.")
    return matched_files[0]

# 업로드
uploaded_video = st.file_uploader("🎥 영상 업로드", type=["mp4", "mov", "mkv"])
uploaded_image = st.file_uploader("🖼 이미지 업로드", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("🎧 음성 업로드", type=["wav", "mp3"])

if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_container_width=True)
    if st.button("이미지 분석 시작"):
        desc = describe_image_with_blip(image_obj)
        result = analyze_with_ollama(f"Image:\n{desc}\n\n{prompt_text}")
        st.subheader("이미지 분석 결과")
        st.write(result)

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

        if st.button("영상 분석 시작"):
            with st.spinner("프레임 추출 중..."):
                frames = extract_keyframes(video_path)
                descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

            with st.spinner("Whisper 음성 분석 중..."):
                audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
                subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                transcript = transcribe_audio_whisper(audio_path)

            with st.spinner("Ollama 분석 중..."):
                final_prompt = summarize_all_inputs(descriptions, transcript, os.path.basename(video_path), prompt_text)
                result = analyze_with_ollama(final_prompt)
                st.subheader("영상 분석 결과")
                st.write(result)

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
        if st.button("음성 파일 분석 시작"):
            transcript = transcribe_audio_whisper(audio_path)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.subheader("음성 분석 결과")
            st.code(transcript)
            st.write(result)

# 유튜브 or 로컬 경로
st.markdown("---")
st.subheader("🔗 유튜브 링크 또는 로컬 오디오 분석")
col1, col2 = st.columns([1, 3])
with col1:
    mode = st.radio("모드 선택", ["유튜브 링크", "로컬 경로"], horizontal=True)
with col2:
    user_input = st.text_input("링크 또는 로컬 경로 입력")

if st.button("오디오 요약 분석 시작"):
    try:
        if mode == "유튜브 링크":
            with st.spinner("유튜브 오디오 다운로드 중..."):
                audio_path = download_youtube_audio(user_input)
        else:
            audio_path = user_input
            if not os.path.exists(audio_path):
                raise FileNotFoundError("입력한 로컬 파일이 존재하지 않습니다.")

        st.text(f"🔎 분석할 오디오 경로: {audio_path}")

        with st.spinner("Whisper 텍스트 변환 중..."):
            transcript = transcribe_audio_whisper(audio_path)

        with st.spinner("Ollama 요약 중..."):
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.success("✅ 분석 완료")
            st.subheader("전체 텍스트")
            st.code(transcript)
            st.subheader("요약 결과")
            st.write(result)

    except Exception as e:
        st.error(f"🚨 오류 발생: {str(e)}")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
