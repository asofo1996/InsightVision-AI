import streamlit as st
import os, tempfile, subprocess, torch, cv2
from PIL import Image
import whisper
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 앱 설정
st.set_page_config(page_title="시온마케팅 콘텐츠 분석기", layout="wide")
st.title("🎯 시온마케팅 AI 콘텐츠 분석 시스템")
st.markdown("---")

# 사용자 입력
prompt_text = st.text_area("분석 프롬프트", 
    "Please analyze the content type, main audience, tone, and suggest 3 improvements.", key="main_prompt")

# 모델 불러오기
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

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

def extract_keyframes(video_path, interval_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)
    frames, count = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
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
    summary = f"🎬 영상 제목: {title}\n\n🖼️ 프레임 설명:\n"
    summary += "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)])
    summary += f"\n\n📝 음성 텍스트:\n{transcript}\n\n🔍 분석 지시:\n{prompt.strip()}"
    return summary

# 업로드 요소
uploaded_video = st.file_uploader("📽️ 영상 파일 업로드", type=["mp4", "mov"], key="video_upload")
uploaded_image = st.file_uploader("🖼️ 이미지 파일 업로드", type=["jpg", "jpeg", "png"], key="image_upload")
uploaded_audio = st.file_uploader("🎧 음성 파일 업로드", type=["mp3", "wav"], key="audio_upload")

# 이미지 분석
if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드 이미지", use_container_width=True)
    if st.button("이미지 분석 시작", key="start_image_analysis"):
        with st.spinner("이미지 설명 생성 중..."):
            desc = describe_image_with_blip(image_obj)
        with st.spinner("Ollama 분석 중..."):
            result = analyze_with_ollama(f"시온마케팅 광고 전문가 기준 분석\n파일명: {uploaded_image.name}\n이미지 설명: {desc}\n\n{prompt_text}")
        st.success("분석 완료 ✅")
        st.write(result)

# 영상 분석
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    st.video(video_path)
    if st.button("영상 분석 시작", key="start_video_analysis"):
        with st.spinner("📸 프레임 추출 중..."):
            frames = extract_keyframes(video_path)
            descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

        with st.spinner("🗣️ Whisper 음성 변환 중..."):
            audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            transcript = transcribe_audio_whisper(audio_path)

        with st.spinner("🧠 Ollama 분석 중..."):
            final_prompt = summarize_all_inputs(descriptions, transcript, os.path.basename(video_path), prompt_text)
            result = analyze_with_ollama(final_prompt)
            st.success("영상 분석 완료 ✅")
            st.subheader("🧠 분석 결과")
            st.write(result)

# 음성 분석
if uploaded_audio:
    suffix = ".mp3" if uploaded_audio.name.endswith(".mp3") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_audio.read())
        audio_path = tmp.name
    if st.button("음성 분석 시작", key="start_audio_analysis"):
        if audio_path.endswith(".mp3"):
            converted_path = audio_path.replace(".mp3", ".wav")
            subprocess.run(["ffmpeg", "-y", "-i", audio_path, converted_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio_path = converted_path
        with st.spinner("Whisper 텍스트 변환 중..."):
            transcript = transcribe_audio_whisper(audio_path)
        with st.spinner("Ollama 분석 중..."):
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
        st.success("음성 분석 완료 ✅")
        st.write("전체 텍스트:")
        st.code(transcript)
        st.write("요약 결과:")
        st.write(result)

# 푸터
st.markdown("---")
st.caption("© 2025 시온마케팅 | 개발자 홍석표")
