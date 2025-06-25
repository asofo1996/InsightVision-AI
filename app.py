import streamlit as st
import os, tempfile, cv2, torch, subprocess, shutil
import whisper
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import yt_dlp

if os.path.exists("/etc/secrets/secrets.toml"):
    os.makedirs(".streamlit", exist_ok=True)
    shutil.copy("/etc/secrets/secrets.toml", ".streamlit/secrets.toml")

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SUPPORTED_VIDEO_TYPES = ['video/mp4']
SUPPORTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/jpg']

def authenticate_google():
    creds = Credentials.from_authorized_user_info(st.secrets["gcp_token"], SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_drive_files(service, filetype='video'):
    if filetype == 'video':
        query = " or ".join([f"mimeType='{m}'" for m in SUPPORTED_VIDEO_TYPES])
    elif filetype == 'image':
        query = " or ".join([f"mimeType='{m}'" for m in SUPPORTED_IMAGE_TYPES])
    results = service.files().list(q=query, pageSize=20, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def download_file(service, file_id, filename):
    request = service.files().get_media(fileId=file_id)
    with open(filename, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return filename

def extract_audio_ffmpeg(video_path):
    audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def extract_all_keyframes(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps * fps)
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

def describe_image_with_blip(pil_image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

def summarize_video_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "Frame Descriptions (1s intervals):\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

def download_youtube_audio_only(youtube_url):
    output_path = os.path.join(tempfile.gettempdir(), "youtube_audio.wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

st.set_page_config(page_title="Insight Vision AI", layout="wide")
st.title("AI 분석 시스템")

prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

video_path = None
image_obj = None

uploaded_video = st.file_uploader("영상 파일 업로드", type=["mp4"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

uploaded_image = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_container_width=True)

if st.button("영상 분석 시작") and video_path:
    with st.spinner("프레임 분석 중..."):
        frames = extract_all_keyframes(video_path)
        descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

    with st.spinner("Whisper를 통한 음성 텍스트 변환 중..."):
        audio_path = extract_audio_ffmpeg(video_path)
        transcript = transcribe_audio_whisper(audio_path)

    with st.spinner("Ollama 종합 분석 중..."):
        final_prompt = summarize_video_inputs(descriptions, transcript, os.path.basename(video_path), prompt_text)
        result = analyze_with_ollama(final_prompt)
        st.subheader("분석 결과")
        st.write(result)

if st.button("이미지 분석 시작") and image_obj:
    with st.spinner("BLIP 이미지 설명 중..."):
        desc = describe_image_with_blip(image_obj)
    with st.spinner("Ollama 분석 중..."):
        result = analyze_with_ollama(f"Image Description:\n{desc}\n\n{prompt_text}")
        st.subheader("분석 결과")
        st.write(result)

# ✅ 하단 추가 기능: 음성만 요약 분석
st.markdown("---")
st.subheader("🎧 음성 또는 유튜브 링크 요약 분석")

source_type = st.radio("입력 방식", ["유튜브 링크", "로컬 음성 파일"], horizontal=True)
input_audio_path = None

if source_type == "유튜브 링크":
    youtube_audio_url = st.text_input("유튜브 링크를 입력하세요")
    if st.button("오디오 요약 분석 시작") and youtube_audio_url:
        with st.spinner("유튜브 오디오 다운로드 중..."):
            input_audio_path = download_youtube_audio_only(youtube_audio_url)
elif source_type == "로컬 음성 파일":
    uploaded_audio = st.file_uploader("음성 파일 업로드", type=["mp3", "wav"], key="audio")
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            input_audio_path = tmp.name

if input_audio_path:
    with st.spinner("Whisper로 음성 텍스트 변환 중..."):
        transcript = transcribe_audio_whisper(input_audio_path)

    with st.spinner("Ollama 요약 및 분석 중..."):
        result = analyze_with_ollama(f"음성 텍스트:\n{transcript}\n\n{prompt_text}")
        st.subheader("📝 요약 결과")
        st.write(result)

if image_obj is None and video_path is None:
    st.warning("영상 또는 이미지를 업로드해 주세요.")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
