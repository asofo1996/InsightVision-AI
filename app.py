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

# ✅ Render 환경 대응
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
    query = " or ".join([f"mimeType='{m}'" for m in SUPPORTED_VIDEO_TYPES]) if filetype == 'video' else " or ".join([f"mimeType='{m}'" for m in SUPPORTED_IMAGE_TYPES])
    results = service.files().list(q=query, pageSize=20, fields="files(id, name)").execute()
    return results.get('files', [])

def download_file(service, file_id, filename):
    request = service.files().get_media(fileId=file_id)
    with open(filename, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return filename

def extract_all_keyframes(video_path, fps=1):
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

def describe_image_with_blip(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

def extract_audio_ffmpeg(video_path):
    audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def download_youtube_video(url):
    video_path = os.path.join(tempfile.gettempdir(), "youtube_video.mp4")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': video_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return os.path.splitext(video_path)[0] + ".wav"

def summarize_video_inputs(descriptions, transcript, title, prompt):
    summary = f"Title: {title}\n\nFrame Descriptions:\n" + "\n".join([f"{i+1}. {d}" for i, d in enumerate(descriptions)])
    summary += f"\n\nTranscript:\n{transcript}\n\n{prompt.strip()}"
    return summary

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("{prompt_text}")
    chain = LLMChain(prompt=template, llm=Ollama(model="llama3"))
    return chain.run(prompt_text=prompt_text)

# 🔷 Streamlit UI
st.set_page_config(page_title="AI 분석 시스템", layout="wide")
st.title("📊 영상 · 이미지 · 오디오 AI 분석 시스템")

prompt_text = st.text_area("분석 프롬프트", "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

video_path, image_obj = None, None

# 🔺 유튜브 링크 입력
st.subheader("🎧 오디오 분석 (유튜브/음성 파일)")
option = st.radio("분석할 소스를 선택하세요", ["유튜브 링크", "로컬 음성파일"])

if option == "유튜브 링크":
    youtube_url = st.text_input("유튜브 링크를 입력하세요")
    if st.button("오디오 요약 분석 시작") and youtube_url:
        try:
            audio_path = download_youtube_video(youtube_url)
            transcript = transcribe_audio_whisper(audio_path)
            st.subheader("🔊 전체 텍스트")
            st.text(transcript)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.subheader("🧠 요약 결과")
            st.write(result)
        except Exception as e:
            st.error(f"실패: {e}")
else:
    uploaded_audio = st.file_uploader("음성 파일 업로드 (WAV)", type=["wav"])
    if uploaded_audio and st.button("오디오 요약 분석 시작"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            transcript = transcribe_audio_whisper(tmp.name)
            st.subheader("🔊 전체 텍스트")
            st.text(transcript)
            result = analyze_with_ollama(f"Transcript:\n{transcript}\n\n{prompt_text}")
            st.subheader("🧠 요약 결과")
            st.write(result)

# 🔷 기타 분석 기능 (생략 가능)
st.divider()
st.subheader("📂 기본 영상 · 이미지 분석 기능")

uploaded_video = st.file_uploader("📹 영상 업로드", type=["mp4"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

uploaded_image = st.file_uploader("🖼️ 이미지 업로드", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_container_width=True)

if st.button("📊 전체 분석 시작"):
    if video_path:
        with st.spinner("🔍 프레임 추출 중..."):
            frames = extract_all_keyframes(video_path)
            descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]
        with st.spinner("🧠 음성 분석 중..."):
            audio_path = extract_audio_ffmpeg(video_path)
            transcript = transcribe_audio_whisper(audio_path)
        with st.spinner("💡 Ollama 분석 중..."):
            title = os.path.basename(video_path)
            prompt = summarize_video_inputs(descriptions, transcript, title, prompt_text)
            result = analyze_with_ollama(prompt)
        st.success("🎉 분석 완료")
        st.write(result)
    elif image_obj:
        with st.spinner("🖼️ 이미지 설명 생성 중..."):
            desc = describe_image_with_blip(image_obj)
        with st.spinner("💡 Ollama 분석 중..."):
            result = analyze_with_ollama(f"Image Description:\n{desc}\n\n{prompt_text}")
        st.success("🎉 분석 완료")
        st.write(result)
    else:
        st.warning("영상 또는 이미지를 업로드해 주세요.")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
