import streamlit as st
import os, tempfile, cv2, torch, subprocess
import whisper
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SUPPORTED_MIMETYPES = ['video/mp4']

def authenticate_google():
    creds = Credentials.from_authorized_user_info(st.secrets["gcp_token"], SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_drive_files(service, filetype='video'):
    query = "mimeType contains 'video/'" if filetype == 'video' else "mimeType contains 'image/'"
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
    command = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
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

def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"Title: {title}\n\n"
    summary += "1-sec interval visual descriptions:\n" + "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)]) + "\n\n"
    summary += f"Transcript:\n{transcript}\n\n"
    summary += prompt.strip()
    return summary

def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("""{prompt_text}""")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

# Streamlit UI
st.set_page_config(page_title="🎥 Insight Vision AI", layout="wide")
st.title("📊 영상·이미지 기반 AI 분석 시스템")

prompt_text = st.text_area("💬 Ollama 분석 프롬프트 작성",
    "Please analyze the type of content, the primary target audience, whether it's appropriate, and provide 3 improvement suggestions.")

service = authenticate_google()
video_path = None

with st.expander("📁 Google Drive에서 영상 선택"):
    files = list_drive_files(service, filetype='video')
    if files:
        file = st.selectbox("🎬 파일 선택:", files, format_func=lambda x: x['name'])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_path = download_file(service, file['id'], tmp.name)
            st.video(video_path)
    else:
        st.warning("Drive에 mp4 파일이 없습니다.")

uploaded_file = st.file_uploader("또는 영상(mp4) 직접 업로드", type=["mp4"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
        st.video(video_path)

if video_path and st.button("🧠 AI 분석 시작"):
    with st.spinner("🎞️ 프레임 추출 중..."):
        frames = extract_all_keyframes(video_path)
        descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

    with st.spinner("🔊 Whisper로 음성 분석 중..."):
        audio_path = extract_audio_ffmpeg(video_path)
        transcript = transcribe_audio_whisper(audio_path)

    with st.spinner("🤖 Ollama 분석 중..."):
        title = os.path.basename(video_path)
        combined_prompt = summarize_all_inputs(descriptions, transcript, title, prompt_text)
        result = analyze_with_ollama(combined_prompt)

    st.success("✅ 분석 완료")
    st.subheader("📄 AI 분석 결과")
    st.write(result)

st.caption("Powered by Whisper + BLIP + Ollama + ffmpeg + Streamlit + Google Drive")
