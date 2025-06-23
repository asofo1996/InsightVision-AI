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

# UI
st.set_page_config(page_title="Insight Vision AI", layout="wide")
st.title("🎥 영상/이미지 자동 분기 AI 분석 시스템")

prompt_text = st.text_area("💬 Ollama 분석 프롬프트",
    "Please analyze the content type, main audience, tone, and suggest 3 improvements.")

# Drive 인증
service = authenticate_google()

# 선택된 파일 저장 변수
video_path = None
image_path = None
image_obj = None

# 📁 Google Drive 영상 선택
with st.expander("📁 Google Drive에서 영상 선택"):
    video_files = list_drive_files(service, filetype='video')
    if video_files:
        selected = st.selectbox("🎬 영상 선택", video_files, format_func=lambda x: x['name'], key="drive_video")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_path = download_file(service, selected['id'], tmp.name)
            st.video(video_path)

# 📁 Google Drive 이미지 선택
with st.expander("📁 Google Drive에서 이미지 선택"):
    image_files = list_drive_files(service, filetype='image')
    if image_files:
        selected = st.selectbox("🖼️ 이미지 선택", image_files, format_func=lambda x: x['name'], key="drive_image")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            download_file(service, selected["id"], tmp.name)
            image_path = tmp.name
            image_obj = Image.open(image_path).convert("RGB")
            st.image(image_obj, caption="Drive 이미지", use_column_width=True)

# 📤 영상 업로드
uploaded_video = st.file_uploader("📤 직접 영상 업로드", type=["mp4"], key="upload_video")
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
        st.video(video_path)

# 📤 이미지 업로드
uploaded_image = st.file_uploader("📤 직접 이미지 업로드", type=["jpg", "jpeg", "png"], key="upload_image")
if uploaded_image:
    image_obj = Image.open(uploaded_image).convert("RGB")
    st.image(image_obj, caption="업로드한 이미지", use_column_width=True)

# 🎯 자동 분기 AI 분석 버튼
if st.button("🧠 AI 분석 시작"):
    if image_obj is not None:
        with st.spinner("📸 이미지 설명 생성 중..."):
            desc = describe_image_with_blip(image_obj)
        with st.spinner("🧠 Ollama 분석 중..."):
            result = analyze_with_ollama(f"Image Description:\n{desc}\n\n{prompt_text}")
        st.success("✅ 이미지 분석 완료")
        st.subheader("📄 분석 결과")
        st.write(result)

    elif video_path is not None:
        with st.spinner("🎞️ 프레임 추출 중..."):
            frames = extract_all_keyframes(video_path)
            descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

        with st.spinner("🔊 Whisper 음성 분석 중..."):
            audio_path = extract_audio_ffmpeg(video_path)
            transcript = transcribe_audio_whisper(audio_path)

        with st.spinner("🧠 Ollama 종합 분석 중..."):
            title = os.path.basename(video_path)
            final_prompt = summarize_video_inputs(descriptions, transcript, title, prompt_text)
            result = analyze_with_ollama(final_prompt)

        st.success("✅ 영상 분석 완료")
        st.subheader("📄 분석 결과")
        st.write(result)

    else:
        st.warning("⚠️ 분석할 영상 또는 이미지를 선택하거나 업로드해 주세요.")

st.caption("🔗 Powered by Whisper + BLIP + Ollama + Streamlit + Google Drive")
