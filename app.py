import streamlit as st
import os, tempfile, cv2, torch, subprocess
import whisper
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SUPPORTED_MIMETYPES = ['video/mp4']

def authenticate_google():
    creds = None
    if os.path.exists('token.json'):
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        client_id = st.secrets["GOOGLE_CLIENT_ID"]
        client_secret = st.secrets["GOOGLE_CLIENT_SECRET"]
        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"]
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = flow.run_console()
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def list_drive_files(service, filetype='video'):
    if filetype == 'video':
        query = " or ".join([f"mimeType='{m}'" for m in SUPPORTED_MIMETYPES])
    elif filetype == 'image':
        query = "mimeType contains 'image/'"
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

def extract_audio(video_path):
    path = os.path.join(tempfile.gettempdir(), "audio.wav")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path

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

# UI 시작
st.set_page_config(page_title="AI 콘텐츠 분석 솔루션", layout="wide")
st.title("📊 AI 기반 영상 및 이미지 분석 시스템")

prompt_text = st.text_area("💬 Ollama에게 분석 요청할 영어 명령어를 작성하세요:",
    "Please analyze the type of content, the primary target audience, whether it's appropriate, and provide 3 improvement suggestions. Respond in English.")

service = authenticate_google()
video_path = None

with st.expander("📁 Google Drive에서 영상 선택하기"):
    files = list_drive_files(service, filetype='video')
    if files:
        file = st.selectbox("🎬 Drive 파일을 선택하세요:", files, format_func=lambda x: x['name'])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_path = download_file(service, file['id'], tmp.name)
            st.video(video_path)
    else:
        st.warning("Drive에서 mp4 파일을 찾을 수 없습니다.")

uploaded_file = st.file_uploader("📂 또는 영상(mp4) 업로드", type=["mp4"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
        st.video(video_path)

with st.expander("🖼️ 이미지 파일 단독 분석"):
    image_file = st.file_uploader("이미지 파일 업로드 (jpg/png)", type=["jpg", "jpeg", "png"])
    if image_file:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="업로드된 이미지", use_column_width=True)
        if st.button("🧠 업로드된 이미지 분석 시작"):
            with st.spinner("BLIP로 이미지 설명 생성 중..."):
                description = describe_image_with_blip(img)
            with st.spinner("Ollama 분석 중..."):
                result_img = analyze_with_ollama(f"Image Description:\n{description}\n\n{prompt_text}")
            st.success("✅ 이미지 분석 완료")
            st.subheader("📄 이미지 분석 결과 (영문)")
            st.write(result_img)

with st.expander("🖼️ Google Drive에서 이미지 파일 선택하기"):
    image_files = list_drive_files(service, filetype='image')
    if image_files:
        selected_image = st.selectbox("이미지 파일을 선택하세요:", image_files, format_func=lambda x: x["name"])
        if selected_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                download_file(service, selected_image["id"], tmp_img.name)
                img = Image.open(tmp_img.name).convert("RGB")
                st.image(img, caption="Google Drive에서 불러온 이미지", use_column_width=True)
                if st.button("🧠 Drive 이미지 분석 시작"):
                    with st.spinner("BLIP로 이미지 설명 생성 중..."):
                        description = describe_image_with_blip(img)
                    with st.spinner("Ollama 분석 중..."):
                        result_drive_img = analyze_with_ollama(f"Image Description:\n{description}\n\n{prompt_text}")
                    st.success("✅ 이미지 분석 완료")
                    st.subheader("📄 이미지 분석 결과 (Google Drive)")
                    st.write(result_drive_img)
    else:
        st.warning("Google Drive에 이미지 파일이 없습니다.")

if video_path:
    st.markdown("---")
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    if st.button("🧠 AI 영상 분석 시작"):
        with st.spinner("🎞️ 1초 프레임 추출 중..."):
            frames = extract_all_keyframes(video_path)
            descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

        with st.spinner("🔊 음성 전사 중..."):
            audio_path = extract_audio(video_path)
            transcript = transcribe_audio_whisper(audio_path)

        with st.spinner("🧠 Ollama 분석 중..."):
            title = os.path.basename(video_path)
            combined_prompt = summarize_all_inputs(descriptions, transcript, title, prompt_text)
            result_en = analyze_with_ollama(combined_prompt)
            st.session_state['result_en'] = result_en

        st.success("✅ 영상 분석 완료")
        st.subheader("📄 영상 분석 결과 (영문)")
        st.write(result_en)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Powered by: Whisper + BLIP + Ollama + LangChain + Streamlit + Google Drive")
