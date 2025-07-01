import streamlit as st
import os, tempfile, subprocess, torch, cv2
from PIL import Image
import whisper
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
from dotenv import load_dotenv
import re
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def parse_title_kor(filename):
    parts = filename.replace(".mp4", "").replace(".mov", "").replace(".mp3", "").replace(".wav", "").replace(".jpg", "").replace(".jpeg", "").replace(".png", "").split("-")
    return {
        "client": parts[0] if len(parts) > 0 else "미지정",
        "category": parts[1] if len(parts) > 1 else "기타",
        "subcontext": "-".join(parts[2:]) if len(parts) > 2 else ""
    }

def fetch_previous_summaries_by_category(category):
    try:
        result = supabase.table("analysis_results") \
            .select("summary_text") \
            .eq("category", category) \
            .order("created_at", desc=True) \
            .limit(5) \
            .execute()
        return [r["summary_text"] for r in result.data if r["summary_text"]]
    except Exception as e:
        print("불러오기 실패:", e)
        return []

def fetch_experiences_by_category(category):
    try:
        result = supabase.table("performance_logs") \
            .select("experience, ctr") \
            .neq("experience", "") \
            .order("recorded_at", desc=True) \
            .limit(10) \
            .execute()
        return [f"- {r['experience']} (CTR: {r['ctr']}%)" for r in result.data if r["experience"]]
    except Exception as e:
        print("경험 불러오기 실패:", e)
        return []

def analyze_with_ollama(prompt_text, category=None):
    previous_summaries = fetch_previous_summaries_by_category(category) if category else []
    experiences = fetch_experiences_by_category(category) if category else []

    context_intro = "\n".join([f"- {s}" for s in previous_summaries])
    exp_intro = "\n".join(experiences)

    final_prompt = f"""🧠 참고할 이전 {category} 광고 분석 요약:
{context_intro}

📊 광고 성과 + 경험:
{exp_intro}

🔍 지금 분석할 콘텐츠:
{prompt_text}
"""
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=final_prompt)

def save_analysis_to_db(client_name, file_name, category, subcontext, summary, transcript, descriptions, prompt_text, content_type):
    supabase.table("analysis_results").insert({
        "client_name": client_name,
        "category": category,
        "subcontext": subcontext,
        "content_type": content_type,
        "file_name": file_name,
        "summary_text": summary,
        "raw_transcript": transcript,
        "frame_descriptions": descriptions,
        "prompt_used": prompt_text,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

def save_performance_to_db(client_name, file_name, views, clicks, conversion, ctr, experience):
    supabase.table("performance_logs").insert({
        "client_name": client_name,
        "file_name": file_name,
        "views": views,
        "clicks": clicks,
        "conversion": conversion,
        "ctr": ctr,
        "experience": experience,
        "recorded_at": datetime.utcnow().isoformat()
    }).execute()

st.set_page_config(page_title="AI 광고 전략 분석기", layout="wide")
st.title("🎯 시온마케팅 콘텐츠 분석 시스템")

prompt_text = st.text_area("분석 프롬프트", "광고 콘텐츠의 타겟, 톤, 메시지를 분석하고 전략을 추천해 주세요.")

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

def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"🎬 콘텐츠 제목: {title}\n\n🖼️ 프레임 설명:\n"
    summary += "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)])
    summary += f"\n\n📝 텍스트:\n{transcript}\n\n🔍 분석 지시:\n{prompt.strip()}"
    return summary

uploaded_image = st.file_uploader("🖼️ 이미지 파일 업로드", type=["jpg", "jpeg", "png"])
if uploaded_image:
    pil_image = Image.open(uploaded_image).convert("RGB")
    st.image(pil_image, caption="업로드된 이미지", use_container_width=True)
    if st.button("이미지 분석 시작"):
        with st.spinner("이미지 설명 생성 중..."):
            image_desc = describe_image_with_blip(pil_image)
        parsed = parse_title_kor(uploaded_image.name)
        client_name = parsed["client"]
        category = parsed["category"]
        subcontext = parsed["subcontext"]
        with st.spinner("Ollama 전략 분석 중..."):
            image_prompt = f"🖼️ 이미지 설명: {image_desc}\n\n{prompt_text}"
            result = analyze_with_ollama(image_prompt, category)
        save_analysis_to_db(client_name, uploaded_image.name, category, subcontext, result, "", [image_desc], prompt_text, content_type="image")
        st.success("이미지 분석 완료 ✅")
        st.subheader("🧠 분석 결과")
        st.write(result)

uploaded_video = st.file_uploader("🎥 영상 파일 업로드", type=["mp4", "mov"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    st.video(video_path)
    if st.button("영상 분석 시작"):
        with st.spinner("프레임 추출 중..."):
            frames = extract_keyframes(video_path)
            descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]
        with st.spinner("음성 텍스트 변환 중..."):
            audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            transcript = transcribe_audio_whisper(audio_path)
        parsed = parse_title_kor(uploaded_video.name)
        client_name = parsed["client"]
        category = parsed["category"]
        subcontext = parsed["subcontext"]
        with st.spinner("Ollama 분석 중..."):
            full_prompt = summarize_all_inputs(descriptions, transcript, uploaded_video.name, prompt_text)
            result = analyze_with_ollama(full_prompt, category)
        save_analysis_to_db(client_name, uploaded_video.name, category, subcontext, result, transcript, descriptions, prompt_text, content_type="video")
        st.success("영상 분석 완료 ✅")
        st.subheader("🧠 분석 결과")
        st.write(result)

st.markdown("---")
st.header("📊 광고 성과 입력")
with st.form("performance_form"):
    perf_file_name = st.text_input("파일명 (예: 하나치과-임플란트-한지배경.mp4)", "")
    parsed = parse_title_kor(perf_file_name)
    perf_client_name = parsed["client"]
    views = st.number_input("조회수", min_value=0)
    clicks = st.number_input("클릭수", min_value=0)
    conversion = st.number_input("전환수", min_value=0)
    ctr = round((clicks / views) * 100, 2) if views else 0.0
    experience = st.text_area("📝 광고 경험 메모", placeholder="예: 한지 배경 넣었더니 CTR 상승")
    submitted = st.form_submit_button("성과 + 경험 저장")
    if submitted and perf_file_name:
        save_performance_to_db(perf_client_name, perf_file_name, views, clicks, conversion, ctr, experience)
        st.success(f"{perf_client_name} 성과 + 경험 저장 완료 ✅")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
