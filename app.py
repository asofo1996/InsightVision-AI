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

# ✅ 환경 변수 로딩
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ 파일명 기반 자동 분류
def parse_title_kor(filename):
    filename = os.path.splitext(filename)[0]
    filename = filename.replace("(", "_").replace(")", "_")
    parts = re.split(r"[_\-]+", filename)
    client = parts[0] if len(parts) > 0 else "미지정"
    category = parts[1] if len(parts) > 1 else "기타"
    subcontext = "_".join(parts[2:]) if len(parts) > 2 else ""
    return {"client": client, "category": category, "subcontext": subcontext}

# ✅ Supabase 기존 데이터 불러오기
def fetch_previous_summaries_by_category(category):
    try:
        result = supabase.table("analysis_results").select("summary_text").eq("category", category).order("created_at", desc=True).limit(5).execute()
        return [r["summary_text"] for r in result.data if r["summary_text"]]
    except Exception as e:
        return []

def fetch_experiences_by_category(category):
    try:
        result = supabase.table("performance_logs").select("experience, ctr").neq("experience", "").order("recorded_at", desc=True).limit(10).execute()
        return [f"- {r['experience']} (CTR: {r['ctr']}%)" for r in result.data if r["experience"]]
    except Exception as e:
        return []

# ✅ 콘텐츠 요약 템플릿
def summarize_all_inputs(frames_desc, transcript, title, prompt):
    summary = f"""🎬 광고 콘텐츠 정밀 분석

📌 광고 제목: {title}

🖼️ 시각 콘텐츠 요약:
{chr(10).join([f"{i+1}. {desc}" for i, desc in enumerate(frames_desc)])}

🎙️ 텍스트 요약:
{transcript}

🔍 분석 목적:
- 업종/종목과 전략적 적합성 평가
- 국내 타겟과의 정합성
- 전환율 관점에서의 콘텐츠 구조 분석
- 실행 가능한 실무 개선 전략 도출 (3가지 이상)

💡 사용자 요청:
{prompt.strip()}
"""
    return summary

# ✅ Ollama 분석 프롬프트 생성
def generate_ollama_prompt(prompt_text, category, file_name, descriptions, transcript, client, subcontext):
    previous_summaries = fetch_previous_summaries_by_category(category)
    experiences = fetch_experiences_by_category(category)
    context_intro = "\n".join(previous_summaries)
    exp_intro = "\n".join(experiences)

    full_prompt = f"""
[분석 대상 정보]
- 고객사: {client}
- 업종: {category}
- 세부 문맥: {subcontext}
- 파일명: {file_name}

[프레임별 시각 설명]
{chr(10).join(descriptions)}

[음성 텍스트 요약]
{transcript}

[과거 유사 사례 요약]
{context_intro}

[성과 기반 실무 경험 요약]
{exp_intro}

[전략 분석 요청 항목]
1. 업종 및 타겟과의 전략적 정합성 분석
2. 시청자 유지력과 CTA 유도력 분석
3. 전환율 극대화를 위한 실무 전략 3가지 이상 도출
"""
    return full_prompt

# ✅ Ollama 분석 실행
def analyze_with_ollama(prompt_text):
    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=prompt_text)

# ✅ DB 저장 함수
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

# ✅ BLIP 로드 (캐싱)
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# ✅ 이미지 설명 생성
def describe_image_with_blip(pil_image):
    processor, model = load_blip()
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# ✅ Whisper 음성 텍스트 변환
def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    return result['text']

# ✅ 프레임 추출
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

# ✅ Streamlit 인터페이스
st.set_page_config(page_title="AI 콘텐츠 전략 분석기", layout="wide")
st.title("📊 시온마케팅 콘텐츠 전략 분석 시스템")

prompt_text = st.text_area("✍️ 분석 요청 메시지", "광고 콘텐츠의 타겟, 전략, 메시지, 구성 측면에서 정밀 분석하고 개선 전략을 3가지 이상 제안해 주세요.")

uploaded_video = st.file_uploader("🎥 분석할 영상 업로드", type=["mp4", "mov"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    st.video(video_path)

    if st.button("🔍 1차 분석 실행"):
        with st.spinner("🖼️ 프레임 추출 중..."):
            frames = extract_keyframes(video_path)
            descriptions = [describe_image_with_blip(Image.open(f)) for f in frames]

        with st.spinner("🎙️ 음성 텍스트 변환 중..."):
            audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            transcript = transcribe_audio_whisper(audio_path)

        parsed = parse_title_kor(uploaded_video.name)
        client, category, subcontext = parsed["client"], parsed["category"], parsed["subcontext"]

        summary_prompt = summarize_all_inputs(descriptions, transcript, uploaded_video.name, prompt_text)
        result = analyze_with_ollama(summary_prompt)

        st.subheader("🧠 전략 분석 결과")
        st.markdown(result)

        # ✅ 고도화된 실전 전략 요청 버튼
        if st.button("📌 더 정밀한 실전 전략 솔루션 요청"):
            deep_prompt = generate_ollama_prompt(prompt_text, category, uploaded_video.name, descriptions, transcript, client, subcontext)
            deep_result = analyze_with_ollama(deep_prompt)
            st.subheader("💡 고도화된 실전 전략 제안")
            st.markdown(deep_result)
            save_analysis_to_db(client, uploaded_video.name, category, subcontext, deep_result, transcript, descriptions, prompt_text, content_type="video")
