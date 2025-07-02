import streamlit as st
import os, tempfile, subprocess, torch, cv2, re
from PIL import Image
import whisper
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ 환경 변수 및 Supabase
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ 파일명 분해
def parse_title_kor(filename):
    filename = os.path.splitext(filename)[0].replace("(", "_").replace(")", "_")
    parts = re.split(r"[_\\-]+", filename)
    client = parts[0] if len(parts) > 0 else "미지정"
    category = parts[1] if len(parts) > 1 else "기타"
    subcontext = "_".join(parts[2:]) if len(parts) > 2 else ""
    return {"client": client, "category": category, "subcontext": subcontext}

# ✅ DB 로딩
def fetch_previous_summaries_by_category(category):
    try:
        result = supabase.table("analysis_results").select("summary_text").eq("category", category).order("created_at", desc=True).limit(5).execute()
        return [r["summary_text"] for r in result.data if r["summary_text"]]
    except: return []

def fetch_experiences_by_category(category):
    try:
        result = supabase.table("performance_logs").select("experience, ctr").neq("experience", "").order("recorded_at", desc=True).limit(10).execute()
        return [f"- {r['experience']} (CTR: {r['ctr']}%)" for r in result.data if r["experience"]]
    except: return []

# ✅ 텍스트 요약
def summarize_all_inputs(frames_desc, transcript, title, prompt):
    return f"""🎬 광고 콘텐츠 정밀 분석

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

# ✅ Ollama용 심화 분석 프롬프트
def generate_ollama_prompt(prompt_text, category, file_name, descriptions, transcript, client, subcontext):
    context_intro = "\n".join(fetch_previous_summaries_by_category(category))
    exp_intro = "\n".join(fetch_experiences_by_category(category))
    return f"""
[광고 분석 대상]
- 고객사: {client}
- 업종/분야: {category}
- 세부 문맥: {subcontext}
- 파일명: {file_name}

[시각 설명 요약]
{chr(10).join(descriptions)}

[텍스트 메시지 요약]
{transcript}

[과거 유사 사례 요약]
{context_intro}

[성과 기반 광고 경험]
{exp_intro}

[정밀 분석 요청 항목]
1. 업종 및 타겟과의 전략적 정합성 분석 (Hook 구조, 콘텐츠 톤, CTA 등)
2. 시청자 유지력과 클릭 유도력 평가
3. CTR과 전환율 관점에서 성과가 높은 콘텐츠와 비교
4. 앞/중/끝 콘텐츠 구조의 개선 방안
5. 최소 3가지 이상의 실무 중심 개선 전략 제시
"""

# ✅ Ollama 실행
def analyze_with_ollama(prompt_text):
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=PromptTemplate.from_template("{prompt_text}"), llm=llm)
    return chain.run(prompt_text=prompt_text)

# ✅ DB 저장
def save_analysis_to_db(client, file, category, subcontext, summary, transcript, descriptions, prompt, ctype):
    supabase.table("analysis_results").insert({
        "client_name": client,
        "category": category,
        "subcontext": subcontext,
        "content_type": ctype,
        "file_name": file,
        "summary_text": summary,
        "raw_transcript": transcript,
        "frame_descriptions": descriptions,
        "prompt_used": prompt,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

def save_performance_to_db(client, file, views, clicks, conversion, ctr, experience):
    supabase.table("performance_logs").insert({
        "client_name": client,
        "file_name": file,
        "views": views,
        "clicks": clicks,
        "conversion": conversion,
        "ctr": ctr,
        "experience": experience,
        "recorded_at": datetime.utcnow().isoformat()
    }).execute()

# ✅ 모델 캐싱
@st.cache_resource
def load_blip():
    return (
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    )

def describe_image_with_blip(image):
    processor, model = load_blip()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    return model.transcribe(audio_path, fp16=torch.cuda.is_available())['text']

def extract_keyframes(video_path, interval_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)
    frames, count = [], 0
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

# ✅ UI 시작
st.set_page_config(page_title="AI 콘텐츠 전략 분석기", layout="wide")
st.title("📊 시온마케팅 콘텐츠 전략 분석 시스템")

prompt_text = st.text_area(
    "✍️ 분석 요청 메시지",
    "광고 콘텐츠의 타겟, 전략, 메시지, 구성 측면에서 정밀 분석하고 개선 전략을 3가지 이상 제안해 주세요."
)

# ✅ 이미지 분석
uploaded_image = st.file_uploader("🖼️ 이미지 업로드", type=["jpg", "jpeg", "png"])
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="업로드된 이미지", use_container_width=True)

    if st.button("이미지 분석 시작"):
        with st.spinner("이미지 설명 생성 중..."):
            image_desc = describe_image_with_blip(img)

        parsed = parse_title_kor(uploaded_image.name)
        full_prompt = f"🖼️ 이미지 설명: {image_desc}\n\n{prompt_text}"
        with st.spinner("전략 분석 중..."):
            result = analyze_with_ollama(full_prompt)

        save_analysis_to_db(parsed["client"], uploaded_image.name, parsed["category"], parsed["subcontext"], result, "", [image_desc], prompt_text, "image")
        st.success("✅ 이미지 분석 완료")
        st.subheader("🧠 분석 결과")
        st.markdown(result)

# ✅ 영상 분석
uploaded_video = st.file_uploader("🎥 영상 업로드", type=["mp4", "mov"])
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    st.video(video_path)

    if st.button("🔍 1차 분석 실행"):
        with st.spinner("프레임 추출 중..."):
            frames = extract_keyframes(video_path)
            descs = [describe_image_with_blip(Image.open(f)) for f in frames]

        with st.spinner("음성 텍스트 변환 중..."):
            audio_path = os.path.join(tempfile.gettempdir(), "audio.wav")
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], stdout=subprocess.DEVNULL)
            transcript = transcribe_audio_whisper(audio_path)

        parsed = parse_title_kor(uploaded_video.name)
        summary = summarize_all_inputs(descs, transcript, uploaded_video.name, prompt_text)
        result = analyze_with_ollama(summary)

        st.success("✅ 1차 분석 완료")
        st.subheader("🧠 전략 분석 결과")
        st.markdown(result)

        # ✅ 2차 정밀 분석 버튼
        if st.button("📌 더 정밀한 실전 전략 솔루션 요청"):
            with st.spinner("🧠 고도화된 전략 분석 중..."):
                full_prompt = generate_ollama_prompt(prompt_text, parsed["category"], uploaded_video.name, descs, transcript, parsed["client"], parsed["subcontext"])
                deep_result = analyze_with_ollama(full_prompt)

            if deep_result:
                st.success("✅ 정밀 전략 분석 완료")
                st.subheader("💡 고도화된 실전 전략 제안")
                st.markdown(deep_result)
                save_analysis_to_db(parsed["client"], uploaded_video.name, parsed["category"], parsed["subcontext"], deep_result, transcript, descs, prompt_text, "video")
            else:
                st.error("❌ 정밀 분석 결과가 비어 있습니다. 프롬프트 또는 Ollama 설정을 점검해 주세요.")

# ✅ 광고 성과 입력
st.markdown("---")
st.header("📊 광고 성과 입력")
with st.form("performance_form"):
    file_name = st.text_input("파일명 (예: 병원명_분야_날짜_버전.mp4)", "")
    parsed = parse_title_kor(file_name)
    views = st.number_input("조회수", min_value=0)
    clicks = st.number_input("클릭수", min_value=0)
    conversion = st.number_input("전환수", min_value=0)
    ctr = round((clicks / views) * 100, 2) if views else 0.0
    experience = st.text_area("📝 광고 경험 메모", placeholder="예: 한지 배경 넣었더니 CTR 상승")
    submitted = st.form_submit_button("성과 + 경험 저장")
    if submitted and file_name:
        save_performance_to_db(parsed["client"], file_name, views, clicks, conversion, ctr, experience)
        st.success(f"✅ {parsed['client']} 광고 경험 저장 완료")

st.caption("© 2025 시온마케팅 | 개발자 홍석표")
