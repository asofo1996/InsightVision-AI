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

# ✅ 파일명 기반 자동 분류 함수 (한글/괄호 포함 구조 대응)
def parse_title_kor(filename):
    filename = os.path.splitext(filename)[0]
    filename = filename.replace("(", "_").replace(")", "_")
    parts = re.split(r"[_\-]+", filename)
    client = parts[0] if len(parts) > 0 else "미지정"
    category = parts[1] if len(parts) > 1 else "기타"
    subcontext = "_".join(parts[2:]) if len(parts) > 2 else ""
    return {
        "client": client,
        "category": category,
        "subcontext": subcontext
    }

# ✅ DB에서 기존 요약 및 경험 불러오기
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

# ✅ 전략 분석 요청
def analyze_with_ollama(prompt_text, category=None):
    previous_summaries = fetch_previous_summaries_by_category(category) if category else []
    experiences = fetch_experiences_by_category(category) if category else []

    context_intro = "\n".join([f"- {s}" for s in previous_summaries])
    exp_intro = "\n".join(experiences)

    full_prompt = f"""[분석 요청 목적]
1. 광고 콘텐츠가 업종과 종목에 전략적으로 적합한지 판단해 주세요.
2. 현재 국내 타겟 시장 및 소비자 특성과 비교했을 때 타겟 정합성이 높은지 평가해 주세요.
3. 클릭률과 전환율을 높이기 위한 콘텐츠 구성 요소가 잘 작동하는지 분석해 주세요.
4. 실무적으로 실행 가능한 개선 전략을 3가지 이상 구체적으로 제안해 주세요.

[콘텐츠 정보]
- 과거 광고 분석 요약 ({category}):
{context_intro}

- 광고 성과 및 경험:
{exp_intro}

- 현재 콘텐츠 요약:
{prompt_text}
"""

    print("\n---🔍 Ollama Prompt Input ---\n", full_prompt)

    template = PromptTemplate.from_template("{prompt_text}")
    llm = Ollama(model="llama3")
    chain = LLMChain(prompt=template, llm=llm)
    return chain.run(prompt_text=full_prompt)

# ✅ Streamlit UI 구성 시작
st.set_page_config(page_title="AI 광고 전략 분석기", layout="wide")
st.title("🎯 시온마케팅 콘텐츠 분석 시스템")

prompt_text = st.text_area("분석 프롬프트", "광고 콘텐츠가 업종·타겟·전환 전략 측면에서 실무에 적합한지 정밀 분석하고, 구체적인 마케팅 개선안을 3가지 이상 제시해 주세요.")
