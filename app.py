import os
import io
import re
import json
import time
import streamlit as st
from dotenv import load_dotenv

# --- IMPORTS FOR PDF/DOCX & REPORTLAB ---
from pypdf import PdfReader
import docx
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- GEMINI SDK ---
from google import genai
from google.genai import types

load_dotenv()

# ==========================================
# 1. PAGE & SESSION SETUP
# ==========================================
st.set_page_config(page_title="AI Resume Assistant", page_icon="📄", layout="wide")

if "resume_data" not in st.session_state:
    st.session_state.resume_data = {
        "personal": {"full_name": "", "title": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": ""},
        "summary": "",
        "experience": [],
        "education": [],
        "skills": {"technical": "", "programming": "", "frameworks": "", "soft": ""},
        "certifications": []
    }
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None

# API Key Retrieval (Handles Streamlit Cloud Secrets and Local .env)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key) if api_key else None

# ==========================================
# 2. HELPER FUNCTIONS & GEMINI CALLS
# ==========================================
def parse_json_safely(text: str) -> dict:
    """Safely extract JSON from Gemini text response."""
    try:
        clean = re.sub(r"```json\s*", "", text)
        clean = re.sub(r"```\s*$", "", clean).strip()
        return json.loads(clean)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("Failed to parse valid JSON from Gemini output.")

def call_gemini(prompt: str) -> str:
    """Call Gemini API with model fallback and automatic retry on 503 high demand."""
    if not client:
        raise ValueError("GEMINI_API_KEY is not configured.")
    
    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite"
    ]
    
    last_error = None
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                return res.text
            except Exception as e:
                last_error = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                break
                    
    raise Exception(f"All model attempts failed. Last error: {str(last_error)}")

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_docx_text(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def parse_resume_with_ai(raw_text: str) -> dict:
    """Parses raw text from PDF/DOCX into structured JSON."""
    prompt = f"""
    You are an expert resume parser. Convert the raw resume text into a clean, structured JSON format.
    Schema required:
    {{
        "personal": {{ "full_name": "", "title": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": "" }},
        "summary": "",
        "experience": [{{ "title": "", "company": "", "start": "", "end": "", "description": "" }}],
        "skills": {{ "technical": "", "programming": "", "frameworks": "", "soft": "" }},
        "education": [{{ "degree": "", "institution": "", "year": "" }}]
    }}

    Raw Resume Text:
    {raw_text}

    Return ONLY valid JSON.
    """
    raw_response = call_gemini(prompt)
    return parse_json_safely(raw_response)

def rewrite_complete_resume_with_ai(current_resume: dict, job_description: str) -> dict:
    """Completely rewrites and optimizes every section of the resume."""
    prompt = f"""
    You are an elite executive resume writer and ATS optimization specialist.
    Analyze the uploaded resume against the target job description.
    REWRITE the ENTIRE resume to fix errors, improve clarity, enhance professional tone, and align achievements with job requirements.

    CRITICAL REWRITE INSTRUCTIONS:
    1. REWRITE EVERY EXPERIENCE ENTRY: Rephrase bullet points into high-impact statements using action verbs and quantified achievements where applicable.
    2. OPTIMIZE SUMMARY: Write a compelling 3-4 sentence professional summary tailored explicitly to the target job description.
    3. SKILLS HARVESTING: Include relevant hard skills, technical tools, and soft skills aligned with the job posting.
    4. ACCURACY: Preserve genuine dates, company names, job titles, and degree names.

    Current Resume JSON:
    {json.dumps(current_resume)}

    Target Job Description:
    {job_description if job_description else "Optimize for general industry best practices and ATS compliance."}

    Return ONLY a single valid JSON matching the exact input structure:
    {{
        "personal": {{ "full_name": "", "title": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": "" }},
        "summary": "",
        "experience": [{{ "title": "", "company": "", "start": "", "end": "", "description": "" }}],
        "skills": {{ "technical": "", "programming": "", "frameworks": "", "soft": "" }},
        "education": [{{ "degree": "", "institution": "", "year": "" }}]
    }}
    """
    raw_response = call_gemini(prompt)
    return parse_json_safely(raw_response)

def generate_pdf(data: dict, template_name: str) -> bytes:
    """Generates an ATS-compliant PDF using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    primary_color = colors.HexColor('#004080') if template_name == "Modern" else (colors.HexColor('#005F73') if template_name == "Technical" else colors.HexColor('#111111'))
    
    name_style = ParagraphStyle('Name', parent=styles['Heading1'], fontSize=20, leading=24, textColor=primary_color)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.HexColor('#222222'))
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=11, leading=14, textColor=primary_color, spaceBefore=8, spaceAfter=4)

    # Personal Details
    p = data.get("personal", {})
    story.append(Paragraph(p.get("full_name", "Your Name").upper(), name_style))
    meta = " | ".join([b for b in [p.get("email"), p.get("phone"), p.get("location"), p.get("linkedin"), p.get("github")] if b])
    if meta:
        story.append(Paragraph(meta, body_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=6))

    # Summary
    if data.get("summary"):
        story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
        story.append(Paragraph(data["summary"], body_style))

    # Experience
    if data.get("experience"):
        story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
        for exp in data["experience"]:
            header = f"<b>{exp.get('title')}</b> — {exp.get('company')} <i>({exp.get('start')} - {exp.get('end')})</i>"
            story.append(Paragraph(header, body_style))
            if exp.get("description"):
                for bullet in exp["description"].split("\n"):
                    if bullet.strip():
                        clean_bullet = bullet.strip().lstrip("•- ")
                        story.append(Paragraph(f"• {clean_bullet}", body_style))
            story.append(Spacer(1, 4))

    # Education
    if data.get("education"):
        story.append(Paragraph("EDUCATION", section_style))
        for edu in data["education"]:
            story.append(Paragraph(f"<b>{edu.get('degree')}</b>, {edu.get('institution')} ({edu.get('year')})", body_style))

    # Skills
    skills_list = [f"<b>{k.title()}:</b> {v}" for k, v in data.get("skills", {}).items() if v]
    if skills_list:
        story.append(Paragraph("SKILLS & COMPETENCIES", section_style))
        story.append(Paragraph("<br/>".join(skills_list), body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("📄 AI Resume Assistant")
st.sidebar.caption("Parse. Analyze. Rewrite. Download.")

nav = st.sidebar.radio("Navigation", [
    "🏠 Home", 
    "📤 Upload & Parse Resume", 
    "🎯 Target Job", 
    "📊 Analyze & Rewrite", 
    "📝 Edit Structured Data", 
    "🎨 Template & PDF Download"
])

if not client:
    st.sidebar.warning("⚠️ GEMINI_API_KEY missing. Add it to Streamlit Secrets or .env.")

# ==========================================
# 4. PAGE CONTROLLERS
# ==========================================
if nav == "🏠 Home":
    st.title("📄 AI Resume Assistant & Optimization Engine")
    st.subheader("Transform Your Resume with AI")
    st.write("Upload an existing resume, target a job description, let AI rewrite the full resume for high ATS matching, and download your newly formatted PDF.")

elif nav == "📤 Upload & Parse Resume":
    st.title("📤 Upload Existing Resume")
    uploaded_file = st.file_uploader("Upload PDF or DOCX file", type=["pdf", "docx"])
    
    if uploaded_file and st.button("Extract & Parse with AI"):
        if not client:
            st.error("API Key is missing.")
        else:
            with st.spinner("Extracting text and converting to structured format..."):
                try:
                    bytes_data = uploaded_file.read()
                    raw_text = extract_pdf_text(bytes_data) if uploaded_file.name.endswith(".pdf") else extract_docx_text(bytes_data)
                    parsed_resume = parse_resume_with_ai(raw_text)
                    st.session_state.resume_data = parsed_resume
                    st.success("Resume parsed successfully! Go to 'Analyze & Rewrite' or inspect in 'Edit Structured Data'.")
                except Exception as e:
                    st.error(f"Failed to parse resume: {str(e)}")

elif nav == "🎯 Target Job":
    st.title("🎯 Target Job Description")
    st.session_state.job_description = st.text_area("Paste Target Job Posting Here:", value=st.session_state.job_description, height=250)
    st.info("Adding a target job description enables Gemini to tailor bullet points, keywords, and summary directly to the job.")

elif nav == "📊 Analyze & Rewrite":
    st.title("📊 AI Resume Analysis & Full Rewrite")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("1. Analyze Match Score"):
            if client:
                with st.spinner("Analyzing resume against job..."):
                    try:
                        prompt = f"Analyze this resume against target job.\nResume: {st.session_state.resume_data}\nJob: {st.session_state.job_description}\nReturn JSON with keys: overall_score, strengths, weaknesses"
                        raw = call_gemini(prompt)
                        st.session_state.resume_analysis = parse_json_safely(raw)
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"API Error: {str(e)}")
            else:
                st.error("API Key missing.")

    with col2:
        if st.button("2. ⚡ Rewrite Entire Resume with AI"):
            if client:
                with st.spinner("Rewriting experience, bullet points, summary, and skills..."):
                    try:
                        rewritten_data = rewrite_complete_resume_with_ai(
                            st.session_state.resume_data, 
                            st.session_state.job_description
                        )
                        st.session_state.resume_data = rewritten_data
                        st.success("Entire resume successfully rewritten and updated!")
                    except Exception as e:
                        st.error(f"Rewrite failed: {str(e)}")
            else:
                st.error("API Key missing.")

    if st.session_state.resume_analysis:
        st.divider()
        ra = st.session_state.resume_analysis
        st.metric("Overall Match Score", f"{ra.get('overall_score', 0)}/100")
        st.write("**Strengths:**", ra.get("strengths", []))
        st.write("**Missing Keywords / Weaknesses:**", ra.get("weaknesses", []))

elif nav == "📝 Edit Structured Data":
    st.title("📝 Edit Resume Content")
    st.write("You can inspect or manually tweak the AI-generated text before generating your PDF.")
    
    p = st.session_state.resume_data["personal"]
    with st.expander("👤 Personal Information", expanded=True):
        c1, c2 = st.columns(2)
        p["full_name"] = c1.text_input("Full Name", p.get("full_name"))
        p["title"] = c2.text_input("Professional Title", p.get("title"))
        p["email"] = c1.text_input("Email", p.get("email"))
        p["phone"] = c2.text_input("Phone", p.get("phone"))
        p["location"] = c1.text_input("Location", p.get("location"))

    with st.expander("📑 Professional Summary", expanded=True):
        st.session_state.resume_data["summary"] = st.text_area("Summary", st.session_state.resume_data.get("summary"), height=120)

    with st.expander("💼 Experience"):
        for idx, exp in enumerate(st.session_state.resume_data.get("experience", [])):
            st.markdown(f"**Job #{idx+1}**")
            c1, c2 = st.columns(2)
            exp["title"] = c1.text_input(f"Title #{idx+1}", exp.get("title"))
            exp["company"] = c2.text_input(f"Company #{idx+1}", exp.get("company"))
            exp["description"] = st.text_area(f"Bullet Points #{idx+1}", exp.get("description"), height=120)

elif nav == "🎨 Template & PDF Download":
    st.title("🎨 Generate & Download PDF")
    
    selected_template = st.radio("Choose PDF Style", ["Modern", "Classic", "Technical"], horizontal=True)
    
    if st.button("Generate Downloadable PDF"):
        pdf_bytes = generate_pdf(st.session_state.resume_data, selected_template)
        st.success("PDF created successfully!")
        st.download_button(
            label="📄 Download Rewritten Resume.pdf", 
            data=pdf_bytes, 
            file_name="Rewritten_Resume.pdf", 
            mime="application/pdf"
        )
