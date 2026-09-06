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
        "personal": {"full_name": "", "title": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": "", "portfolio": ""},
        "summary": "",
        "experience": [],
        "education": [],
        "skills": {"technical": "", "programming": "", "cybersecurity": "", "frameworks": "", "soft": "", "other": ""},
        "projects": [],
        "certifications": []
    }
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "job_analysis" not in st.session_state:
    st.session_state.job_analysis = None
if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None
if "selected_template" not in st.session_state:
    st.session_state.selected_template = "Modern"
if "generated_pdf" not in st.session_state:
    st.session_state.generated_pdf = None

# API Key Retrieval (Handles Streamlit Cloud Secrets and Local .env)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key) if api_key else None

# ==========================================
# 2. HELPER FUNCTIONS & GEMINI INTEGRATION
# ==========================================
def parse_json_safely(text: str) -> dict:
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
    if not client:
        raise ValueError("GEMINI_API_KEY is not configured.")
    
    # Priority list of models to fall back on in case of deprecation or high demand
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]
    
    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                return res.text
            except Exception as e:
                last_error = e
                # Retry if service is temporarily unavailable
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(1.5)
                    continue
                break
                    
    raise Exception(f"All model attempts failed. Last error: {str(last_error)}")

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_docx_text(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def generate_pdf(data: dict, template_name: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    primary_color = colors.HexColor('#004080') if template_name == "Modern" else (colors.HexColor('#005F73') if template_name == "Technical" else colors.HexColor('#111111'))
    
    name_style = ParagraphStyle('Name', parent=styles['Heading1'], fontSize=20, leading=24, textColor=primary_color)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.HexColor('#222222'))
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=12, leading=15, textColor=primary_color, spaceBefore=10, spaceAfter=4)

    p = data.get("personal", {})
    story.append(Paragraph(p.get("full_name", "Your Name").upper(), name_style))
    meta = " | ".join([b for b in [p.get("email"), p.get("phone"), p.get("location"), p.get("linkedin"), p.get("github")] if b])
    if meta:
        story.append(Paragraph(meta, body_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=8))

    if data.get("summary"):
        story.append(Paragraph("SUMMARY", section_style))
        story.append(Paragraph(data["summary"], body_style))

    if data.get("experience"):
        story.append(Paragraph("EXPERIENCE", section_style))
        for exp in data["experience"]:
            story.append(Paragraph(f"<b>{exp.get('title')}</b> - {exp.get('company')} ({exp.get('start')} - {exp.get('end')})", body_style))
            if exp.get("description"):
                for bullet in exp["description"].split("\n"):
                    if bullet.strip():
                        story.append(Paragraph(f"• {bullet.strip()}", body_style))

    skills = [v for v in data.get("skills", {}).values() if v]
    if skills:
        story.append(Paragraph("SKILLS", section_style))
        story.append(Paragraph(", ".join(skills), body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("📄 AI Resume Assistant")
st.sidebar.caption("Create. Improve. Optimize. Get Hired.")

nav = st.sidebar.radio("Navigation", [
    "🏠 Home", "📝 Resume Builder", "📤 Upload Resume", 
    "🎯 Target Job", "📊 Resume Analysis", "🎨 Resume Template", 
    "👁 Resume Preview", "📄 Download PDF"
])

if not client:
    st.sidebar.warning("⚠️ GEMINI_API_KEY missing. Add it to Streamlit Secrets or .env.")

# ==========================================
# 4. PAGE CONTROLLERS
# ==========================================
if nav == "🏠 Home":
    st.title("📄 AI Resume Assistant")
    st.subheader("Create a Resume That Gets Noticed")
    st.write("Use AI to create, analyze, improve, and optimize your resume for your target job.")
    c1, c2 = st.columns(2)
    with c1: st.info("**🤖 AI Analysis:** Get instant feedback on your resume strengths and weaknesses.")
    with c2: st.info("**🎯 Job Match:** Tailor your experience to pass ATS screeners.")

elif nav == "📝 Resume Builder":
    st.title("📝 Resume Builder")
    p = st.session_state.resume_data["personal"]
    with st.expander("👤 Personal Information", expanded=True):
        c1, c2 = st.columns(2)
        p["full_name"] = c1.text_input("Full Name", p.get("full_name"))
        p["title"] = c2.text_input("Professional Title", p.get("title"))
        p["email"] = c1.text_input("Email", p.get("email"))
        p["phone"] = c2.text_input("Phone", p.get("phone"))
        p["location"] = c1.text_input("Location", p.get("location"))

    with st.expander("📑 Summary", expanded=True):
        st.session_state.resume_data["summary"] = st.text_area("Professional Summary", st.session_state.resume_data.get("summary"))

    with st.expander("💼 Experience"):
        if st.button("➕ Add Job"):
            st.session_state.resume_data["experience"].append({"title": "", "company": "", "start": "", "end": "", "description": ""})
            st.rerun()
        for idx, exp in enumerate(st.session_state.resume_data["experience"]):
            c1, c2 = st.columns(2)
            exp["title"] = c1.text_input(f"Job Title #{idx+1}", exp.get("title"))
            exp["company"] = c2.text_input(f"Company #{idx+1}", exp.get("company"))
            exp["description"] = st.text_area(f"Description / Accomplishments #{idx+1}", exp.get("description"))

elif nav == "📤 Upload Resume":
    st.title("📤 Upload Resume")
    uploaded_file = st.file_uploader("Upload existing PDF or DOCX", type=["pdf", "docx"])
    if uploaded_file and st.button("Extract Text"):
        bytes_data = uploaded_file.read()
        raw_text = extract_pdf_text(bytes_data) if uploaded_file.name.endswith(".pdf") else extract_docx_text(bytes_data)
        st.session_state.resume_data["summary"] = raw_text[:500]
        st.success("Extracted text populated into Summary!")

elif nav == "🎯 Target Job":
    st.title("🎯 Target Job Description")
    st.session_state.job_description = st.text_area("Paste target job posting here:", st.session_state.job_description, height=200)

elif nav == "📊 Resume Analysis":
    st.title("📊 AI Resume Analysis")
    if st.button("Analyze with Gemini"):
        if client:
            with st.spinner("Analyzing resume against job requirements..."):
                try:
                    prompt = f"Analyze this resume against the target job description.\nResume: {st.session_state.resume_data}\nJob: {st.session_state.job_description}\nReturn JSON with keys: overall_score, strengths, weaknesses"
                    raw = call_gemini(prompt)
                    st.session_state.resume_analysis = parse_json_safely(raw)
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"⚠️ API Error: {str(e)}")
        else:
            st.error("API Key missing. Please check Secrets in Streamlit Cloud.")
    
    if st.session_state.resume_analysis:
        ra = st.session_state.resume_analysis
        st.metric("Overall Match Score", f"{ra.get('overall_score', 0)}/100")
        st.write("**Strengths:**", ra.get("strengths", []))
        st.write("**Weaknesses / Missing Keywords:**", ra.get("weaknesses", []))

elif nav == "🎨 Resume Template":
    st.title("🎨 Select Resume Template")
    st.session_state.selected_template = st.radio("Choose Style", ["Modern", "Classic", "Technical"])

elif nav == "👁 Resume Preview":
    st.title("👁 Resume Preview")
    st.json(st.session_state.resume_data)

elif nav == "📄 Download PDF":
    st.title("📄 Download PDF")
    if st.button("Generate PDF"):
        pdf_bytes = generate_pdf(st.session_state.resume_data, st.session_state.selected_template)
        st.session_state.generated_pdf = pdf_bytes
        st.success("PDF generated successfully!")

    if st.session_state.generated_pdf:
        st.download_button("📄 Download Resume.pdf", st.session_state.generated_pdf, "Resume.pdf", "application/pdf")
