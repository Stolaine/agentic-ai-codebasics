"""
Blood Report Analyzer — Streamlit App
======================================
Two-stage LLM pipeline:
  Stage 1 → Extract & classify test values (HIGH / LOW / NORMAL)
  Stage 2 → Generate a health summary + Indian diet plan

Uses the same prompts and model as the companion notebook
(blood_work_analysis.ipynb).
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Load API keys from .env at project root ─────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ── LLM setup (matches notebook) ────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Report Analyzer",
    page_icon="🩸",
    layout="wide",
)

# ── Custom CSS for a premium dark-themed look ────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f87171, #fb923c, #facc15);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: .3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Glass card */
    .glass-card {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin: 1rem 0;
    }
    .glass-card h3 {
        margin-top: 0;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* Column header labels */
    .col-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Divider gradient */
    .grad-divider {
        height: 3px;
        border: none;
        border-radius: 2px;
        background: linear-gradient(90deg, #f87171, #fb923c, #facc15);
        margin: 1.5rem 0;
    }

    /* Subtle animation on result cards */
    .result-card {
        animation: fadeUp 0.45s ease-out both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Placeholder text in right column */
    .placeholder-msg {
        color: #64748b;
        text-align: center;
        padding: 4rem 1rem;
        font-size: 1.05rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #64748b;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🩸 Blood Report Analyzer</h1>
        <p>Paste your blood report on the left and let AI craft a
        personalized health summary &amp; diet plan on the right.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sample report for quick demo ─────────────────────────────────────────
SAMPLE_REPORT = """\
Patient: Rajesh Sharma, Age 48, Male
Date: May 7, 2026

COMPLETE BLOOD COUNT (CBC)
--------------------------
Hemoglobin:        15.1 g/dL        (Normal: 13.5–17.5)
Hematocrit:        44%              (Normal: 41–53%)
WBC:               6.8 x10^3/uL     (Normal: 4.5–11.0)
Platelets:         220 x10^3/uL     (Normal: 150–400)

LIPID PANEL
-----------
Total Cholesterol: 238 mg/dL        (Normal: <200)
LDL Cholesterol:   162 mg/dL        (Normal: <100)
HDL Cholesterol:   36 mg/dL         (Normal: >40)
Triglycerides:     188 mg/dL        (Normal: <150)

METABOLIC PANEL
---------------
Glucose (Fasting): 92 mg/dL         (Normal: 70–99)
HbA1c:             5.3%             (Normal: <5.7%)
Creatinine:        1.0 mg/dL        (Normal: 0.7–1.3)
eGFR:              82 mL/min        (Normal: >60)

LIVER FUNCTION
--------------
ALT:               28 U/L           (Normal: 7–40)
AST:               25 U/L           (Normal: 10–40)
Bilirubin Total:   0.8 mg/dL        (Normal: 0.2–1.2)

Reviewing Physician: Dr. Priya Nair"""

# ── Two-column layout ───────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

# ── LEFT COLUMN: User input ──────────────────────────────────────────────
with col_input:
    st.markdown('<div class="col-header">📋 Blood Report Input</div>', unsafe_allow_html=True)
    use_sample = st.checkbox("Load sample blood report", value=False)
    blood_report = st.text_area(
        "Paste your blood report here",
        value=SAMPLE_REPORT if use_sample else "",
        height=420,
        placeholder="Paste the full text of your blood report…",
        label_visibility="collapsed",
    )
    analyse_clicked = st.button("🔬 Analyse", type="primary", use_container_width=True)

# ── RIGHT COLUMN: Results ────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="col-header">🩺 Health Summary & Diet Plan</div>', unsafe_allow_html=True)

    if analyse_clicked:
        if not blood_report.strip():
            st.warning("Please paste a blood report first.")
            st.stop()

        # ── STAGE 1 (hidden): Extraction ─────────────────────────────────
        with st.spinner("🔍 Extracting & classifying test values…"):
            extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""
            extraction_response = llm.invoke(extraction_prompt)
            extracted_value = extraction_response.text

        # ── STAGE 2: Diet Plan & Health Summary ──────────────────────────
        with st.spinner("🥗 Generating health summary & diet plan…"):
            diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 3 lines explaining the patient's condition in simple language.
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of.

Do not include any other sections in diet plan.

Blood Work Analysis:
{extracted_value}
"""
            diet_response = llm.invoke(diet_prompt)
            diet_text = diet_response.text

        # ── Display results ──────────────────────────────────────────────
        st.markdown('<hr class="grad-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-card result-card">'
            "<h3>🥗 Health Summary & Diet Plan</h3>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(diet_text)
    else:
        st.markdown(
            '<div class="placeholder-msg">'
            "👈 Paste a blood report and click <strong>Analyse</strong> "
            "to see your personalized results here."
            "</div>",
            unsafe_allow_html=True,
        )

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Built with Streamlit &amp; Gemini 2.5 Flash · Not a substitute for professional medical advice</div>',
    unsafe_allow_html=True,
)
