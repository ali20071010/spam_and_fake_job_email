import streamlit as st
import requests

st.set_page_config(
    page_title="AI Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.success-box {
    background-color: #d4edda;
    color: #155724;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #28a745;
    font-size: 20px;
    font-weight: bold;
}

.error-box {
    background-color: #f8d7da;
    color: #721c24;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dc3545;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Detection System")
st.write("Detect spam emails and fake job postings using Machine Learning.")

email_tab, job_tab = st.tabs([
    "📧 Email Detection",
    "💼 Fake Job Detection"
])

with email_tab:
    st.header("Email Spam Detection")
    st.write("Paste the email text below to check if it is spam or not.")

    email_text = st.text_area(
        "Email Text",
        placeholder="Paste the email text here...",
        height=250,
        key="email_input"
    )

    if st.button("Analyze Email", use_container_width=True):
        if email_text.strip():
            response = requests.post(
                "http://127.0.0.1:8000/predict/spam",
                json={"text": email_text}
            )
            result = response.json()["result"]

            if result == "ham":
                st.markdown(
                    '<div class="success-box">✅ Ham — This email is real.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="error-box">❌ Spam — This email appears to be spam.</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter the email text first.")

with job_tab:
    st.header("Fake Job Detection")
    st.write("Paste the job description below to check if it is real or fake.")

    job_text = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        height=250,
        key="job_input"
    )

    if st.button("Analyze Job", use_container_width=True):
        if job_text.strip():
            response = requests.post(
                "http://127.0.0.1:8000/predict/job",
                json={"text": job_text}
            )
            result = response.json()["result"]

            if result == "real":
                st.markdown(
                    '<div class="success-box">✅ Real — This job appears to be real.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="error-box">❌ Fake Job — This job appears to be fake.</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please enter the job description first.")

st.divider()
st.caption("AI Detection System | Machine Learning Project")