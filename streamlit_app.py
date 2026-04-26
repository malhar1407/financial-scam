"""Streamlit UI for Financial Scam Detection."""
import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Financial Scam Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Financial Scam Detection System")
st.markdown("**Multi-Modal NLP + GNN Framework** — Real-time fraud detection with explainability")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This system combines:\n"
        "- **NLP**: DistilBERT + LLM intent analysis\n"
        "- **GNN**: Graph Attention Network on transaction patterns\n"
        "- **RAG**: Pattern matching against known scams"
    )

# Initialize session state
if "message" not in st.session_state:
    st.session_state.message = "Your SBI account will be blocked. Update KYC now."

# Main input
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📨 Message to Analyze")
    message = st.text_area(
        "Enter SMS, email, or payment note:",
        value=st.session_state.message,
        height=120,
    )

with col2:
    st.subheader("💳 Transaction Features (optional)")
    amount = st.number_input("Amount (₹)", value=49999.0, step=100.0)
    hour = st.slider("Hour of day", 0, 23, 2)
    day = st.slider("Day of week", 0, 6, 5)
    is_new = st.checkbox("New account", value=True)

# Predict button
if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
    # Update session state with current input
    st.session_state.message = message
    
    with st.spinner("Analyzing..."):
        payload = {
            "message": message,
            "nodes": [{"node_id": 0, "features": [amount, hour, day, int(is_new)]}],
            "edges": [],
            "target_node": 0,
            "explain": False,
        }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Risk score display
            risk = result["risk_score"]
            label = result["label"]
            
            st.markdown("---")
            st.subheader("📊 Detection Result")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                color = "🔴" if label == "SCAM" else "🟢"
                st.metric("Verdict", f"{color} {label}")
            
            with col_b:
                st.metric("Risk Score", f"{risk:.1%}")
            
            with col_c:
                confidence = "High" if risk > 0.8 or risk < 0.2 else "Medium"
                st.metric("Confidence", confidence)
            
            # Progress bar for risk
            st.progress(risk)
            
            # Intent analysis
            st.markdown("---")
            st.subheader("🧠 Intent Analysis")
            intent_data = result["nlp_intent"]
            
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.markdown(f"**Detected Intent:** `{intent_data['intent']}`")
                if intent_data["tactics"]:
                    st.markdown(f"**Tactics:** {', '.join(intent_data['tactics'])}")
            with col_i2:
                st.markdown(f"**LLM Risk Score:** {intent_data['risk_score']:.2f}")
                st.caption(intent_data["reason"])
            
            # RAG matches
            if result["rag_matches"]:
                st.markdown("---")
                st.subheader("🔎 Similar Known Scams")
                for i, match in enumerate(result["rag_matches"][:3], 1):
                    with st.expander(f"Match {i}: {match['label']} (distance: {match['distance']:.2f})"):
                        st.text(match["text"])
            
            # Raw JSON (collapsible)
            with st.expander("🔧 Raw API Response"):
                st.json(result)
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {e}")
            st.info("Make sure the API is running: `uvicorn scam_detection.api.app:app --port 8000`")

# Examples
st.markdown("---")
st.subheader("💡 Try These Examples")

examples = [
    ("KYC Scam", "Your KYC is expired. Update immediately or your account will be blocked."),
    ("Lottery Scam", "Congratulations! You have won Rs 50 lakh lottery. Send OTP to claim prize."),
    ("UPI Phishing", "Your UPI PIN has been compromised. Reset via this link now."),
    ("Benign", "Your transaction of Rs 500 was successful. Thank you for shopping with us."),
]

cols = st.columns(len(examples))
for col, (name, text) in zip(cols, examples):
    with col:
        if st.button(name, use_container_width=True):
            st.session_state.message = text
