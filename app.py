import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)

# ============================================================
# LOAD BERT MODEL
# ============================================================

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('rakhikumari/quora-duplicate-bert')
    model     = BertForSequenceClassification.from_pretrained('rakhikumari/quora-duplicate-bert')
    model.eval()
    return tokenizer, model

with st.spinner("⏳ Loading BERT model... (first time only)"):
    tokenizer, model = load_model()

st.success("✅ Model loaded!")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(q1, q2):
    inputs = tokenizer(
        q1, q2,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0].tolist()
        pred    = torch.argmax(outputs.logits, dim=1).item()
    return pred, probs

# ============================================================
# MAIN UI
# ============================================================

st.title("🔍 Quora Duplicate Question Detector")
st.write("Check if two questions are asking the same thing!")
st.write("---")

q1 = st.text_input("Enter Question 1", placeholder="e.g. What is Python?")
q2 = st.text_input("Enter Question 2", placeholder="e.g. What is Python programming language?")

st.write("")

if st.button("Check Duplicate", use_container_width=True):
    if q1 and q2:
        with st.spinner("Analyzing questions..."):
            pred, probs = predict(q1, q2)

        st.write("---")

        if pred == 1:
            st.success("✅ Duplicate Questions!")
            st.metric("Confidence", f"{round(probs[1]*100, 2)}%")
        else:
            st.error("❌ Not Duplicate!")
            st.metric("Confidence", f"{round(probs[0]*100, 2)}%")

        st.write("Duplicate Probability:")
        st.progress(float(probs[1]))

        # show details
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Question 1:**\n{q1}")
        with col2:
            st.info(f"**Question 2:**\n{q2}")

    else:
        st.warning("⚠️ Please enter both questions!")

st.write("---")

# example questions
st.subheader("📝 Try these examples:")

col1, col2 = st.columns(2)

with col1:
    st.write("**Duplicate ✅**")
    st.info("Q1: What is the best way to learn Python?\nQ2: What is the best method to learn Python?")

with col2:
    st.write("**Not Duplicate ❌**")
    st.info("Q1: What is Python?\nQ2: How do I make pizza at home?")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app uses **BERT** to detect if two questions 
are asking the same thing.
""")

st.sidebar.write("---")
st.sidebar.write("**Model Results:**")
st.sidebar.table({
    "Model":    ["XGBoost",  "LightGBM", "BERT"],
    "Accuracy": ["81.42%",   "81.77%",   "87.73%"]
})

st.sidebar.write("---")
st.sidebar.write("**Tech Stack:**")
st.sidebar.write("""
- 🤗 BERT (HuggingFace)
- ⚡ PyTorch
- 🎯 Streamlit
""")

st.sidebar.write("---")
st.sidebar.write("**Dataset:**")
st.sidebar.write("""
- Quora Question Pairs
- 400k question pairs
- 37% duplicate
""")

st.sidebar.write("---")
st.sidebar.write("Made with ❤️ by Rakhi")