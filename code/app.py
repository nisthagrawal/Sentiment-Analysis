import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from predict import predict


st.set_page_config(page_title="AI Review Intelligence", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }

.stTextArea textarea {
    background-color: #1c1f26 !important;
    color: white !important;
    font-size: 18px !important;
    caret-color: white !important;
}

.stButton>button {
    background-color: #00ADB5;
    color: white;
    font-size: 18px;
    padding: 10px 20px;
    border-radius: 10px;
}

button[data-baseweb="tab"] {
    font-size: 20px !important;
    font-weight: bold !important;
    padding: 12px 22px !important;
    color: white !important;
}

.big-box {
    padding: 16px;
    border-radius: 10px;
    font-size: 22px;
    margin-bottom: 12px;
}

.input-box { background-color: #1c2e4a; }
.success-box { background-color: #14532d; color: #00ff9f; }
.error-box { background-color: #5c1a1a; color: #ff6b6b; }
.info-box { background-color: #1c2e4a; color: #9ecbff; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Review Intelligence")
st.markdown("### Sentiment + Emotion Dashboard")

# ---------------- SMART FIX ----------------
def adjust_prediction(text, label, prob):
    text = text.lower()

    negative_keywords = [
        "hate", "worst", "bad", "terrible",
        "sad", "disappointed", "not happy", "angry"
    ]

    for word in negative_keywords:
        if word in text and prob < 0.75:
            return "Negative", max(1 - prob, 0.7)

    return label, prob

# ---------------- EMOTION ----------------
def detect_emotion(text):
    text = text.lower()

    emotion_dict = {
        "Joy 😊": ["love", "amazing", "great", "happy", "good"],
        "Anger 😡": ["hate", "worst", "angry", "frustrating"],
        "Sad 😢": ["sad", "disappointed", "unhappy"],
        "Fear 😨": ["afraid"],
        "Surprise 😲": ["wow"],
        "Disgust 🤢": ["gross"],
        "Bored 😴": ["boring"]
    }

    scores = {k: 0 for k in emotion_dict}

    for emo, words in emotion_dict.items():
        for w in words:
            if w in text:
                scores[emo] += 1

    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Neutral 😐"

# ---------------- STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔍 Analyze", "📊 Insights"])

# ================= ANALYZE =================
with tab1:

    text = st.text_area("Enter Review", height=160)

    if st.button("Analyze Review"):
        if text.strip() == "":
            st.warning("Enter text")
        else:
            try:
                result = predict(text)
                if "error" in result:
                    raise ValueError(result["error"])

                label, prob = adjust_prediction(
                    text,
                    result["label"],
                    result["probability"]
                )

                emotion = detect_emotion(text)

                st.session_state.history.append({
                    "text": text,
                    "label": label,
                    "confidence": prob,
                    "emotion": emotion
                })

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### 📝 Input")
                    st.markdown(f'<div class="big-box input-box">{text}</div>', unsafe_allow_html=True)

                    st.markdown("### 🎯 Prediction")
                    if label == "Positive":
                        st.markdown(f'<div class="big-box success-box">{label} ({prob:.2f})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="big-box error-box">{label} ({prob:.2f})</div>', unsafe_allow_html=True)

                    st.markdown("### 😊 Emotion")
                    st.markdown(f'<div class="big-box info-box">{emotion}</div>', unsafe_allow_html=True)

                    st.markdown("### 🧾 Summary")
                    summary = " ".join(text.split()[:15]) + "..."
                    st.markdown(f'<div class="big-box input-box">{summary}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown("### 📊 Confidence")

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={'text': "Confidence (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "green" if label == "Positive" else "red"}
                        }
                    ))

                    fig.update_layout(
                        height=320,
                        width=320,
                        paper_bgcolor="#0E1117",
                        font_color="white"
                    )

                    st.plotly_chart(fig)

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

# ================= INSIGHTS =================
with tab2:

    if st.session_state.history:

        counts = {}
        for item in st.session_state.history:
            counts[item["emotion"]] = counts.get(item["emotion"], 0) + 1

        col1, col2 = st.columns([1.5, 1])

        # 🔥 FIXED PIE CHART
        with col1:
            st.markdown("### 🎭 Emotion Distribution")

            fig = px.pie(
                names=list(counts.keys()),
                values=list(counts.values()),
                hole=0.3
            )

            fig.update_layout(
                height=500,
                paper_bgcolor="#0E1117",
                font_color="white",
                legend=dict(
                    font=dict(size=18, color="white"),  # 👈 FIXED
                    orientation="v"
                )
            )

            fig.update_traces(
                textfont_size=16,
                textfont_color="white"
            )

            st.plotly_chart(fig, use_container_width=True)

        # SIDE PANEL
        with col2:
            st.markdown("### 🏆 Most Frequent Emotion")
            st.markdown(
                f'<div class="big-box success-box">{max(counts, key=counts.get)}</div>',
                unsafe_allow_html=True
            )

            st.markdown("### 📜 Recent")
            for item in reversed(st.session_state.history[-5:]):
                st.markdown(f"""
                <div class="big-box input-box">
                {item['label']} ({item['confidence']:.2f}) <br>
                {item['emotion']}
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("No data yet")