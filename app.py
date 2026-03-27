"""
Sentiment Analyzer - Streamlit Web Application
Author: Matthew Cromaz (TheCromazone)

A real-time NLP app for sentiment classification using HuggingFace transformers.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import SentimentModel
from utils.preprocessing import clean_text
from utils.visualizations import generate_wordcloud_fig, plot_sentiment_distribution
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str) -> SentimentModel:
    return SentimentModel(model_name)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["DistilBERT (fast)", "RoBERTa (accurate)", "VADER (lexicon)"],
        index=0,
    )
    model_map = {
        "DistilBERT (fast)": "distilbert-base-uncased-finetuned-sst-2-english",
        "RoBERTa (accurate)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "VADER (lexicon)": "vader",
    }
    selected_model = model_map[model_choice]
    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This app uses transformer-based NLP models to classify "
        "sentiment as **positive**, **negative**, or **neutral**."
    )

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🧠 Sentiment Analyzer")
st.markdown("Classify text sentiment instantly using state-of-the-art NLP models.")

tab1, tab2 = st.tabs(["✏️ Single Text", "📂 Batch Analysis"])

# ── Tab 1: Single Text ────────────────────────────────────────────────────────
with tab1:
    user_input = st.text_area(
        "Enter text to analyze:",
        placeholder="e.g. 'This product completely changed my life for the better!'",
        height=150,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    if analyze_btn and user_input.strip():
        model = load_model(selected_model)
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            result = model.predict(cleaned)

        # Display result
        label = result["label"]
        score = result["score"]
        emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}.get(label, "⚪")

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Sentiment", f"{emoji} {label.title()}")
        with col_b:
            st.metric("Confidence", f"{score:.1%}")
        with col_c:
            st.metric("Model", model_choice.split(" ")[0])

        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if label == "POSITIVE" else "#e74c3c"},
                "steps": [
                    {"range": [0, 50], "color": "#f8f9fa"},
                    {"range": [50, 75], "color": "#e9ecef"},
                    {"range": [75, 100], "color": "#dee2e6"},
                ],
                "threshold": {"line": {"color": "gray", "width": 2}, "value": 50},
            },
            title={"text": "Confidence Score"},
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

# ── Tab 2: Batch Analysis ─────────────────────────────────────────────────────
with tab2:
    st.markdown("Upload a CSV with a `text` column to analyze multiple records at once.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must contain a column named `text`.")
        else:
            st.write(f"Loaded **{len(df):,}** rows. Preview:")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Run Batch Analysis", type="primary"):
                model = load_model(selected_model)
                progress = st.progress(0, text="Analyzing texts...")
                results = []

                for i, text in enumerate(df["text"].astype(str)):
                    cleaned = clean_text(text)
                    res = model.predict(cleaned)
                    results.append(res)
                    progress.progress((i + 1) / len(df))

                df["sentiment"] = [r["label"] for r in results]
                df["confidence"] = [r["score"] for r in results]
                progress.empty()

                st.success(f"✅ Analyzed {len(df):,} texts!")

                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        plot_sentiment_distribution(df["sentiment"]),
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        generate_wordcloud_fig(df["text"].tolist()),
                        use_container_width=True,
                    )

                # Download
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out,
                    "sentiment_results.csv",
                    "text/csv",
                    use_container_width=True,
                )
