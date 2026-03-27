"""
visualizations.py — Chart helpers for the Streamlit sentiment dashboard.
"""

from typing import List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


SENTIMENT_COLORS = {
    "POSITIVE": "#2ecc71",
    "NEGATIVE": "#e74c3c",
    "NEUTRAL":  "#f39c12",
}


def plot_sentiment_distribution(labels: pd.Series) -> go.Figure:
    """
    Bar chart showing the count of each sentiment label.

    Args:
        labels: Series of sentiment strings ("POSITIVE", "NEGATIVE", "NEUTRAL").

    Returns:
        Plotly Figure.
    """
    counts = labels.value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    counts["Color"] = counts["Sentiment"].map(SENTIMENT_COLORS)

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Number of Texts",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def generate_wordcloud_fig(texts: List[str]) -> go.Figure:
    """
    Generate a word-frequency bar chart as a lightweight word cloud alternative.

    Args:
        texts: List of raw text strings.

    Returns:
        Plotly Figure.
    """
    import re
    from collections import Counter

    STOPWORDS = {
        "the", "a", "an", "is", "it", "in", "on", "and", "or", "to",
        "of", "for", "with", "this", "was", "are", "be", "at", "by",
        "that", "as", "i", "my", "we", "you", "he", "she", "they",
    }

    words = []
    for text in texts:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        words.extend([t for t in tokens if t not in STOPWORDS])

    top_words = Counter(words).most_common(20)
    if not top_words:
        return go.Figure()

    words_list, counts_list = zip(*top_words)
    fig = px.bar(
        x=counts_list,
        y=words_list,
        orientation="h",
        title="Top 20 Most Frequent Words",
        labels={"x": "Frequency", "y": "Word"},
        color=counts_list,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    return fig
