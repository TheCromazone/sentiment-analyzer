"""
preprocessing.py — Text cleaning utilities for the sentiment analyzer.
"""

import re
import html


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text before model inference.

    Steps:
      1. Decode HTML entities  (&amp; → &)
      2. Strip URLs
      3. Remove Twitter-style mentions and hashtags
      4. Collapse excess whitespace
      5. Truncate to 512 characters (transformer limit)

    Args:
        text: Raw input string.

    Returns:
        Cleaned string ready for tokenization.
    """
    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove @mentions and #hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove non-ASCII characters (keep basic punctuation)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to 512 characters
    return text[:512]


def batch_clean(texts: list[str]) -> list[str]:
    """Apply clean_text to a list of strings."""
    return [clean_text(t) for t in texts]
