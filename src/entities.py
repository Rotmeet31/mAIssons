"""
Named-entity extraction using spaCy.

Run once after install:
    python -m spacy download en_core_web_sm
"""
from loguru import logger

# Entity types worth indexing for news search
_KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT"}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> list[dict]:
    """
    Return a deduplicated list of named entities from text.
    Each entry: { text, normalized, label }
    Returns [] if spaCy is not installed or model is missing.
    """
    try:
        nlp = _get_nlp()
    except Exception as exc:
        logger.warning("spaCy unavailable — skipping entity extraction: {}", exc)
        return []

    doc = nlp(text[:5000])  # cap for speed; news articles rarely need more

    seen: set[tuple[str, str]] = set()
    result: list[dict] = []

    for ent in doc.ents:
        if ent.label_ not in _KEEP_LABELS:
            continue
        text_clean = ent.text.strip()
        if len(text_clean) < 2:
            continue
        normalized = text_clean.lower()
        key = (normalized, ent.label_)
        if key in seen:
            continue
        seen.add(key)
        result.append({"text": text_clean, "normalized": normalized, "label": ent.label_})

    return result
