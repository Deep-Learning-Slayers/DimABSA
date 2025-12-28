import json
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import spacy
from transformers import pipeline

# ===============================
# CONFIG
# ===============================

STAR_TO_VALENCE = {
    5: (8.0, 9.0),
    4: (7.0, 7.9),
    3: (5.0, 6.9),
    2: (3.0, 4.9),
    1: (1.0, 2.9),
}

ASPECT_CATEGORY_MAP = {
    "food": "RESTAURANT#FOOD",
    "meal": "RESTAURANT#FOOD",
    "dish": "RESTAURANT#FOOD",
    "bowl": "RESTAURANT#FOOD",
    "steak": "RESTAURANT#FOOD",
    "chicken": "RESTAURANT#FOOD",

    "service": "RESTAURANT#SERVICE",
    "staff": "RESTAURANT#SERVICE",
    "waiter": "RESTAURANT#SERVICE",
    "lady": "RESTAURANT#SERVICE",

    "place": "RESTAURANT#AMBIENCE",
    "restaurant": "RESTAURANT#AMBIENCE",
    "store": "RESTAURANT#AMBIENCE",
}

STOP_ASPECTS = {"thing", "something", "anything", "everything", "time", "day"}

# ===============================
# LOAD MODELS
# ===============================

def load_models(nlp_model: str):
    nlp = spacy.load(nlp_model)

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
    )

    return nlp, sentiment_pipe


# ===============================
# UTILITIES
# ===============================

def sample_valence(stars: int) -> float:
    low, high = STAR_TO_VALENCE.get(stars, (5.0, 5.0))
    return round(random.uniform(low, high), 2)


def estimate_arousal(text: str, sentiment_label: str) -> float:
    exclam = text.count("!")
    caps = sum(1 for w in text.split() if w.isupper() and len(w) > 2)

    base = 5.0
    if sentiment_label == "negative":
        base += 1.2
    elif sentiment_label == "positive":
        base += 0.6

    arousal = base + exclam * 0.6 + caps * 0.3
    return round(max(1.0, min(9.0, arousal)), 2)


def get_category(aspect: str) -> str:
    for k, v in ASPECT_CATEGORY_MAP.items():
        if k in aspect:
            return v
    return "RESTAURANT#GENERAL"


# ===============================
# ASPECT + OPINION EXTRACTION
# ===============================

def extract_aspect_opinion(doc) -> List[Tuple[str, str]]:
    pairs = []

    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in STOP_ASPECTS:
            aspect = token.lemma_.lower()

            # adjectival modifier
            for child in token.children:
                if child.dep_ == "amod":
                    pairs.append((aspect, child.lemma_.lower()))

            # copular adjective (service is bad)
            if token.dep_ == "nsubj":
                for head in [token.head]:
                    if head.pos_ == "ADJ":
                        pairs.append((aspect, head.lemma_.lower()))

    return list(set(pairs))


# ===============================
# MAIN CONVERTER
# ===============================

def convert_csv_to_dimabsa_restaurant(
    input_csv: Path,
    output_jsonl: Path,
    text_col: str = "review_text",
    star_col: str = "star_rating",
    nlp_model: str = "en_core_web_sm",
):
    df = pd.read_csv(input_csv)

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found")

    nlp, sentiment_pipe = load_models(nlp_model)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for idx, row in df.iterrows():
            text = str(row[text_col])
            stars = int(row[star_col]) if star_col in df.columns else 3

            doc = nlp(text)
            pairs = extract_aspect_opinion(doc)

            if not pairs:
                continue

            quads = []
            seen = set()

            for aspect, opinion in pairs:
                key = (aspect, opinion)
                if key in seen:
                    continue
                seen.add(key)

                sent = sentiment_pipe(opinion)[0]
                label = sent["label"].lower()

                val = sample_valence(stars)
                aro = estimate_arousal(text, label)

                quad = {
                    "Aspect": aspect,
                    "Category": get_category(aspect),
                    "Opinion": opinion,
                    "VA": f"{val:.2f}#{aro:.2f}"
                }
                quads.append(quad)

            if not quads:
                continue

            entry = {
                "ID": f"restaurant_{idx}",
                "Text": text,
                "Quadruplet": quads
            }

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote DimABSA restaurant JSONL to {output_jsonl}")
