
import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------
# 1. PROJECT PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "raw" / "customer_reviews.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. INITIALIZE SENTIMENT ANALYZER
# --------------------------------

analyzer = SentimentIntensityAnalyzer()

# --------------------------------
# 3. DEFINE ASPECT KEYWORDS
# --------------------------------

aspect_keywords = {
    "Product Quality": [
        "quality", "build", "durable", "cheap",
        "fragile", "premium"
    ],
    "Delivery": [
        "delivery", "delayed", "arrived",
        "package", "packaging"
    ],
    "Price": [
        "price", "priced", "overpriced",
        "money", "cost", "worth", "value"
    ],
    "Customer Support": [
        "support", "assistance", "helpful",
        "complaint", "issue", "problem"
    ],
    "Performance": [
        "performance", "works", "working",
        "slow", "freezes", "stops",
        "reliable", "smooth"
    ],
    "Battery": [
        "battery", "charging", "charge",
        "drains", "lasts"
    ]
}

# --------------------------------
# 4. SENTIMENT CLASSIFICATION
# --------------------------------

def classify_sentiment(text):

    scores = analyzer.polarity_scores(str(text))

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"

    elif compound <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment, compound


# --------------------------------
# 5. ASPECT EXTRACTION
# --------------------------------

def find_aspects(text):

    text = str(text).lower()

    detected = []

    for aspect, keywords in aspect_keywords.items():

        for keyword in keywords:

            if re.search(
                r"\b" + re.escape(keyword) + r"\b",
                text
            ):
                detected.append(aspect)
                break

    return detected


# --------------------------------
# 6. ASPECT-LEVEL SENTIMENT
# --------------------------------

def analyze_aspects(text):

    text = str(text)

    clauses = re.split(
        r"\bbut\b|\bhowever\b|\balthough\b|;",
        text,
        flags=re.IGNORECASE
    )

    aspect_results = {}

    for clause in clauses:

        detected = find_aspects(clause)

        if not detected:
            continue

        sentiment, score = classify_sentiment(clause)

        for aspect in detected:

            # Preserve every observation
            aspect_results.setdefault(aspect, []).append({
                "sentiment": sentiment,
                "score": score,
                "clause": clause.strip()
            })

    return aspect_results



# --------------------------------
# 7. LOAD DATASET
# --------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 55)
print("          FEEDLYTICS - NLP ANALYSIS")
print("=" * 55)

# --------------------------------
# 8. APPLY SENTIMENT ANALYSIS
# --------------------------------

results = df["review_text"].apply(classify_sentiment)

df["predicted_sentiment"] = results.apply(
    lambda x: x[0]
)

df["sentiment_score"] = results.apply(
    lambda x: x[1]
)

# --------------------------------
# 9. APPLY ASPECT EXTRACTION
# --------------------------------

df["detected_aspects"] = df["review_text"].apply(
    find_aspects
)

df["aspect_sentiments"] = df["review_text"].apply(
    analyze_aspects
)

# --------------------------------
# 10. IDENTIFY MIXED FEEDBACK
# --------------------------------


def identify_mixed_feedback(aspect_results):

    all_sentiments = []

    for observations in aspect_results.values():

        for item in observations:
            all_sentiments.append(item["sentiment"])

    if (
        "Positive" in all_sentiments
        and "Negative" in all_sentiments
    ):
        return "Mixed"

    return None



df["mixed_feedback"] = df["aspect_sentiments"].apply(
    identify_mixed_feedback
)

df["final_sentiment"] = df.apply(
    lambda row: (
        "Mixed"
        if row["mixed_feedback"] == "Mixed"
        else row["predicted_sentiment"]
    ),
    axis=1
)

# --------------------------------
# 11. MODEL EVALUATION
# --------------------------------

print("\n1. SENTIMENT DISTRIBUTION")

print(
    df["final_sentiment"].value_counts()
)

print("\n2. MODEL ACCURACY")

accuracy = accuracy_score(
    df["expected_sentiment"],
    df["final_sentiment"]
)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\n3. CLASSIFICATION REPORT")

print(
    classification_report(
        df["expected_sentiment"],
        df["final_sentiment"],
        zero_division=0
    )
)

# --------------------------------
# 12. ASPECT ANALYSIS
# --------------------------------

print("\n4. DETECTED CUSTOMER ASPECTS")

aspect_counts = {}

for aspects in df["detected_aspects"]:

    for aspect in aspects:

        aspect_counts[aspect] = (
            aspect_counts.get(aspect, 0) + 1
        )

aspect_summary = pd.Series(
    aspect_counts,
    name="review_count"
).sort_values(ascending=False)

print(aspect_summary)

# --------------------------------
# 13. SAVE RESULTS
# --------------------------------

output_path = OUTPUT_DIR / "analyzed_reviews.csv"

df.to_csv(output_path, index=False)

aspect_summary.to_csv(
    OUTPUT_DIR / "aspect_detection_summary.csv"
)

print("\n" + "=" * 55)
print("NLP ANALYSIS COMPLETED!")
print(f"Results saved to: {output_path}")
print("=" * 55)
