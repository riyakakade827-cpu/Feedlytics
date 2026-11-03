
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "raw" / "customer_reviews.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("          FEEDLYTICS - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print("-" * 40)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# 2. Missing values
print("\n2. MISSING VALUE ANALYSIS")
print("-" * 40)

missing = df.isnull().sum()
print(missing)

# 3. Duplicate analysis
print("\n3. DUPLICATE ANALYSIS")
print("-" * 40)

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate review IDs:", df["review_id"].duplicated().sum())

# 4. Rating analysis
print("\n4. RATING ANALYSIS")
print("-" * 40)

print(df["rating"].value_counts().sort_index())
print("\nAverage rating:", round(df["rating"].mean(), 2))

# 5. Category analysis
print("\n5. PRODUCT CATEGORY ANALYSIS")
print("-" * 40)

category_summary = (
    df.groupby("category")
    .agg(
        total_reviews=("review_id", "count"),
        average_rating=("rating", "mean")
    )
    .round(2)
    .sort_values("total_reviews", ascending=False)
)

print(category_summary)

# 6. Sentiment analysis
print("\n6. SENTIMENT DISTRIBUTION")
print("-" * 40)

sentiment_summary = (
    df["expected_sentiment"]
    .value_counts()
    .to_frame("total_reviews")
)

sentiment_summary["percentage"] = (
    sentiment_summary["total_reviews"] / len(df) * 100
).round(2)

print(sentiment_summary)

# 7. Aspect analysis
print("\n7. CUSTOMER FEEDBACK ASPECTS")
print("-" * 40)

aspect_summary = (
    df.groupby("primary_aspect")
    .agg(
        total_reviews=("review_id", "count"),
        average_rating=("rating", "mean")
    )
    .round(2)
    .sort_values("total_reviews", ascending=False)
)

print(aspect_summary)

# 8. Negative feedback analysis
print("\n8. NEGATIVE FEEDBACK BY ASPECT")
print("-" * 40)

negative_reviews = df[
    df["expected_sentiment"] == "Negative"
]

negative_aspects = (
    negative_reviews["primary_aspect"]
    .value_counts()
)

print(negative_aspects)

# 9. Monthly review trends
print("\n9. MONTHLY REVIEW TRENDS")
print("-" * 40)

df["review_date"] = pd.to_datetime(df["review_date"])

monthly_trends = (
    df.groupby(df["review_date"].dt.to_period("M"))
    .size()
)

print(monthly_trends)

# 10. Save analytical reports
category_summary.to_csv(
    REPORT_DIR / "category_summary.csv"
)

sentiment_summary.to_csv(
    REPORT_DIR / "sentiment_summary.csv"
)

aspect_summary.to_csv(
    REPORT_DIR / "aspect_summary.csv"
)

negative_aspects.to_csv(
    REPORT_DIR / "negative_feedback.csv"
)

print("\n" + "=" * 55)
print("EDA COMPLETED SUCCESSFULLY!")
print("Reports saved inside the reports folder.")
print("=" * 55)
