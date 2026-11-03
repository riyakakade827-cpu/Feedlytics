
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "analyzed_reviews.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(INPUT_PATH)

# Extract numerical features from review text
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=500,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["review_text"])

# Discover 5 groups of similar reviews
model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["topic_cluster"] = model.fit_predict(X)

# Find important terms in each cluster
terms = vectorizer.get_feature_names_out()

print("\nFEEDLYTICS - CUSTOMER TOPIC DISCOVERY")
print("=" * 50)

for cluster_id in range(5):

    cluster_indices = (
        df["topic_cluster"] == cluster_id
    )

    cluster_data = X[cluster_indices.values]

    mean_scores = cluster_data.mean(axis=0).A1

    top_indices = mean_scores.argsort()[-8:][::-1]

    top_words = [
        terms[index] for index in top_indices
    ]

    print(f"\nCluster {cluster_id}")
    print("Top keywords:", ", ".join(top_words))
    print("Number of reviews:", cluster_indices.sum())

# Save results
output_path = OUTPUT_DIR / "topic_clusters.csv"

df.to_csv(output_path, index=False)

print("\nTopic discovery completed!")
print("Saved:", output_path)
