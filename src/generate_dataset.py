
import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# Make results reproducible
random.seed(42)

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Product catalogue
products = {
    "Smartphones": ["Nova X5", "PixelPro S", "Galaxy Lite"],
    "Laptops": ["BookAir 14", "TechBook Pro", "NoteMax 15"],
    "Headphones": ["SoundBeat Pro", "BassPods X", "AudioFlex"],
    "Smartwatches": ["FitTrack S", "WristPro 2", "ActiveWatch"],
    "Accessories": ["FastCharge 65W", "PowerBank Plus", "USB Hub Pro"]
}

# Review text templates
templates = {
    "Positive": {
        "Product Quality": [
            "The build quality is excellent.",
            "I am impressed with the premium quality.",
            "The product feels durable and well made."
        ],
        "Delivery": [
            "Delivery was faster than expected.",
            "The package arrived earlier than promised.",
            "The delivery experience was excellent."
        ],
        "Price": [
            "Excellent value for money.",
            "The price is reasonable for these features.",
            "Definitely worth the amount paid."
        ],
        "Customer Support": [
            "Customer support resolved my issue quickly.",
            "The support team was extremely helpful.",
            "I received excellent assistance."
        ],
        "Performance": [
            "Performance is smooth and reliable.",
            "Everything works perfectly.",
            "The product performs better than expected."
        ],
        "Battery": [
            "Battery life is excellent.",
            "The battery lasts throughout the day.",
            "I am very satisfied with the battery."
        ]
    },
    "Negative": {
        "Product Quality": [
            "The build quality is disappointing.",
            "The product feels cheap and fragile.",
            "The quality is much worse than expected."
        ],
        "Delivery": [
            "The delivery was extremely delayed.",
            "My package arrived much later than promised.",
            "The delivery experience was frustrating."
        ],
        "Price": [
            "The product is overpriced.",
            "It is not worth the money.",
            "The price is too high for these features."
        ],
        "Customer Support": [
            "Customer support was unhelpful.",
            "Nobody resolved my complaint.",
            "The support team was disappointing."
        ],
        "Performance": [
            "The product is slow and unreliable.",
            "Performance is very disappointing.",
            "It frequently stops working."
        ],
        "Battery": [
            "The battery drains very quickly.",
            "Battery life is extremely poor.",
            "I need to charge it constantly."
        ]
    },
    "Neutral": {
        "Product Quality": [
            "The quality is average.",
            "The product meets basic expectations.",
            "Nothing particularly special about the build."
        ],
        "Delivery": [
            "The package arrived on the scheduled date.",
            "Delivery was neither early nor late.",
            "The delivery experience was normal."
        ],
        "Price": [
            "The price is fairly average.",
            "It is reasonably priced.",
            "The cost is acceptable."
        ],
        "Customer Support": [
            "Customer support provided standard assistance.",
            "The support experience was average.",
            "My question received a basic response."
        ],
        "Performance": [
            "Performance is acceptable for daily use.",
            "It works as expected.",
            "The performance is neither impressive nor poor."
        ],
        "Battery": [
            "Battery life is average.",
            "The battery lasts a reasonable amount of time.",
            "Battery performance is as expected."
        ]
    },
    "Mixed": {
        "Performance": [
            "Performance is excellent, but the battery drains quickly.",
            "The product works smoothly, but it sometimes freezes."
        ],
        "Delivery": [
            "Product quality is great, but delivery was delayed.",
            "I love the product, but the package arrived late."
        ],
        "Price": [
            "The features are impressive, but it feels overpriced.",
            "The quality is good, but the price is too high."
        ],
        "Customer Support": [
            "The product works well, but customer support was disappointing.",
            "I like the product, but getting help was difficult."
        ],
        "Product Quality": [
            "The design is beautiful, but the build quality could improve.",
            "It looks premium, but some parts feel cheap."
        ],
        "Battery": [
            "Charging is fast, but battery life is disappointing.",
            "The battery charges quickly, but drains too soon."
        ]
    }
}

ratings = {
    "Positive": [4, 5],
    "Negative": [1, 2],
    "Neutral": [3],
    "Mixed": [3, 4]
}

sentiments = ["Positive", "Negative", "Neutral", "Mixed"]
weights = [45, 25, 15, 15]

records = []

for i in range(1, 501):

    category = random.choice(list(products.keys()))
    product = random.choice(products[category])

    sentiment = random.choices(
        sentiments,
        weights=weights,
        k=1
    )[0]

    aspect = random.choice(
        list(templates[sentiment].keys())
    )

    review = random.choice(
        templates[sentiment][aspect]
    )

    rating = random.choice(ratings[sentiment])

    review_date = (
        datetime.today()
        - timedelta(days=random.randint(0, 180))
    )

    records.append({
        "review_id": f"REV{i:04d}",
        "product": product,
        "category": category,
        "review_text": review,
        "rating": rating,
        "review_date": review_date.strftime("%Y-%m-%d"),
        "verified_purchase": random.choice([True, False]),
        "primary_aspect": aspect,
        "expected_sentiment": sentiment,
        "is_synthetic": True
    })

# Create DataFrame
df = pd.DataFrame(records)

# Save dataset
output_path = RAW_DIR / "customer_reviews.csv"
df.to_csv(output_path, index=False)

# Display summary
print("=" * 45)
print("       FEEDLYTICS DATA GENERATOR")
print("=" * 45)

print(f"\nTotal reviews generated: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nDataset saved at: {output_path}")

print("\nSentiment distribution:")
print(df["expected_sentiment"].value_counts())

print("\nFirst five reviews:")
print(df[["product", "rating", "review_text"]].head())

print("\nDataset generation completed!")
