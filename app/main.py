
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Feedlytics | Customer Intelligence",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# 2. PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "topic_clusters.csv"
)

# -----------------------------------
# 3. LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["review_date"] = pd.to_datetime(
        df["review_date"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    return df


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "Dataset not found. Run the NLP and topic discovery scripts first."
    )
    st.stop()

# -----------------------------------
# 4. DASHBOARD HEADER
# -----------------------------------

st.title("📊 Feedlytics")

st.subheader(
    "AI-Powered Customer Feedback Intelligence"
)

st.caption(
    "Transforming customer reviews into actionable business insights."
)

st.warning(
    "Demo environment: this dashboard currently uses "
    "synthetic customer reviews, not real customer data."
)

# -----------------------------------
# 5. SIDEBAR FILTERS
# -----------------------------------

st.sidebar.title("Dashboard Filters")

categories = sorted(df["category"].dropna().unique())

selected_categories = st.sidebar.multiselect(
    "Select product categories",
    options=categories,
    default=categories
)

sentiments = sorted(df["final_sentiment"].dropna().unique())

selected_sentiments = st.sidebar.multiselect(
    "Select sentiments",
    options=sentiments,
    default=sentiments
)

rating_range = st.sidebar.slider(
    "Select rating range",
    min_value=1,
    max_value=5,
    value=(1, 5)
)

search_text = st.sidebar.text_input(
    "Search customer reviews"
)

# -----------------------------------
# 6. APPLY FILTERS
# -----------------------------------

filtered_df = df[
    df["category"].isin(selected_categories)
    & df["final_sentiment"].isin(selected_sentiments)
    & df["rating"].between(
        rating_range[0],
        rating_range[1]
    )
].copy()

if search_text:
    filtered_df = filtered_df[
        filtered_df["review_text"].str.contains(
            search_text,
            case=False,
            na=False
        )
    ]

# -----------------------------------
# 7. KPI CARDS
# -----------------------------------

st.divider()

st.header("Business Overview")

total_reviews = len(filtered_df)

average_rating = (
    filtered_df["rating"].mean()
    if total_reviews else 0
)

positive_pct = (
    filtered_df["final_sentiment"]
    .eq("Positive").mean() * 100
    if total_reviews else 0
)

negative_pct = (
    filtered_df["final_sentiment"]
    .eq("Negative").mean() * 100
    if total_reviews else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Reviews",
    f"{total_reviews:,}"
)

col2.metric(
    "Average Rating",
    f"{average_rating:.2f} / 5"
)

col3.metric(
    "Positive Reviews",
    f"{positive_pct:.1f}%"
)

col4.metric(
    "Negative Reviews",
    f"{negative_pct:.1f}%"
)

if total_reviews == 0:
    st.info("No reviews match the selected filters.")
    st.stop()

# -----------------------------------
# 8. SENTIMENT ANALYSIS
# -----------------------------------

st.divider()

st.header("Sentiment Intelligence")

col1, col2 = st.columns(2)

sentiment_counts = (
    filtered_df["final_sentiment"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Reviews")
)

with col1:

    fig_sentiment = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Reviews",
        hole=0.55,
        title="Customer Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#16A34A",
            "Negative": "#DC5656",
            "Neutral": "#D69E36",
            "Mixed": "#6685D9"
        }
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )

with col2:

    rating_counts = (
        filtered_df["rating"]
        .value_counts()
        .sort_index()
        .rename_axis("Rating")
        .reset_index(name="Reviews")
    )

    fig_rating = px.bar(
        rating_counts,
        x="Rating",
        y="Reviews",
        title="Customer Rating Distribution",
        text="Reviews"
    )

    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )

# -----------------------------------
# 9. PRODUCT ANALYSIS
# -----------------------------------

st.divider()

st.header("Product Performance")

product_summary = (
    filtered_df.groupby("category")
    .agg(
        total_reviews=("review_id", "count"),
        average_rating=("rating", "mean")
    )
    .reset_index()
)

fig_products = px.bar(
    product_summary,
    x="category",
    y="average_rating",
    color="category",
    text="average_rating",
    title="Average Rating by Product Category"
)

fig_products.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

fig_products.update_layout(
    yaxis_range=[0, 5],
    showlegend=False
)

st.plotly_chart(
    fig_products,
    use_container_width=True
)

# -----------------------------------
# 10. CUSTOMER PAIN POINTS
# -----------------------------------

st.divider()

st.header("Customer Pain Points")

negative_df = filtered_df[
    filtered_df["final_sentiment"] == "Negative"
]

if not negative_df.empty:

    pain_points = (
        negative_df["primary_aspect"]
        .value_counts()
        .rename_axis("Aspect")
        .reset_index(name="Complaints")
    )

    fig_pain = px.bar(
        pain_points,
        x="Complaints",
        y="Aspect",
        orientation="h",
        title="Negative Feedback by Aspect",
        text="Complaints"
    )

    st.plotly_chart(
        fig_pain,
        use_container_width=True
    )

else:
    st.info("No negative reviews in this selection.")

# -----------------------------------
# 11. TOPIC DISCOVERY
# -----------------------------------

st.divider()

st.header("Customer Feedback Themes")

topic_labels = {
    0: "Positive Experiences",
    1: "Product Value and Quality",
    2: "Delivery Experience",
    3: "Customer Support",
    4: "General Product Feedback"
}

topic_data = (
    filtered_df["topic_cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Reviews")
)

topic_data["Theme"] = (
    topic_data["Cluster"]
    .map(topic_labels)
)

fig_topics = px.bar(
    topic_data,
    x="Theme",
    y="Reviews",
    color="Theme",
    title="Discovered Customer Feedback Clusters",
    text="Reviews"
)

fig_topics.update_layout(
    showlegend=False
)

st.plotly_chart(
    fig_topics,
    use_container_width=True
)

st.caption(
    "Cluster labels are preliminary human interpretations "
    "of TF-IDF keywords. They are not verified AI-generated topics."
)

# -----------------------------------
# 12. BUSINESS INSIGHTS
# -----------------------------------

st.divider()

st.header("Business Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:

    if not product_summary.empty:

        weakest = product_summary.sort_values(
            "average_rating"
        ).iloc[0]

        st.info(
            f"**Product to investigate:** "
            f"{weakest['category']} has the lowest average "
            f"rating ({weakest['average_rating']:.2f}/5) "
            f"among the selected categories."
        )

with insight_col2:

    if not negative_df.empty:

        most_common = (
            negative_df["primary_aspect"]
            .value_counts()
        )

        st.warning(
            f"**Priority complaint:** "
            f"{most_common.index[0]} has the most negative "
            f"reviews ({most_common.iloc[0]})."
        )

# -----------------------------------
# 13. REVIEW EXPLORER
# -----------------------------------

st.divider()

st.header("Customer Review Explorer")

display_columns = [
    "review_id",
    "product",
    "category",
    "review_text",
    "rating",
    "final_sentiment",
    "primary_aspect",
    "review_date"
]

st.dataframe(
    filtered_df[display_columns].sort_values(
        "review_date",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# -----------------------------------
# 14. DOWNLOAD FILTERED DATA
# -----------------------------------

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Reviews",
    data=csv_data,
    file_name="feedlytics_filtered_reviews.csv",
    mime="text/csv"
)

# -----------------------------------
# 15. FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Feedlytics | Built with Python, Pandas, "
    "Scikit-learn, Plotly and Streamlit."
)
