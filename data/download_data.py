"""Download and prepare the training dataset.

Source: McAuley-Lab/Amazon-Reviews-2023 (Hugging Face), several product
categories. Builds a balanced 3-class dataset with labels derived from star
ratings, following the project contract:
    1-2 stars -> negative, 3 stars -> neutral, 4-5 stars -> positive

Files are streamed, not fully downloaded: only the first MAX_LINES_PER_CATEGORY
reviews of each category are read.

Output: data/raw/amazon_reviews.csv (versioned with DVC, NOT with Git)

Usage:
    python data/download_data.py
"""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Baby_Products",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Home_and_Kitchen",
    "Office_Products",
    "Sports_and_Outdoors",
    "Toys_and_Games",
]
SAMPLES_PER_CLASS_PER_CATEGORY = 1_000  # 10 categories x 3 classes x 1000 = 30k
MAX_LINES_PER_CATEGORY = 100_000
RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent / "raw" / "amazon_reviews.csv"
NA_LIKE_TEXTS = {"n/a", "na", "nan", "null", "none", "-", "."}


def rating_to_label(rating: int) -> str:
    """Map a 1-5 star rating to a sentiment label (project contract)."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def stream_category(fs: HfFileSystem, category: str) -> pd.DataFrame:
    """Read the first MAX_LINES_PER_CATEGORY reviews of one category."""
    path = f"datasets/{REPO_ID}/raw/review_categories/{category}.jsonl"
    rows = []
    with fs.open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if line_number >= MAX_LINES_PER_CATEGORY:
                break
            record = json.loads(line)
            rows.append({"rating": record.get("rating"), "text": record.get("text")})
    df = pd.DataFrame(rows)
    df["category"] = category
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty and duplicate reviews, add the sentiment label."""
    df = df.dropna(subset=["rating", "text"]).copy()
    df["text"] = df["text"].astype(str).str.replace("<br />", " ", regex=False).str.strip()
    df = df[df["text"] != ""]
    df = df[~df["text"].str.lower().isin(NA_LIKE_TEXTS)]
    df = df.drop_duplicates(subset="text")
    df["rating"] = df["rating"].astype(int)
    df["label"] = df["rating"].apply(rating_to_label)
    return df


def sample_balanced(df: pd.DataFrame) -> pd.DataFrame:
    """Take the same number of reviews from each class of one category."""
    samples = []
    for label, group in df.groupby("label"):
        n = min(len(group), SAMPLES_PER_CLASS_PER_CATEGORY)
        if n < SAMPLES_PER_CLASS_PER_CATEGORY:
            print(f"  Warning: only {n} '{label}' reviews available")
        samples.append(group.sample(n=n, random_state=RANDOM_SEED))
    return pd.concat(samples)


def main() -> None:
    fs = HfFileSystem()
    parts = []
    for index, category in enumerate(CATEGORIES, start=1):
        print(f"[{index}/{len(CATEGORIES)}] Reading '{category}'...")
        reviews = clean_reviews(stream_category(fs, category))
        parts.append(sample_balanced(reviews))

    dataset = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED)
    dataset["source"] = f"huggingface:{REPO_ID}"
    dataset = dataset[["text", "rating", "label", "category", "source"]]
    dataset = dataset.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(dataset)} reviews to {OUTPUT_PATH}")
    print("\nReviews per category and label:")
    print(pd.crosstab(dataset["category"], dataset["label"]).to_string())


if __name__ == "__main__":
    main()
