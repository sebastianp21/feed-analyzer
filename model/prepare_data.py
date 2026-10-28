"""Build the training data: 4 balanced categories from public datasets.

Usage (from the model/ folder):
    python prepare_data.py
    python prepare_data.py --webis path/to/clickbait17-train-170630.zip

Sources -> categories:
    clickbait  Webis Clickbait 2017, posts most judges called clickbait
    news       Webis posts most judges called NOT clickbait (half)
               + TweetTopic "news_&_social_concern" (half)
    outrage    TweetEval "offensive", label offensive
    other      TweetTopic tweets with no news topic

Output: data/processed/{train,val,test}.csv with columns text, category, source.
The test split is never used in training.
"""

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent

# Reuse the server's category list and text cleanup, so training and the live
# app see text in exactly the same format.
sys.path.insert(0, str(MODEL_DIR.parent / "server"))
from app.categories import CATEGORIES  # noqa: E402
from app.preprocess import normalize_text  # noqa: E402

SEED = 42
PER_CATEGORY = 2000  # posts per category after balancing
WEBIS_NEWS_SHARE = 0.5  # share of "news" posts taken from Webis (rest from TweetTopic)
VAL_SHARE = 0.15
TEST_SHARE = 0.15
MIN_CHARS = 10  # drop posts with fewer real characters than this (not counting @user/http)

DEFAULT_WEBIS = MODEL_DIR / "data/raw/webis/clickbait17-train-170630.zip"
PROCESSED_DIR = MODEL_DIR / "data/processed"
NEWS_TOPIC = "news_&_social_concern"


# ---------- Mapping rules (small pure functions, covered by tests) ----------


def is_clickbait_by_majority(judgments):
    """Webis has 5 judges scoring 0 to 1. Clickbait if more than half score >= 0.5."""
    votes = sum(1 for score in judgments if score >= 0.5)
    return votes > len(judgments) / 2


def webis_category(judgments):
    return "clickbait" if is_clickbait_by_majority(judgments) else "news"


def topic_category(topic_names):
    """TweetTopic tweets can have several topics; news beats everything else."""
    return "news" if NEWS_TOPIC in topic_names else "other"


# ---------- Loaders: each returns a list of {"text", "category", "source"} ----------


def load_webis(path):
    """Read Webis Clickbait 2017 from the downloaded zip or an unzipped folder."""
    path = Path(path)

    def read_jsonl(name):
        if path.is_dir():
            files = list(path.rglob(name))
            if not files:
                raise FileNotFoundError(f"No {name} under {path}")
            lines = files[0].read_text(encoding="utf-8").splitlines()
        else:
            with zipfile.ZipFile(path) as archive:
                member = next(n for n in archive.namelist() if n.endswith(name))
                lines = archive.read(member).decode("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    judgments = {row["id"]: row["truthJudgments"] for row in read_jsonl("truth.jsonl")}
    records = []
    for post in read_jsonl("instances.jsonl"):
        text = " ".join(post["postText"])
        records.append({
            "text": text,
            "category": webis_category(judgments[post["id"]]),
            "source": "webis",
        })
    return records


def load_tweet_topic():
    from datasets import load_dataset

    # The two coling2022 splits together hold every TweetTopic tweet once.
    data = load_dataset("cardiffnlp/tweet_topic_multi")
    records, seen_ids = [], set()
    for split in ["train_coling2022", "test_coling2022"]:
        for row in data[split]:
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            records.append({
                "text": row["text"],
                "category": topic_category(row["label_name"]),
                "source": "tweet_topic",
            })
    return records


def load_tweet_eval_offensive():
    from datasets import load_dataset

    data = load_dataset("cardiffnlp/tweet_eval", "offensive")
    offensive = data["train"].features["label"].str2int("offensive")
    return [
        {"text": row["text"], "category": "outrage", "source": "tweet_eval"}
        for split in data
        for row in data[split]
        if row["label"] == offensive
    ]


# ---------- Cleaning, balancing, splitting ----------


def clean(records):
    """Normalize text, drop very short posts and duplicates.

    Texts that appear with two different categories are dropped entirely,
    because we can't tell which label is right.
    """
    df = pd.DataFrame(records)
    df["text"] = df["text"].map(normalize_text)
    words_only = df["text"].str.replace(r"@user|http", "", regex=True).str.strip()
    df = df[words_only.str.len() >= MIN_CHARS]

    df = df.assign(key=df["text"].str.lower())
    df = df[df.groupby("key")["category"].transform("nunique") == 1]
    return df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)


def balance(df, per_category=PER_CATEGORY, seed=SEED):
    """Take per_category posts per category. News is split between Webis and TweetTopic."""

    def take(rows, n, what):
        if len(rows) < n:
            raise ValueError(f"Only {len(rows)} posts for {what}, need {n}")
        return rows.sample(n=n, random_state=seed)

    webis_news = round(per_category * WEBIS_NEWS_SHARE)
    parts = [
        take(df[df.category == "clickbait"], per_category, "clickbait"),
        take(df[(df.category == "news") & (df.source == "webis")], webis_news, "news (webis)"),
        take(df[(df.category == "news") & (df.source == "tweet_topic")],
             per_category - webis_news, "news (tweet_topic)"),
        take(df[df.category == "outrage"], per_category, "outrage"),
        take(df[df.category == "other"], per_category, "other"),
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def split(df, seed=SEED):
    """Stratified train/val/test split."""
    train_val, test = train_test_split(
        df, test_size=TEST_SHARE, stratify=df["category"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=VAL_SHARE / (1 - TEST_SHARE),
        stratify=train_val["category"],
        random_state=seed,
    )
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webis", default=DEFAULT_WEBIS,
                        help="Webis zip or unzipped folder (default: %(default)s)")
    args = parser.parse_args()
    random.seed(SEED)

    print("Loading Webis...")
    records = load_webis(args.webis)
    print("Loading TweetTopic...")
    records += load_tweet_topic()
    print("Loading TweetEval offensive...")
    records += load_tweet_eval_offensive()

    df = clean(records)
    print("\nAvailable after cleaning:")
    print(df.groupby(["category", "source"]).size().to_string())

    train, val, test = split(balance(df))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"\n{name}: {len(part)} posts")
        print(part["category"].value_counts().sort_index().to_string())
    print(f"\nWrote {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
