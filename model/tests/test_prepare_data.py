"""Tests for the data mapping. They use tiny fake data, so nothing is downloaded."""

import json
import zipfile

import pandas as pd
import pytest

import prepare_data as prep
from app.preprocess import normalize_text


# ---------- Mapping rules ----------


def test_webis_majority_clickbait():
    assert prep.webis_category([1.0, 1.0, 0.66, 0.0, 0.33]) == "clickbait"  # 3 of 5


def test_webis_minority_is_news():
    assert prep.webis_category([1.0, 0.66, 0.33, 0.0, 0.0]) == "news"  # 2 of 5


def test_topic_news_wins_over_other_topics():
    assert prep.topic_category(["film_tv_&_video", "news_&_social_concern"]) == "news"


def test_topic_without_news_is_other():
    assert prep.topic_category(["sports", "music"]) == "other"


# ---------- Webis loader ----------


def make_webis_zip(path):
    instances = [
        {"id": "1", "postText": ["You won't believe this"]},
        {"id": "2", "postText": ["Senate passes the funding bill"]},
    ]
    truth = [
        {"id": "1", "truthJudgments": [1.0, 1.0, 1.0, 0.66, 0.0]},
        {"id": "2", "truthJudgments": [0.0, 0.0, 0.33, 0.0, 0.0]},
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("folder/instances.jsonl", "\n".join(map(json.dumps, instances)))
        archive.writestr("folder/truth.jsonl", "\n".join(map(json.dumps, truth)))


def test_load_webis_from_zip(tmp_path):
    zip_path = tmp_path / "webis.zip"
    make_webis_zip(zip_path)
    records = prep.load_webis(zip_path)
    assert [(r["text"], r["category"]) for r in records] == [
        ("You won't believe this", "clickbait"),
        ("Senate passes the funding bill", "news"),
    ]


# ---------- Text normalization ----------


@pytest.mark.parametrize("raw, expected", [
    ("Hi {{USERNAME}} see {{URL}}", "Hi @user see http"),
    ("Thanks {@WA Police Force@} !", "Thanks @user !"),
    ("@USER @bob you are wrong URL", "you are wrong http"),  # leading reply mentions removed
    ("Read https://t.co/abc123 now @jack", "Read http now @user"),
    ("Tom &amp; Jerry\n\n  again", "Tom & Jerry again"),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


# ---------- Cleaning, balancing, splitting ----------


def test_clean_drops_duplicates_conflicts_and_short_posts():
    records = [
        {"text": "Same text in two places", "category": "news", "source": "webis"},
        {"text": "same TEXT in two places", "category": "news", "source": "tweet_topic"},
        {"text": "Text with two different labels", "category": "news", "source": "webis"},
        {"text": "Text with two different labels", "category": "outrage", "source": "tweet_eval_offensive"},
        {"text": "@user http", "category": "other", "source": "tweet_topic"},
    ]
    df = prep.clean(records)
    assert df["text"].tolist() == ["Same text in two places"]


def fake_pool(n=50):
    """n posts for every category/source combination balance() uses."""
    groups = [(cat, src) for cat, sources in prep.MIX.items() for src in sources]
    rows = [{"text": f"{cat} {src} post number {i}", "category": cat, "source": src}
            for cat, src in groups for i in range(n)]
    return pd.DataFrame(rows)


def test_balance_gives_equal_categories_and_mixed_sources():
    df = prep.balance(fake_pool(), per_category=20)
    assert df["category"].value_counts().to_dict() == {c: 20 for c in prep.CATEGORIES}
    news_sources = df[df.category == "news"]["source"].value_counts().to_dict()
    assert news_sources == {"webis": 10, "tweet_topic": 10}
    other_sources = df[df.category == "other"]["source"].value_counts().to_dict()
    assert other_sources == {"tweet_topic": 10, "tweet_eval_emoji": 10}


def test_every_category_mix_adds_up_to_one():
    assert sorted(prep.MIX) == sorted(prep.CATEGORIES)
    for sources in prep.MIX.values():
        assert sum(sources.values()) == 1.0


@pytest.mark.parametrize("raw, expected", [
    ("Beautiful day for a game \ufe0f @ Mariners Diamond Club", "Beautiful day for a game"),
    ("Friends who workout together (@ Muv Training)", "Friends who workout together"),
    ("Thank you festival, until next year…", "Thank you festival, until next year"),
    ("Never a bad day with @user here", "Never a bad day with @user here"),  # mentions stay
])
def test_clean_emoji_tweet(raw, expected):
    assert prep.clean_emoji_tweet(raw) == expected


def test_balance_fails_loudly_when_short():
    with pytest.raises(ValueError, match="clickbait"):
        prep.balance(fake_pool(n=5), per_category=20)


def test_balance_is_reproducible():
    pool = fake_pool()
    assert prep.balance(pool, 20).equals(prep.balance(pool, 20))


def test_split_has_no_overlap_and_keeps_balance():
    df = prep.balance(fake_pool(), per_category=40)
    train, val, test = prep.split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert not set(train.text) & set(test.text)
    assert not set(val.text) & set(test.text)
    assert test["category"].value_counts().nunique() == 1  # same count per category
