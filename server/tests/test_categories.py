"""Tests for the tie rule: outrage > clickbait > news > other."""

from app.categories import CATEGORIES, HYPOTHESES, PRIORITY, pick_category


def test_single_fit_wins():
    scores = {"clickbait": 0.1, "news": 0.8, "outrage": 0.2, "other": 0.3}
    assert pick_category(scores) == ("news", 0.8)


def test_outrage_beats_clickbait():
    scores = {"clickbait": 0.95, "news": 0.1, "outrage": 0.6, "other": 0.1}
    assert pick_category(scores) == ("outrage", 0.6)


def test_clickbait_beats_news():
    scores = {"clickbait": 0.55, "news": 0.9, "outrage": 0.1, "other": 0.1}
    assert pick_category(scores)[0] == "clickbait"


def test_news_beats_other():
    scores = {"clickbait": 0.1, "news": 0.5, "outrage": 0.1, "other": 0.99}
    assert pick_category(scores)[0] == "news"


def test_nothing_fits_uses_highest_score():
    scores = {"clickbait": 0.2, "news": 0.1, "outrage": 0.3, "other": 0.4}
    assert pick_category(scores) == ("other", 0.4)


def test_every_category_has_priority_and_hypothesis():
    assert sorted(PRIORITY) == sorted(CATEGORIES)
    assert sorted(HYPOTHESES) == sorted(CATEGORIES)
