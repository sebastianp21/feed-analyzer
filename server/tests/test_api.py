"""API contract tests. A fake classifier is used, so no model is downloaded."""

from fastapi.testclient import TestClient

from app import config
from app.main import app, get_classifier


class FakeClassifier:
    """Says every post is news and remembers what it was sent."""

    def __init__(self):
        self.received = []

    def classify(self, texts):
        self.received = texts
        return [{"category": "news", "confidence": 0.9} for _ in texts]


fake = FakeClassifier()
app.dependency_overrides[get_classifier] = lambda: fake

# Not used as a context manager, so the startup hook (real model load) never runs.
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_one_result_per_post():
    response = client.post("/classify", json={"posts": ["text 1", "text 2", "text 3"]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert results[0] == {"category": "news", "confidence": 0.9}


def test_empty_list():
    response = client.post("/classify", json={"posts": []})
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_blank_post_is_other_and_skips_model():
    response = client.post("/classify", json={"posts": ["real post", "   "]})
    results = response.json()["results"]
    assert results[1] == {"category": "other", "confidence": 0.0}
    assert fake.received == ["real post"]


def test_long_post_is_truncated():
    client.post("/classify", json={"posts": ["a" * (config.MAX_CHARS + 500)]})
    assert len(fake.received[0]) == config.MAX_CHARS


def test_too_many_posts_rejected():
    posts = ["x"] * (config.MAX_POSTS + 1)
    assert client.post("/classify", json={"posts": posts}).status_code == 422


def test_missing_posts_field_rejected():
    assert client.post("/classify", json={}).status_code == 422
