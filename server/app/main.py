"""FastAPI server for Feed Analyzer.

Run from the server/ folder:
    uvicorn app.main:app --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .preprocess import normalize_text

# The model is loaded once and shared by all requests.
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from .classifier import Classifier

        _classifier = Classifier(config.MODEL_PATH, batch_size=config.BATCH_SIZE)
    return _classifier


@asynccontextmanager
async def lifespan(app):
    get_classifier()  # load at startup so the first request isn't slow
    yield


app = FastAPI(title="Feed Analyzer", lifespan=lifespan)

# Open for now. The extension's service worker doesn't need CORS (its
# host_permissions allow the request). Phase 4 locks this to the extension ID.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ClassifyRequest(BaseModel):
    posts: list[str] = Field(max_length=config.MAX_POSTS)


class PostResult(BaseModel):
    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    results: list[PostResult]


@app.get("/health")
def health():
    return {"status": "ok"}


# A plain `def` (not async) so FastAPI runs the slow model call in a worker thread.
@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest, classifier=Depends(get_classifier)):
    # Same cleanup as the training data (mentions -> @user, links -> http).
    texts = [normalize_text(post)[: config.MAX_CHARS] for post in request.posts]

    # Only send non-empty posts to the model; empty ones are "other" with 0 confidence.
    non_empty = [i for i, text in enumerate(texts) if text]
    model_results = classifier.classify([texts[i] for i in non_empty])

    results = [{"category": "other", "confidence": 0.0} for _ in texts]
    for i, result in zip(non_empty, model_results):
        results[i] = result
    return {"results": results}
