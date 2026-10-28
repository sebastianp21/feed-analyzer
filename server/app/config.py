"""Server settings, read from environment variables.

Change the model without touching code, e.g.:
    MODEL_PATH=../model/output/modernbert uvicorn app.main:app
"""

import os

# A Hugging Face model id or a local folder.
# - Zero-shot NLI model (like the default) -> zero-shot over our 4 categories.
# - Fine-tuned model whose labels are our 4 categories -> plain classification.
MODEL_PATH = os.environ.get("MODEL_PATH", "facebook/bart-large-mnli")

# Most posts accepted in one /classify request.
MAX_POSTS = int(os.environ.get("MAX_POSTS", "64"))

# Longer posts are cut to this many characters before classification.
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1000"))
