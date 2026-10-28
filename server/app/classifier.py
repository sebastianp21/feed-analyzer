"""Loads one model and classifies lists of posts.

The model type is detected from its config, so swapping models only needs MODEL_PATH:
- labels are exactly our 4 categories -> fine-tuned text classifier
- anything else (e.g. an MNLI model)  -> zero-shot over HYPOTHESES
"""

from .categories import CATEGORIES, HYPOTHESES, pick_category

# Maps a zero-shot hypothesis sentence back to its category name.
HYPOTHESIS_TO_CATEGORY = {sentence: cat for cat, sentence in HYPOTHESES.items()}


class Classifier:
    def __init__(self, model_path, batch_size=8):
        # Imported here so the API tests can run without loading torch/transformers.
        import torch
        from transformers import AutoConfig, pipeline

        device = 0 if torch.cuda.is_available() else -1
        labels = set(AutoConfig.from_pretrained(model_path).id2label.values())
        self.zero_shot = labels != set(CATEGORIES)
        self.batch_size = batch_size

        task = "zero-shot-classification" if self.zero_shot else "text-classification"
        self.pipe = pipeline(task, model=model_path, device=device)
        print(f"Loaded {model_path} as {task} on {'GPU' if device == 0 else 'CPU'}")

    def classify(self, texts):
        """Return one {"category", "confidence"} dict per input text, in order."""
        if not texts:
            return []
        if self.zero_shot:
            return self._classify_zero_shot(texts)
        return self._classify_fine_tuned(texts)

    def _classify_zero_shot(self, texts):
        outputs = self.pipe(
            texts,
            candidate_labels=list(HYPOTHESES.values()),
            hypothesis_template="{}",  # our hypotheses are already full sentences
            multi_label=True,  # score each category on its own, so the tie rule can apply
            batch_size=self.batch_size,
        )
        results = []
        for out in outputs:
            scores = {
                HYPOTHESIS_TO_CATEGORY[label]: score
                for label, score in zip(out["labels"], out["scores"])
            }
            category, confidence = pick_category(scores)
            results.append({"category": category, "confidence": round(confidence, 4)})
        return results

    def _classify_fine_tuned(self, texts):
        # Returns the top label per text: [{"label": "news", "score": 0.93}, ...]
        outputs = self.pipe(texts, truncation=True, batch_size=self.batch_size)
        return [
            {"category": out["label"], "confidence": round(out["score"], 4)}
            for out in outputs
        ]
