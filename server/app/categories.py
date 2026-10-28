"""The 4 categories and the rule for picking one when a post fits several."""

CATEGORIES = ["clickbait", "news", "outrage", "other"]

# When a post fits two categories, the one listed first wins.
PRIORITY = ["outrage", "clickbait", "news", "other"]

# A category "fits" when its score is at least this high.
FIT_THRESHOLD = 0.5

# Sentences the zero-shot model checks each post against.
# NLI models do much better with a full sentence than with a bare word like "news".
# Tune these freely; they only affect the zero-shot baseline.
HYPOTHESES = {
    "clickbait": "This post is clickbait that teases something or begs for clicks, likes or replies.",
    "news": "This post reports on a real news event.",
    "outrage": "This post insults or attacks a person or a group.",
    "other": "This post is a personal update, joke, meme, sports talk, music, or an ad.",
}


def pick_category(scores):
    """Pick one category from {category: score} and return (category, confidence).

    Scores are independent (each between 0 and 1), not summing to 1.
    - If one or more categories reach FIT_THRESHOLD, the highest-priority one wins.
    - Otherwise the highest score wins (its low confidence marks it "uncertain").
    """
    fitting = [c for c in PRIORITY if scores.get(c, 0.0) >= FIT_THRESHOLD]
    if fitting:
        best = fitting[0]
    else:
        best = max(CATEGORIES, key=lambda c: scores.get(c, 0.0))
    return best, scores.get(best, 0.0)
