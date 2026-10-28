# model/

| File | What it does |
|------|--------------|
| `prepare_data.py` | Builds `data/processed/{train,val,test}.csv` from the public datasets |
| `train.py` | Fine-tunes one model (`--model modernbert|roberta|distilbert`) |
| `train_colab.ipynb` | Runs `train.py` for all three models on a Colab GPU and saves them to Google Drive |
| `tests/` | Tests for the data mapping (`pytest`) |

## Data

| Category | Source | Rule |
|----------|--------|------|
| clickbait | [Webis Clickbait 2017](https://zenodo.org/record/5530410) training set | most of the 5 judges scored it ≥ 0.5 |
| news | Webis (half) + [TweetTopic](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi) (half) | Webis: most judges said *not* clickbait. TweetTopic: has the `news_&_social_concern` topic |
| outrage | [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `offensive` | label = offensive |
| other | TweetTopic (half) + [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `emoji` (half) | TweetTopic: no news topic (daily life, sports, music, gaming, ...). TweetEval emoji: everyday captions, with Instagram location tags removed |

- **Size:** 2,000 posts per category. The split is 70/15/15 for train/val/test (5,600 / 1,200 / 1,200), stratified, with seed 42. The **test split is never used in training**.
- **News from two sources:** news posts come from Webis too, so the model learns clickbait *wording* rather than "news outlet = clickbait".
- **Text cleanup:** all text goes through `server/app/preprocess.py`. Mentions become `@user` and links become `http`, and the `@user @user` mentions at the start of replies are removed. The server applies the same cleanup to live posts.
- **Length balance:** with TweetTopic alone, only 4 of the ~500 training posts with 6 words or fewer were "other", so the model learned "short post = clickbait/outrage" and mislabelled short X captions (61% clickbait on a live feed). The TweetEval emoji tweets bring short "other" posts up to 14%, the same as clickbait and outrage. `prepare_data.py` prints this length check every time it runs.
- **Duplicates:** exact duplicates are removed before splitting. Texts that appear with two different labels are dropped.

### Known limitations
- The data is from 2016–2021, so it may not fully match a 2026 feed.
- TweetEval's "offensive" also covers plain swearing aimed at no one, so some harmless posts with profanity will be labelled outrage.
- Only English posts are covered.
