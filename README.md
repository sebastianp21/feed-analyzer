# Feed Analyzer

A browser extension (Chrome and Firefox) that reads the posts already shown in your X (x.com) feed, sends them to a small local server, and sorts each post into one of 4 categories:

| Category  | Meaning                                                   |
|-----------|-----------------------------------------------------------|
| clickbait | teases or begs for clicks/replies                         |
| news      | reports on real events                                    |
| outrage   | insults and hostility toward people or groups             |
| other     | everything else (personal posts, memes, sports, ads)      |

If a post fits two categories: **outrage > clickbait > news > other**.

There are no LLMs or external AI APIs. Classification uses local Hugging Face encoder models only.

## Repo layout

```
extension/  browser extension (Manifest V3, plain JavaScript)
server/     FastAPI server that runs the model
model/      data prep, training (+ Colab notebook), evaluation
```

## 1. Run the server

There are two ways to run it:
- **Fine-tuned model (recommended):** first train one (see [Training a model](#3-training-a-model)), then point `MODEL_PATH` at it.
- **Zero-shot baseline:** no training needed, but slow and much less accurate.

Python 3.11–3.14 works. 3.12 matches Google Colab, which we use for training.

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only machines: install the small CPU build of torch first
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# Fine-tuned model:
MODEL_PATH=../model/output/modernbert uvicorn app.main:app --port 8000
# or the zero-shot baseline:
uvicorn app.main:app --port 8000
```

With no `MODEL_PATH`, the first start downloads `facebook/bart-large-mnli` (~1.6 GB) into the Hugging Face cache, so it takes a while. Wait for `Application startup complete`.

Try it:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/classify -H "Content-Type: application/json" \
  -d '{"posts": ["You won'\''t believe what happened next 👇", "Senate passes the funding bill 52-48"]}'
```

### Settings (environment variables)

| Variable     | Default                    | What it does                                      |
|--------------|----------------------------|---------------------------------------------------|
| `MODEL_PATH` | `facebook/bart-large-mnli` | HF model id or local folder. Fine-tuned models with our 4 labels are detected automatically. |
| `MAX_POSTS`  | `64`                       | Most posts per request                            |
| `MAX_CHARS`  | `1000`                     | Posts are cut to this length                      |
| `BATCH_SIZE` | `16`                       | Posts the model processes at once                 |

### Tests

```bash
cd server
pytest
```

The tests use a fake model, so they run in seconds and download nothing.

## 2. Load the extension

The same `extension/` folder works in Chrome (121+) and Firefox (121+).

**Chrome**
1. Open `chrome://extensions` and turn on **Developer mode** (top right).
2. Click **Load unpacked** and pick the `extension/` folder.
3. To update after code changes, click the reload icon on the extension card **and** reload the x.com tab.

Chrome may show a warning about the unrecognized `browser_specific_settings` key or `background.scripts`. Both are only for Firefox, so the warning is harmless.

**Firefox**
1. Open `about:debugging#/runtime/this-firefox`.
2. Click **Load Temporary Add-on…** and pick `extension/manifest.json`.
3. Open `about:addons` → Feed Analyzer → **Permissions** and make sure access to x.com and localhost is **on**. Firefox may not grant site access on install, or after the manifest's permissions change.
4. To update after code changes, click **Reload** on the add-on in `about:debugging` and reload the x.com tab.

Temporary add-ons are removed when Firefox closes, so load it again after a restart.

**Then, in either browser**
1. Open https://x.com and scroll your feed.
2. Click the Feed Analyzer icon in the toolbar (in Chrome it's under the puzzle-piece menu; pin it to keep it visible). The popup shows:
   - the share of each category in the last 50 English posts, and a summary like "Mostly news (42%)"
   - how many non-English posts were skipped (only English is classified)
   - whether the server is online, with an error message if it can't be reached
   - **Settings**: the server address (default `http://localhost:8000`) and a button to reset the stats
3. For debugging: DevTools (F12) → **Console** on the x.com tab shows one line per post:
   ```
   [FeedAnalyzer] news (0.87): "Senate passes the funding bill 52–48"
   ```

**Troubleshooting**
- `Can't reach the server at http://localhost:8000`: the server isn't running, or it's still loading the model.
- No `[FeedAnalyzer]` lines at all: in Firefox, check the site permissions (step 3 above) first. Otherwise X may have changed its page, so check `SELECTORS` at the top of `extension/content.js`.
- Background script errors: in Chrome, click **service worker** on the extension card; in Firefox, click **Inspect** on the add-on in `about:debugging`.

## 3. Training a model

All training data comes from public datasets; nothing is hand-labelled. See [model/README.md](model/README.md) for the details.

**Step 1: build the data (on your computer, once).** The result, `model/data/processed/*.csv`, is committed to git, so teammates can skip this step.

1. Download `clickbait17-train-170630.zip` (937 MB) from https://zenodo.org/record/5530410 into `model/data/raw/webis/`.
2. Run:
   ```bash
   cd model
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python prepare_data.py
   pytest
   ```

**Step 2: train (on Google Colab, free T4 GPU).**
1. Open `model/train_colab.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
2. Set Runtime → Change runtime type → **T4 GPU**.
3. Set `REPO_URL` in step 3 of the notebook, then run all cells.
4. The trained models are saved to `MyDrive/feed_analyzer/models/` in Google Drive. Download one and unzip it into `model/output/<name>/`.

To check that training works without a GPU (takes about a minute on CPU):
```bash
python train.py --model distilbert --epochs 1 --limit 200
```

## API

```
POST /classify
  request:  {"posts": ["text 1", "text 2"]}
  response: {"results": [{"category": "clickbait", "confidence": 0.91}, ...]}
GET /health -> {"status": "ok"}
```

There is one result per post, in the same order. Posts with confidence below 0.5 will show as "uncertain" in the popup (Phase 3).
