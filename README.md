# Feed Analyzer

A Chrome extension that reads the posts already shown in your X (x.com) feed, sends them to a small local server, and sorts each post into one of 4 categories:

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
extension/  Chrome extension (Manifest V3, plain JavaScript)
server/     FastAPI server that runs the model
model/      data prep, training, evaluation (Phase 2+)
```

## 1. Run the server

Python 3.11–3.14 works. 3.12 matches Google Colab, which we use for training.

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only machines: install the small CPU build of torch first
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

uvicorn app.main:app --port 8000
```

The first start downloads `facebook/bart-large-mnli` (~1.6 GB) into the Hugging Face cache, so it takes a while. Wait for `Application startup complete`.

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

Example: `MODEL_PATH=../model/output/modernbert uvicorn app.main:app --port 8000`

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
3. Open `about:addons` → Feed Analyzer → **Permissions** and make sure access to x.com and localhost is **on**. Firefox may not grant site access on install.
4. To update after code changes, click **Reload** on the add-on in `about:debugging` and reload the x.com tab.

Temporary add-ons are removed when Firefox closes, so load it again after a restart.

**Then, in either browser**
1. Open https://x.com, then open DevTools (F12) → **Console**.
2. Scroll your feed. You should see lines like:
   ```
   [FeedAnalyzer] news (0.87): "Senate passes the funding bill 52–48"
   ```

**Troubleshooting**
- `Can't reach the server at http://localhost:8000`: the server isn't running, or it's still loading the model.
- No `[FeedAnalyzer]` lines at all: in Firefox, check the site permissions (step 3 above) first. Otherwise X may have changed its page, so check `SELECTORS` at the top of `extension/content.js`.
- Background script errors: in Chrome, click **service worker** on the extension card; in Firefox, click **Inspect** on the add-on in `about:debugging`.

## API

```
POST /classify
  request:  {"posts": ["text 1", "text 2"]}
  response: {"results": [{"category": "clickbait", "confidence": 0.91}, ...]}
GET /health -> {"status": "ok"}
```

There is one result per post, in the same order. Posts with confidence below 0.5 will show as "uncertain" in the popup (Phase 3).
