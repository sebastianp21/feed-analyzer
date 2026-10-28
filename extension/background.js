// Feed Analyzer service worker.
// Receives batches of post texts from the content script, sends them to the
// local server, and replies with the results (or an error message).

// Phase 2 moves this into a setting in the popup.
const SERVER_URL = "http://localhost:8000";

// Zero-shot on CPU is slow; give a batch up to a minute.
const REQUEST_TIMEOUT_MS = 60000;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "classify") return false;

  classify(message.posts)
    .then((results) => sendResponse({ ok: true, results }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));

  return true; // tells Chrome we'll call sendResponse later (async)
});

// POST the posts to /classify and return the results array.
async function classify(posts) {
  let response;
  try {
    response = await fetch(`${SERVER_URL}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ posts }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    throw new Error(`Can't reach the server at ${SERVER_URL}. Is it running? (${error.message})`);
  }

  if (!response.ok) {
    throw new Error(`Server error ${response.status}: ${await response.text()}`);
  }
  const data = await response.json();
  return data.results;
}
