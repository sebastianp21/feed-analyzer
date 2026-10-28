// Feed Analyzer background script (a service worker in Chrome, an event page in Firefox).
// - Receives batches of post texts from the content script, sends them to the
//   local server, and replies with the results (or an error message).
// - Keeps the stats the popup shows: the last 50 results, how many posts were
//   skipped as non-English, and the last error.

// Firefox has the promise-based `browser` API; Chrome's `chrome` API also returns promises in MV3.
const api = globalThis.browser ?? chrome;

const DEFAULT_SERVER_URL = "http://localhost:8000";
const RECENT_LIMIT = 50; // the popup shows stats for this many posts
const REQUEST_TIMEOUT_MS = 60000; // CPU models can be slow; give a batch up to a minute

// Stats live in session storage: they survive the background script going to
// sleep, but reset when the browser closes.
const EMPTY_STATS = { recent: [], skipped: 0, lastError: null };

async function getServerUrl() {
  const { serverUrl } = await api.storage.local.get("serverUrl");
  return serverUrl || DEFAULT_SERVER_URL;
}

// Updates to the stats run one after another, so two batches finishing at the
// same time can't overwrite each other.
let statsQueue = Promise.resolve();
function updateStats(change) {
  statsQueue = statsQueue.then(async () => {
    const { stats } = await api.storage.session.get("stats");
    await api.storage.session.set({ stats: change(stats ?? EMPTY_STATS) });
  });
  return statsQueue;
}

api.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "skipped") {
    updateStats((stats) => ({ ...stats, skipped: stats.skipped + message.count }));
    return false;
  }
  if (message.type !== "classify") return false;

  classify(message.posts)
    .then(async (results) => {
      await updateStats((stats) => ({
        ...stats,
        recent: [...stats.recent, ...results].slice(-RECENT_LIMIT),
        lastError: null,
      }));
      sendResponse({ ok: true, results });
    })
    .catch(async (error) => {
      await updateStats((stats) => ({ ...stats, lastError: error.message }));
      sendResponse({ ok: false, error: error.message });
    });

  return true; // tells the browser we'll call sendResponse later (async)
});

// POST the posts to /classify and return the results array.
async function classify(posts) {
  const serverUrl = await getServerUrl();
  let response;
  try {
    response = await fetch(`${serverUrl}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ posts }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    throw new Error(`Can't reach the server at ${serverUrl}. Is it running? (${error.message})`);
  }

  if (!response.ok) {
    throw new Error(`Server error ${response.status}: ${await response.text()}`);
  }
  const data = await response.json();
  return data.results;
}
