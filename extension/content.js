// Feed Analyzer content script: runs on x.com.
// Finds posts as they appear, skips ones already seen, and sends them in
// batches to the background service worker, which calls our server.
// (We can't call the server from here: x.com's page security blocks it.)

// All x.com DOM selectors live here. If X changes its page, fix them here only.
const SELECTORS = {
  post: 'article[data-testid="tweet"]', // one post in the feed
  text: '[data-testid="tweetText"]', // the post's text (the first match is the main post, not a quote)
  permalink: 'a[href*="/status/"]:has(> time)', // the timestamp link, which points to the post itself
};

// X puts a language code on each post's text (lang="en", lang="es", ...).
// We only classify English, plus X's codes for "no real language" (emoji only,
// links only, undetermined), since those are usually short English-feed posts.
const LANGUAGES_TO_CLASSIFY = new Set(["en", "und", "zxx", "qme", "qam", "qct", "qht", "qst"]);

const BATCH_SIZE = 10; // send as soon as this many posts are waiting...
const BATCH_DELAY_MS = 1500; // ...or this long after the last new post
const SCAN_DELAY_MS = 500; // X changes the page constantly; scan at most this often

const seenKeys = new Set(); // posts already sent (or waiting to be sent)
let queue = []; // [{ key, text }] waiting to be sent
let batchTimer = null;
let scanTimer = null;

// Read a post's text, keeping emoji (X draws them as <img alt="👇">).
function readText(textElement) {
  const copy = textElement.cloneNode(true);
  copy.querySelectorAll("img[alt]").forEach((img) => img.replaceWith(img.alt));
  return copy.textContent.trim();
}

// A stable ID for a post: its /status/ URL, or the text itself if there's no link.
function postKey(article, text) {
  const link = article.querySelector(SELECTORS.permalink);
  return link ? link.getAttribute("href") : "text:" + text;
}

// True if the post is English (or has no language). Posts without a lang tag are kept.
function isEnglish(textElement) {
  const lang = textElement.getAttribute("lang");
  return !lang || LANGUAGES_TO_CLASSIFY.has(lang);
}

// Look for new posts on the page and queue them.
function scanPage() {
  scanTimer = null;
  let skipped = 0;
  for (const article of document.querySelectorAll(SELECTORS.post)) {
    const textElement = article.querySelector(SELECTORS.text);
    if (!textElement) continue; // image/video-only posts have no text to classify

    const text = readText(textElement);
    const key = postKey(article, text);
    if (!text || seenKeys.has(key)) continue;

    seenKeys.add(key);
    if (isEnglish(textElement)) {
      queue.push({ key, text });
    } else {
      skipped++;
    }
  }

  if (skipped > 0) {
    try {
      // The empty callback reads lastError so the browser doesn't warn about it.
      chrome.runtime.sendMessage({ type: "skipped", count: skipped }, () => chrome.runtime.lastError);
    } catch {
      // Extension was reloaded while this tab stayed open; nothing to do.
    }
  }

  if (queue.length >= BATCH_SIZE) {
    sendBatch();
  } else if (queue.length > 0) {
    clearTimeout(batchTimer);
    batchTimer = setTimeout(sendBatch, BATCH_DELAY_MS);
  }
}

// Send everything in the queue to the service worker and log the results.
function sendBatch() {
  clearTimeout(batchTimer);
  batchTimer = null;
  if (queue.length === 0) return;

  const batch = queue;
  queue = [];

  const onFailure = (message) => {
    console.warn(`[FeedAnalyzer] ${message}`);
    // Forget these posts so they're retried on the next scan.
    batch.forEach((post) => seenKeys.delete(post.key));
  };

  try {
    chrome.runtime.sendMessage(
      { type: "classify", posts: batch.map((post) => post.text) },
      (response) => {
        if (chrome.runtime.lastError) return onFailure(chrome.runtime.lastError.message);
        if (!response.ok) return onFailure(response.error);

        response.results.forEach((result, i) => {
          const preview = batch[i].text.slice(0, 80).replace(/\s+/g, " ");
          console.log(
            `[FeedAnalyzer] ${result.category} (${result.confidence.toFixed(2)}): "${preview}"`
          );
        });
      }
    );
  } catch (error) {
    // Happens after the extension is reloaded while this tab stays open.
    onFailure(`${error.message}. Reload the page.`);
  }
}

// Re-scan whenever X adds content (scrolling, new posts loading).
new MutationObserver(() => {
  if (!scanTimer) scanTimer = setTimeout(scanPage, SCAN_DELAY_MS);
}).observe(document.body, { childList: true, subtree: true });

scanPage();
console.log("[FeedAnalyzer] content script loaded");
