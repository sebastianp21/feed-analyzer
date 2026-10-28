// Feed Analyzer popup: shows the category split of the last 50 posts,
// the server status, and the server address setting.

const api = globalThis.browser ?? chrome;

// Same default as background.js.
const DEFAULT_SERVER_URL = "http://localhost:8000";
const HEALTH_TIMEOUT_MS = 3000;

// Display order (same as the tie rule) and names.
const CATEGORIES = [
  { id: "outrage", label: "Outrage" },
  { id: "clickbait", label: "Clickbait" },
  { id: "news", label: "News" },
  { id: "other", label: "Other" },
];

const $ = (id) => document.getElementById(id);

// ---------- Stats ----------

async function renderStats() {
  const { stats } = await api.storage.session.get("stats");
  const recent = stats?.recent ?? [];
  const skipped = stats?.skipped ?? 0;
  const total = recent.length;

  // Count posts per category.
  const counts = Object.fromEntries(CATEGORIES.map((c) => [c.id, 0]));
  for (const result of recent) {
    if (result.category in counts) counts[result.category]++;
  }
  const percent = (id) => (total ? Math.round((counts[id] * 100) / total) : 0);

  // Summary line, e.g. "Mostly news (42%)".
  if (total === 0) {
    $("summary").textContent = "No posts analyzed yet.";
    $("based-on").textContent = "Scroll your X feed to start.";
  } else {
    const top = CATEGORIES.reduce((best, c) => (counts[c.id] > counts[best.id] ? c : best));
    $("summary").textContent = `Mostly ${top.label.toLowerCase()} (${percent(top.id)}%)`;
    $("based-on").textContent = `Based on the last ${total} English posts`;
  }

  // One bar per category.
  $("bars").replaceChildren(
    ...CATEGORIES.map((c) => {
      const row = document.createElement("li");
      const name = document.createElement("span");
      const track = document.createElement("div");
      const fill = document.createElement("div");
      const value = document.createElement("span");

      name.textContent = c.label;
      track.className = "track";
      fill.className = "fill";
      fill.style.width = `${percent(c.id)}%`;
      fill.style.background = `var(--${c.id})`;
      value.className = "percent";
      value.textContent = `${percent(c.id)}%`;

      track.append(fill);
      row.append(name, track, value);
      return row;
    })
  );

  $("skipped").textContent = skipped ? `${skipped} non-English posts skipped` : "";
  return stats?.lastError ?? null;
}

// ---------- Server status and errors ----------

async function getServerUrl() {
  const { serverUrl } = await api.storage.local.get("serverUrl");
  return serverUrl || DEFAULT_SERVER_URL;
}

// Returns null if the server answers /health, otherwise an error message.
async function checkServer(serverUrl) {
  try {
    const response = await fetch(`${serverUrl}/health`, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    if (!response.ok) return `Server at ${serverUrl} answered with error ${response.status}.`;
    return null;
  } catch {
    return `Can't reach the server at ${serverUrl}. Start it with: uvicorn app.main:app --port 8000`;
  }
}

function showError(message) {
  $("error").hidden = !message;
  $("error").textContent = message ?? "";
}

async function refresh() {
  const serverUrl = await getServerUrl();
  const [lastError, healthError] = await Promise.all([renderStats(), checkServer(serverUrl)]);

  $("server-status").textContent = healthError ? "● server offline" : "● server online";
  $("server-status").className = `status ${healthError ? "bad" : "ok"}`;
  // A failed health check matters most; otherwise show the last classify error, if any.
  showError(healthError ?? (lastError ? `Last request failed: ${lastError}` : null));
}

// ---------- Settings ----------

$("settings").addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = $("server-url").value.trim().replace(/\/+$/, ""); // drop trailing "/"
  const url = new URL(value);

  // The extension only has permission to talk to localhost (see manifest.json).
  if (!["localhost", "127.0.0.1"].includes(url.hostname)) {
    showError("The server address must be on localhost or 127.0.0.1.");
    return;
  }
  await api.storage.local.set({ serverUrl: value });
  refresh();
});

$("reset").addEventListener("click", () => {
  api.storage.session.set({ stats: { recent: [], skipped: 0, lastError: null } });
});

// Update live while the popup is open and new results come in.
api.storage.onChanged.addListener((changes, area) => {
  if (area === "session" && changes.stats) refresh();
});

getServerUrl().then((url) => ($("server-url").value = url));
refresh();
