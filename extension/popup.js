// Set this once after deploying the backend. Leave it empty for local-only use.
const DEPLOYED_API_BASE = "https://transcript-chat.onrender.com";
const LOCAL_API_BASE = "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 60000; // fetch() never times out on its own - without this a hung backend looks like the UI "stopped."
const HEALTH_CHECK_TIMEOUT_MS = 1500;
const MAX_HISTORY_PER_VIDEO = 30;
const CONTEXT_TURNS_PER_REQUEST = 6;

const statusEl = document.getElementById("status");
const threadEl = document.getElementById("thread");
const emptyStateEl = document.getElementById("emptyState");
const questionEl = document.getElementById("question");
const askBtn = document.getElementById("ask");
const clearBtn = document.getElementById("clear");
const themeToggle = document.getElementById("themeToggle");
const sunIcon = document.getElementById("sunIcon");
const moonIcon = document.getElementById("moonIcon");
const quickQuestionBtns = document.querySelectorAll(".quick-question");

let currentVideoId = null;
let history = []; // [{question, answer}, ...]
let apiBase = LOCAL_API_BASE;

const THEME_KEY = "theme"; // global preference, not per-video

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const isDark = theme === "dark";
  sunIcon.style.display = isDark ? "none" : "block";
  moonIcon.style.display = isDark ? "block" : "none";
}

async function loadTheme() {
  const stored = await chrome.storage.local.get(THEME_KEY);
  const theme = stored[THEME_KEY] || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  applyTheme(theme);
}

async function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  await chrome.storage.local.set({ [THEME_KEY]: next });
}

function extractVideoId(url) {
  try {
    const u = new URL(url);
    if (u.hostname === "youtu.be") return u.pathname.slice(1);
    if (u.searchParams.has("v")) return u.searchParams.get("v");
    return null;
  } catch {
    return null;
  }
}

function historyKey(videoId) {
  return `history:${videoId}`;
}

function renderThread() {
  threadEl.innerHTML = "";

  if (history.length === 0) {
    threadEl.appendChild(emptyStateEl);
    return;
  }

  for (const { question, answer, timestamp } of history) {
    const wrap = document.createElement("div");
    wrap.className = "turn";

    const qRow = document.createElement("div");
    qRow.className = "q-row";
    const qBubble = document.createElement("div");
    qBubble.className = "q-bubble";
    qBubble.textContent = question;
    qRow.appendChild(qBubble);

    const time = document.createElement("div");
    time.className = "turn-time";
    time.textContent = formatTimestamp(timestamp);

    const aRow = document.createElement("div");
    aRow.className = "a-row";
    const aBubble = document.createElement("div");
    aBubble.className = "a-bubble";
    aBubble.textContent = answer;
    aRow.appendChild(aBubble);

    wrap.appendChild(qRow);
    if (time.textContent) wrap.appendChild(time);
    wrap.appendChild(aRow);
    threadEl.appendChild(wrap);
  }

  threadEl.scrollTop = threadEl.scrollHeight;
}

function formatTimestamp(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "turn";
  row.id = "typingTurn";
  row.innerHTML = `
    <div class="typing-row">
      <div class="typing-bubble"><span></span><span></span><span></span></div>
    </div>
  `;
  threadEl.appendChild(row);
  threadEl.scrollTop = threadEl.scrollHeight;
}

function hideTyping() {
  document.getElementById("typingTurn")?.remove();
}

function showError(message) {
  hideTyping();
  const row = document.createElement("div");
  row.className = "turn";
  row.innerHTML = `<div class="error-card">${message}</div>`;
  threadEl.appendChild(row);
  threadEl.scrollTop = threadEl.scrollHeight;
}

async function init() {
  await loadTheme(); // apply before any content renders, to avoid a flash
  await selectApiBase();

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentVideoId = tab?.url ? extractVideoId(tab.url) : null;

  if (!currentVideoId) {
    statusEl.textContent = "Open a YouTube video to use this";
    askBtn.disabled = true;
    questionEl.disabled = true;
    setQuickQuestionState(true);
    return;
  }

  statusEl.textContent = "Video detected — ready to answer questions";

  const key = historyKey(currentVideoId);
  const stored = await chrome.storage.local.get(key);
  history = stored[key] || [];
  renderThread();
}

async function saveHistory() {
  const key = historyKey(currentVideoId);
  if (history.length > MAX_HISTORY_PER_VIDEO) {
    history = history.slice(history.length - MAX_HISTORY_PER_VIDEO);
  }
  await chrome.storage.local.set({ [key]: history });
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function isBackendAvailable(baseUrl) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);
  try {
    const response = await fetch(`${baseUrl}/health`, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function selectApiBase() {
  const candidates = [LOCAL_API_BASE, DEPLOYED_API_BASE].filter(Boolean);
  for (const candidate of candidates) {
    const baseUrl = candidate.replace(/\/$/, "");
    if (await isBackendAvailable(baseUrl)) {
      apiBase = baseUrl;
      return;
    }
  }
}

function setQuickQuestionState(disabled) {
  quickQuestionBtns.forEach((button) => {
    button.disabled = disabled;
  });
}

async function askQuestion(questionOverride) {
  const question = (questionOverride || questionEl.value).trim();
  if (!question || !currentVideoId) return;

  if (history.length === 0) threadEl.innerHTML = ""; // drop empty state
  askBtn.disabled = true;
  setQuickQuestionState(true);
  questionEl.value = "";
  autoGrow();
  showTyping();

  try {
    const res = await fetchWithTimeout(
      `${apiBase}/query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: currentVideoId,
          question,
          history: history.slice(-CONTEXT_TURNS_PER_REQUEST).map(({ question, answer }) => ({ question, answer })),
        }),
      },
      REQUEST_TIMEOUT_MS
    );

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Something went wrong.");

    hideTyping();
    history.push({ question, answer: data.answer, timestamp: new Date().toISOString() });
    renderThread();
    await saveHistory();
  } catch (err) {
    const isTimeout = err.name === "AbortError";
    const message = isTimeout
      ? `Request took over ${REQUEST_TIMEOUT_MS / 1000}s and was cancelled — the model may be slow right now. Try again.`
      : `${err.message} — is the backend running at ${apiBase}?`;
    showError(message);
  } finally {
    askBtn.disabled = false;
    setQuickQuestionState(false);
  }
}

function autoGrow() {
  questionEl.style.height = "auto";
  questionEl.style.height = Math.min(questionEl.scrollHeight, 90) + "px";
}

askBtn.addEventListener("click", askQuestion);
quickQuestionBtns.forEach((button) => {
  button.addEventListener("click", () => askQuestion(button.dataset.question));
});
themeToggle.addEventListener("click", toggleTheme);
clearBtn.addEventListener("click", async () => {
  history = [];
  renderThread();
  if (currentVideoId) await chrome.storage.local.remove(historyKey(currentVideoId));
});
questionEl.addEventListener("input", autoGrow);
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});

init();
