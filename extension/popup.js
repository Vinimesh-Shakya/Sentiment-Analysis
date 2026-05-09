/**
 * popup.js
 * ========
 * Orchestrates the full one-click flow:
 *  1. On button click → ask content.js for the current tweet.
 *  2. Show loading skeleton while the FastAPI call is in flight.
 *  3. Render results: sentiment pill, confidence, pie chart, mini-bars.
 *  4. Handle all error / empty states gracefully.
 */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────
const API_URL = 'http://localhost:8000/predict';

/** Label map mirrors the Python backend */
const LABELS = {
  0: { text: 'Negative', emoji: '🔴', cssClass: 'pill-negative', color: '#f87171', barColor: '#f87171' },
  1: { text: 'Neutral',  emoji: '⚪', cssClass: 'pill-neutral',  color: '#94a3b8', barColor: '#94a3b8' },
  2: { text: 'Positive', emoji: '🟢', cssClass: 'pill-positive', color: '#34d399', barColor: '#34d399' },
};

// ─────────────────────────────────────────────────────────────────────────────
// DOM REFS
// ─────────────────────────────────────────────────────────────────────────────
const stateLoading = document.getElementById('state-loading');
const stateError   = document.getElementById('state-error');
const stateIdle    = document.getElementById('state-idle');
const stateResult  = document.getElementById('state-result');

const errorMessage   = document.getElementById('error-message');
const tweetPreview   = document.getElementById('tweet-preview');
const sentimentPill  = document.getElementById('sentiment-pill');
const confidenceVal  = document.getElementById('confidence-value');
const probBars       = document.getElementById('prob-bars');
const pieRing        = document.getElementById('pie-ring');
const pieLabel       = document.getElementById('pie-label');

const btnAnalyze = document.getElementById('btn-analyze');
const btnNext    = document.getElementById('btn-next');

// ─────────────────────────────────────────────────────────────────────────────
// STATE MANAGEMENT
// ─────────────────────────────────────────────────────────────────────────────

/** Show exactly one state panel; hide the rest. */
function showState(name) {
  const states = { loading: stateLoading, error: stateError, idle: stateIdle, result: stateResult };
  for (const [key, el] of Object.entries(states)) {
    el.hidden = key !== name;
  }
}

/** Disable/enable both buttons while loading. */
function setButtonsDisabled(disabled) {
  btnAnalyze.disabled = disabled;
  btnNext.disabled    = disabled;
}

// ─────────────────────────────────────────────────────────────────────────────
// TWEET SCRAPING  — talks to content.js via message passing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * getActiveTweetFromPage(resetHover)
 * Queries the active tab's content script for the most prominent tweet.
 *
 * @param {boolean} resetHover - pass true for "Analyze Next" so the hover
 *                               cache is cleared and a fresh tweet is picked.
 * @returns {Promise<{success, text?, preview?, error?}>}
 */
async function getActiveTweetFromPage(resetHover = false) {
  // Get the currently active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab) {
    return { success: false, error: 'No active tab found.' };
  }

  // Verify we are on Twitter / X
  const url = tab.url || '';
  if (!url.includes('twitter.com') && !url.includes('x.com')) {
    return {
      success: false,
      error: 'Please navigate to Twitter / X first, then click Analyze.',
    };
  }

  return new Promise((resolve) => {
    try {
      chrome.tabs.sendMessage(
        tab.id,
        { action: 'getTweet', resetHover },
        (response) => {
          if (chrome.runtime.lastError) {
            // Content script not yet injected (e.g., on a fresh page load)
            resolve({
              success: false,
              error:
                'Content script not ready. Please refresh the Twitter page and try again.',
            });
          } else {
            resolve(response);
          }
        }
      );
    } catch (err) {
      resolve({ success: false, error: err.message });
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// FASTAPI CALL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * fetchSentiment(text)
 * POSTs the tweet text to the local FastAPI backend.
 *
 * @param {string} text
 * @returns {Promise<{label_idx, label_str, confidence, probs}>}
 */
async function fetchSentiment(text) {
  const response = await fetch(API_URL, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errBody = await response.text();
    throw new Error(`API error ${response.status}: ${errBody}`);
  }

  return response.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// UI RENDERERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * renderResult(data, preview)
 * Populates the result panel with sentiment info, pie chart, and mini-bars.
 *
 * @param {{ label_idx, label_str, confidence, probs }} data
 * @param {string} preview  - truncated tweet text for the preview strip
 */
function renderResult(data, preview) {
  const { label_idx, confidence, probs } = data;
  const meta = LABELS[label_idx] ?? LABELS[1];

  // ── Tweet preview ──────────────────────────────────────────────────────
  tweetPreview.textContent = preview;

  // ── Sentiment pill ────────────────────────────────────────────────────
  sentimentPill.className    = `pill ${meta.cssClass}`;
  sentimentPill.textContent  = `${meta.emoji} ${meta.text}`;

  // ── Confidence text ───────────────────────────────────────────────────
  confidenceVal.textContent = `${confidence.toFixed(1)}%`;

  // ── Pie / ring chart ──────────────────────────────────────────────────
  const degrees = (confidence / 100) * 360;
  pieRing.style.setProperty('--pie-color', meta.color);
  pieRing.style.setProperty('--pie-deg',   `${degrees}deg`);
  pieLabel.textContent = `${confidence.toFixed(0)}%`;

  // ── Mini probability bars ─────────────────────────────────────────────
  probBars.innerHTML = '';

  const allProbs = probs ?? [0, 0, 0];   // fallback if backend doesn't return probs

  [0, 1, 2].forEach((idx) => {
    const val   = allProbs[idx] ?? 0;
    const lMeta = LABELS[idx];

    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <div class="prob-meta">
        <span>${lMeta.emoji} ${lMeta.text}</span>
        <span>${val.toFixed(1)}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width:0%; background:${lMeta.barColor};"></div>
      </div>
    `;
    probBars.appendChild(row);

    // Animate the bar width in on next tick
    requestAnimationFrame(() => {
      const fill = row.querySelector('.prob-fill');
      requestAnimationFrame(() => { fill.style.width = `${val}%`; });
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN ANALYSIS FLOW
// ─────────────────────────────────────────────────────────────────────────────

/**
 * runAnalysis(resetHover)
 * Full pipeline: scrape tweet → call API → render result.
 */
async function runAnalysis(resetHover = false) {
  showState('loading');
  setButtonsDisabled(true);

  try {
    // ── Step 1: get tweet from page ──────────────────────────────────────
    const tweetData = await getActiveTweetFromPage(resetHover);

    if (!tweetData.success) {
      errorMessage.textContent = tweetData.error;
      showState('error');
      return;
    }

    // ── Step 2: call FastAPI ─────────────────────────────────────────────
    let apiData;
    try {
      apiData = await fetchSentiment(tweetData.text);
    } catch (apiErr) {
      errorMessage.textContent =
        `Backend error: ${apiErr.message}. ` +
        'Make sure the FastAPI server is running on http://localhost:8000';
      showState('error');
      return;
    }

    // ── Step 3: render results ───────────────────────────────────────────
    renderResult(apiData, tweetData.preview);
    showState('result');

    // Show the "Analyze Next" button after first successful result
    btnNext.hidden = false;

  } finally {
    setButtonsDisabled(false);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// EVENT LISTENERS
// ─────────────────────────────────────────────────────────────────────────────

btnAnalyze.addEventListener('click', () => runAnalysis(false));

btnNext.addEventListener('click', () => runAnalysis(true));

// ─────────────────────────────────────────────────────────────────────────────
// INIT — show idle state when popup opens
// ─────────────────────────────────────────────────────────────────────────────
showState('idle');
