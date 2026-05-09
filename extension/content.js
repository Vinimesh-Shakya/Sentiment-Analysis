/**
 * content.js
 * ==========
 * Injected into Twitter/X pages. Responsible for:
 *  1. Detecting the most prominent (viewport-centre) tweet on screen.
 *  2. Extracting its clean text.
 *  3. Responding to messages from popup.js via chrome.runtime.onMessage.
 *
 * Twitter uses a React-based virtual DOM, so we must be resilient to
 * dynamic rendering and rely on stable data-testid attributes.
 */

// ─────────────────────────────────────────────────────────────────────────────
// SELECTORS  — based on Twitter's stable test-ids
// ─────────────────────────────────────────────────────────────────────────────
const TWEET_ARTICLE_SELECTOR = 'article[data-testid="tweet"]';
const TWEET_TEXT_SELECTOR    = '[data-testid="tweetText"]';

// ─────────────────────────────────────────────────────────────────────────────
// HOVER TRACKING
// Keep a reference to the last tweet the user hovered over.
// This is used as the primary source; viewport-centre is the fallback.
// ─────────────────────────────────────────────────────────────────────────────
let lastHoveredTweet = null;

document.addEventListener('mouseover', (e) => {
  const article = e.target.closest(TWEET_ARTICLE_SELECTOR);
  if (article) {
    lastHoveredTweet = article;
  }
}, { passive: true });


// ─────────────────────────────────────────────────────────────────────────────
// TWEET EXTRACTION HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * extractTextFromArticle(article)
 * Pulls the visible tweet text from a tweet <article> element.
 * Falls back to innerText if the specific testid span isn't found.
 *
 * @param {HTMLElement} article
 * @returns {string|null}
 */
function extractTextFromArticle(article) {
  if (!article) return null;

  // Primary: Twitter annotates the tweet body with this test-id
  const textSpan = article.querySelector(TWEET_TEXT_SELECTOR);
  if (textSpan) {
    return textSpan.innerText.trim() || null;
  }

  // Fallback: scrape all visible text (noisier but works if testid changes)
  const rawText = article.innerText.trim();
  return rawText.length > 0 ? rawText.split('\n').filter(Boolean).join(' ') : null;
}


/**
 * distanceToViewportCentre(element)
 * Returns the absolute distance (px) of an element's centre from the
 * vertical mid-point of the current viewport.
 *
 * @param {HTMLElement} element
 * @returns {number}
 */
function distanceToViewportCentre(element) {
  const rect         = element.getBoundingClientRect();
  const elementCentreY = rect.top + rect.height / 2;
  const viewportCentreY = window.innerHeight / 2;
  return Math.abs(elementCentreY - viewportCentreY);
}


/**
 * getMostProminentTweet()
 * Priority order:
 *  1. Last tweet the user hovered over (if still visible)
 *  2. Tweet whose centre is closest to the viewport centre
 *
 * @returns {{ text: string, preview: string } | null}
 */
function getMostProminentTweet() {
  const articles = Array.from(document.querySelectorAll(TWEET_ARTICLE_SELECTOR));

  if (articles.length === 0) return null;

  // ── Priority 1: hovered tweet ──────────────────────────────────────────
  if (lastHoveredTweet && document.contains(lastHoveredTweet)) {
    const text = extractTextFromArticle(lastHoveredTweet);
    if (text) {
      return {
        text,
        preview: text.slice(0, 120) + (text.length > 120 ? '…' : ''),
        source: 'hover',
      };
    }
  }

  // ── Priority 2: viewport-centre tweet ─────────────────────────────────
  // Filter to only partially-visible tweets
  const visibleArticles = articles.filter((a) => {
    const rect = a.getBoundingClientRect();
    return rect.bottom > 0 && rect.top < window.innerHeight;
  });

  if (visibleArticles.length === 0) return null;

  // Sort by distance to viewport centre; pick the closest
  visibleArticles.sort(
    (a, b) => distanceToViewportCentre(a) - distanceToViewportCentre(b)
  );

  for (const article of visibleArticles) {
    const text = extractTextFromArticle(article);
    if (text) {
      return {
        text,
        preview: text.slice(0, 120) + (text.length > 120 ? '…' : ''),
        source: 'viewport',
      };
    }
  }

  return null;
}


// ─────────────────────────────────────────────────────────────────────────────
// MESSAGE LISTENER  — popup.js talks to us here
// ─────────────────────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request.action === 'getTweet') {
    // Reset hover cache when "Analyze Next" is clicked
    if (request.resetHover) {
      lastHoveredTweet = null;
    }

    const result = getMostProminentTweet();

    if (result) {
      sendResponse({ success: true, ...result });
    } else {
      sendResponse({
        success: false,
        error:
          'No tweet detected in the current viewport. ' +
          'Please scroll until a tweet is visible, then try again.',
      });
    }

    // Return true to indicate we'll respond asynchronously if needed
    return true;
  }
});
