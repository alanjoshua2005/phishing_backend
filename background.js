const API_URL = "https://phishing-backend-exbz.onrender.com/predict";

let extensionEnabled = true;
let checkedUrls = {};
let whitelistedUrls = {};

// Listen for extension icon click to toggle on/off
chrome.action.onClicked.addListener(() => {
  extensionEnabled = !extensionEnabled;
  
  if (extensionEnabled) {
    chrome.action.setIcon({ path: "icon.png" });
  } else {
    chrome.action.setIcon({ path: "icon.png" });
  }
  
  chrome.storage.local.set({ enabled: extensionEnabled });
});

// Listen for messages from warning page
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "whitelistUrl") {
    whitelistedUrls[message.url] = Date.now();
    sendResponse({ success: true });
  }
  return true;
});

// Load initial state
chrome.storage.local.get("enabled", (result) => {
  if (result.enabled !== undefined) {
    extensionEnabled = result.enabled;
  }
});

// Check URL when navigation starts
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  // Only check main frame (not iframes)
  if (details.frameId !== 0) return;
  
  // Skip if extension is disabled
  if (!extensionEnabled) return;
  
  const url = details.url;
  const tabId = details.tabId;
  
  // Skip non-http URLs (chrome://, about:, etc.)
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    return;
  }
  
  // Skip if user whitelisted this URL
  if (whitelistedUrls[url]) {
    return;
  }
  
  // Skip if already checked recently
  if (checkedUrls[url] && Date.now() - checkedUrls[url] < 300000) {
    return;
  }
  
  try {
    // Call backend API
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url: url })
    });
    
    const data = await response.json();
    
    // Simple if-else logic
    if (data.prediction === "phishing") {
      // Block and show warning page
      chrome.tabs.update(tabId, {
        url: chrome.runtime.getURL("warning.html") + "?blocked=" + encodeURIComponent(url)
      });
    } else {
      // Allow - mark as checked
      checkedUrls[url] = Date.now();
    }
    
  } catch (error) {
    console.error("Error checking URL:", error);
    // On error, allow the navigation
  }
});