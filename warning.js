// Get blocked URL from query parameter
const urlParams = new URLSearchParams(window.location.search);
const blockedUrl = urlParams.get('blocked');

if (blockedUrl) {
    document.getElementById('blockedUrl').textContent = decodeURIComponent(blockedUrl);
}

// Proceed anyway button (removed go back button)
document.getElementById('proceedBtn').addEventListener('click', function() {
    if (blockedUrl) {
        const actualUrl = decodeURIComponent(blockedUrl);
        
        // Tell background script to whitelist this URL
        chrome.runtime.sendMessage({
            action: "whitelistUrl",
            url: actualUrl
        }, function() {
            // Now navigate to the URL
            window.location.href = actualUrl;
        });
    }
});