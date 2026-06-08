// Runtime API base. For local dev leave undefined or empty.
// For static hosting (e.g., GitHub Pages), point this to your deployed backend.
// Example: window.RISA_API_BASE = 'https://your-backend.example.com';

(function () {
	const hostname = window.location.hostname;
	const protocol = window.location.protocol;
	const isLocal = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '0.0.0.0';
	const isFileProtocol = protocol === 'file:';
	const defaultRemote = 'https://your-deployed-backend.up.railway.app'; // replace with your Railway URL
	const defaultLocalApi = 'http://127.0.0.1:8000';

	// For local dev, prefer same-origin requests when served over HTTP.
	// For file:// previews, point directly at the local backend.
	// For static hosting, fall back to the deployed backend.
	window.RISA_API_BASE = window.RISA_API_BASE || (isLocal ? '' : isFileProtocol ? defaultLocalApi : defaultRemote);
})();
