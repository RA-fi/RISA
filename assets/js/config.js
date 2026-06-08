// Runtime API base. For local dev leave undefined or empty.
// For static hosting (e.g., GitHub Pages), point this to your deployed backend.
// Example: window.RISA_API_BASE = 'https://your-backend.example.com';

(function () {
	const hostname = window.location.hostname;
	const protocol = window.location.protocol;
	const isLocal = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '0.0.0.0';
	const isFileProtocol = protocol === 'file:';
	const defaultLocalApi = 'http://127.0.0.1:8000';

	// Railway serves the frontend and backend from the same origin, so the
	// default should be same-origin unless a custom API base is explicitly set.
	// file:// previews still need a direct local backend URL.
	window.RISA_API_BASE = window.RISA_API_BASE || (isLocal ? '' : isFileProtocol ? defaultLocalApi : '');
})();
