// Relative paths: nginx.conf proxies /api/ -> backend and /ai/ -> ai-service
// over the Docker network, so the browser only ever talks to one origin
// (wherever the frontend container itself is exposed) regardless of what
// host port backend/ai-service are published on.
export const environment = {
  production: true,
  apiBaseUrl: '/api',
  aiServiceBaseUrl: '/ai',
};
