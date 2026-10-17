import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Dev: Vite serves the SPA on :5177 and proxies the API to FastAPI on :8085 (both 127.0.0.1,
// D9). Built, FastAPI serves the SPA from the same origin. A portfolio-site build sets
// SHOWCASE_BASE (nmp-central-ai s06), e.g. /p/tau-telecom-sft/; the router reads it back as
// import.meta.env.BASE_URL.
const API = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8085';

export default defineConfig({
  base: process.env.SHOWCASE_BASE ?? '/',
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5177,
    strictPort: true,
    // The house UI lives in ../.lavish (house.css, project.css, fonts/), imported from
    // src/main.tsx. Dev serving needs permission to read it; the build bundles it.
    fs: { allow: ['..'] },
    proxy: {
      '/api': { target: API, changeOrigin: false },
      '/healthz': { target: API, changeOrigin: false },
    },
  },
});
