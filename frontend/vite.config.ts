import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Dev: the UI runs on :5173 and proxies /api to the Python server on :7070.
// Build: output goes straight into the Python package so `neurotome serve` ships the UI.
// Test: vitest runs the unit tests in src/ only; e2e/ is Playwright's (npm run e2e).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:7070" },
  },
  build: {
    outDir: "../backend/neurotome/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 2000,
  },
  test: { include: ["src/**/*.test.ts"], css: { include: [/\?raw$/] } },
});
