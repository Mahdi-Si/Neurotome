import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the UI runs on :5173 and proxies /api to the Python server on :7070.
// Build: output goes straight into the Python package so `dissect serve` ships the UI.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:7070" },
  },
  build: {
    outDir: "../backend/dissect/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 2000,
  },
});
