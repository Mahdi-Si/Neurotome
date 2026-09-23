import { defineConfig, devices } from "@playwright/test";

// Browser end-to-end tests over the built UI. global-setup starts `neurotome serve` on a free port with a temp
// runs dir and exports its URL as NEUROTOME_E2E_URL. Run with `npm run e2e` (builds the UI first).
export default defineConfig({
  testDir: "e2e",
  globalSetup: "./e2e/global-setup.ts",
  workers: 1,
  fullyParallel: false,
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 900 },
    colorScheme: "light",   // the theme test toggles light → dark
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
