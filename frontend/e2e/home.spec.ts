import { expect, test, type Page } from "@playwright/test";
import { MODEL, url } from "./util";

const capture = (page: Page) => page.getByRole("button", { name: /Capture forward pass/ });

test("capture from the Home form runs a capture job, shows its progress and opens the graph", async ({ page }) => {
  // hold job updates briefly so the progress UI is on screen however fast the capture is
  await page.route("**/api/jobs/**", async (r) => { await new Promise((ok) => setTimeout(ok, 1500)); await r.continue(); });
  await page.goto(url("#/"));
  await page.getByLabel("Model", { exact: true }).fill(MODEL);
  const post = page.waitForResponse((r) => r.url().endsWith("/api/runs") && r.request().method() === "POST");
  await capture(page).click();
  const res = await post;
  expect(res.status()).toBe(202);
  expect((await res.json()).kind).toBe("capture");
  await expect(page.getByRole("button", { name: /Capturing/ })).toBeDisabled();
  await expect(page.getByRole("progressbar", { name: "capture job" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible();
  await expect(page).toHaveURL(/#\/runs\/run_[^/]+\/graph/);
  await expect(page.getByText("Forward trace")).toBeVisible();
});

test("a model spec that cannot load shows the job error", async ({ page }) => {
  await page.goto(url("#/"));
  await page.getByLabel("Model", { exact: true }).fill("py:nope_missing:build");
  await capture(page).click();
  await expect(page.locator(".err")).toContainText("Cannot load model");
  await expect(page).toHaveURL(/#\/$/);
});

test("selective capture shows the Selective banner", async ({ page }) => {
  await page.goto(url("#/"));
  await page.getByLabel("Model", { exact: true }).fill(MODEL);
  await page.getByLabel("Capture modules").fill("blocks.0");
  await capture(page).click();
  await expect(page).toHaveURL(/#\/runs\/run_[^/]+\/graph/);
  await expect(page.getByText("Selective", { exact: true })).toBeVisible();
});

test("the theme toggle switches to dark and survives a reload", async ({ page }) => {
  await page.goto(url("#/"));
  await page.getByRole("button", { name: "Switch to dark theme" }).first().click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("button", { name: "Switch to light theme" }).first()).toBeVisible();
});
