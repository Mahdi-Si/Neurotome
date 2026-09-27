import { expect, test } from "@playwright/test";
import type { GraphView, RunMeta } from "../src/lib/types";
import { createRun, getJson, url } from "./util";

// Text runs of a tiny character-level GPT-2 (offline; needs the hf extra): attention, logit lens, token attribution.
const LM = "py:examples.tiny_lm:build";
const TEXT = "the cat sat on the mat";

let run: RunMeta;
let softmax: string;
test.beforeAll(async ({ playwright }) => {
  const request = await playwright.request.newContext();
  run = await createRun(request, { model: LM, sample: { kind: "text", text: TEXT } });
  const view = await getJson<GraphView>(request, `/api/runs/${run.id}/graph?depth=12`);
  softmax = view.nodes.find((n) => n.type === "softmax" && n.stats)!.id;
  await request.dispose();
});

test("a text sample from the Home form opens the graph", async ({ page }) => {
  await page.goto(url("#/"));
  await page.getByLabel("Model", { exact: true }).fill(LM);
  await page.getByRole("group", { name: "Sample kind" }).getByRole("button", { name: "Text", exact: true }).click();
  await page.getByLabel("Sample text").fill(TEXT);
  await page.getByRole("button", { name: /Capture forward pass/ }).click();
  await expect(page).toHaveURL(/#\/runs\/run_[^/]+\/graph/, { timeout: 45_000 });
});

test("a softmax op opens as attention heads", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/tensors?node=${encodeURIComponent(softmax)}`));
  await expect(page.getByRole("heading", { name: "Attention" })).toBeVisible();
  const head = page.getByRole("button", { name: /^Head 1, entropy .* bits$/ });
  await head.click();
  await expect(head).toHaveAttribute("aria-pressed", "true");
  await expect(page).toHaveURL(/[?&]c=1/);
  await expect(page.getByRole("img", { name: "Head 1 attention" })).toBeVisible();
  await expect(page.getByRole("table", { name: "Head summary" })).toBeVisible();
});

test("the logit lens runs from the Lens tab", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/graph`));
  await page.getByRole("link", { name: "Lens", exact: true }).click();
  await expect(page).toHaveURL(/\/lens/);
  await page.getByRole("button", { name: "Run lens" }).click();
  await expect(page.getByRole("img", { name: "Logit lens" })).toBeVisible({ timeout: 30_000 });
  await page.getByRole("group", { name: "Lens colour" }).getByRole("button", { name: "Target", exact: true }).click();
  await expect(page).toHaveURL(/mode=target/);
});

test("integrated gradients on a token input fill the token strip", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/trace`));
  await page.getByRole("button", { name: "Run attribution" }).first().click();
  await expect(page.getByRole("group", { name: "Token attribution" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Completeness Δ")).toBeVisible();
});

test("a model that is not a language model has the Lens tab disabled", async ({ page, playwright }) => {
  const request = await playwright.request.newContext();
  const other = await createRun(request);
  await request.dispose();
  await page.goto(url(`#/runs/${other.id}/graph`));
  await expect(page.getByText("Lens", { exact: true })).toHaveAttribute("aria-disabled", "true");
});
