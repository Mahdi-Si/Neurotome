import { expect, test } from "@playwright/test";
import type { GraphView, RunMeta } from "../src/lib/types";
import { createRun, getJson, url } from "./util";

let run: RunMeta;
let conv: string;
test.beforeAll(async ({ playwright }) => {
  const request = await playwright.request.newContext();
  run = await createRun(request);
  const view = await getJson<GraphView>(request, `/api/runs/${run.id}/graph?depth=12`);
  conv = view.nodes.find((n) => /conv/i.test(n.type) && n.shape.length === 4 && n.stats)!.id;
  await request.dispose();
});

test("integrated gradients, then method agreement", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/trace`));
  await page.getByRole("group", { name: "Attribution method" }).getByRole("button", { name: "Integrated grad" }).click();
  await page.getByRole("button", { name: "Run attribution" }).click();
  const progress = page.getByRole("progressbar", { name: "attribution job" });
  await expect(progress.or(page.getByText("Completeness Δ")).first()).toBeVisible();
  await expect(page.getByText("Completeness Δ")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Compare methods" }).click();
  await expect(page.getByRole("img", { name: /· jaccard$/ }).first()).toBeVisible({ timeout: 30_000 });
});

test("patch & re-run colours the graph by Patch Δ", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/graph?depth=12&node=${encodeURIComponent(conv)}`));
  await page.getByRole("button", { name: "Patch & re-run" }).click();
  await expect(page).toHaveURL(/color=patch/);
  await expect(page.getByText(/^Patch Δ = /)).toBeVisible({ timeout: 30_000 });
});
