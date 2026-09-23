import { expect, test } from "@playwright/test";
import type { GraphView, RunMeta } from "../src/lib/types";
import { createRun, getJson, url } from "./util";

// Compose, run, open, rename and delete an intervention on the custom model; steer + generate on the tiny LM.
const LM = "py:examples.tiny_lm:build";
test.describe.configure({ mode: "serial" });

let run: RunMeta;
let conv: string;
let lm: RunMeta;
let block0: string;
let opened: string;
test.beforeAll(async ({ playwright }) => {
  const request = await playwright.request.newContext();
  run = await createRun(request);
  const view = await getJson<GraphView>(request, `/api/runs/${run.id}/graph?depth=12`);
  conv = view.nodes.find((n) => /conv/i.test(n.type) && n.shape.length === 4 && n.stats)!.output_op;
  lm = await createRun(request, { model: LM, sample: { kind: "text", text: "the cat sat on the mat" } });
  const top = await getJson<GraphView>(request, `/api/runs/${lm.id}/graph?depth=1`);
  block0 = top.nodes.find((n) => n.path === "transformer.h.0")!.output_op;
  await request.dispose();
});

const saved = (page: import("@playwright/test").Page) => page.getByRole("list", { name: "Saved interventions" }).getByRole("listitem");

test("compose a zero edit and run it", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/interventions?node=${encodeURIComponent(conv)}`));
  await expect(page.getByRole("group", { name: "Edit kind" }).getByRole("button", { name: "Zero" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Add to set" }).click();
  await page.getByRole("button", { name: "Run interventions" }).click();
  await expect(page.getByRole("region", { name: "Intervention result" })).toBeVisible({ timeout: 30_000 });
  await expect(page).toHaveURL(/[?&]iv=[0-9a-f]{12}/);
  await expect(saved(page)).toHaveCount(1);
  opened = page.url();
});

test("the result opens the graph coloured by Intervention Δ", async ({ page }) => {
  await page.goto(opened);
  await page.getByRole("link", { name: "Graph · Intervention Δ →" }).click();
  await expect(page).toHaveURL(/[?&]iv=/);
  await expect(page).toHaveURL(/[?&]color=iv/);
  await expect(page.getByText(/^Intervention Δ = /)).toBeVisible({ timeout: 30_000 });
});

test("clean vs intervened opens the tensor compare", async ({ page }) => {
  await page.goto(opened);
  await page.getByRole("link", { name: "Graph · Intervention Δ →" }).click();
  await expect(page).toHaveURL(/\/graph/);
  await page.goBack();
  await page.getByRole("link", { name: "Clean vs intervened →" }).click();
  await expect(page).toHaveURL(/\/tensors\?.*vs=clean/);
  await expect(page.getByText("A · clean").first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Δ = B − A")).toBeVisible();
});

test("rename, then delete the saved intervention", async ({ page }) => {
  await page.goto(opened);
  await saved(page).getByRole("button", { name: "Rename" }).click();
  const name = page.getByRole("textbox", { name: "Intervention name" });
  await name.fill("ablate conv");
  await name.press("Enter");
  await expect(page.getByRole("button", { name: "ablate conv" })).toBeVisible();
  await saved(page).getByRole("button", { name: "Delete", exact: true }).click();
  await page.getByRole("button", { name: "Confirm delete" }).click();
  await expect(page.getByText("No saved interventions · run a set to save one")).toBeVisible();
  await expect(page).not.toHaveURL(/[?&]iv=/);
});

test("steer the tiny LM and generate both continuations", async ({ page }) => {
  await page.goto(url(`#/runs/${lm.id}/interventions?node=${encodeURIComponent(block0)}`));
  await page.getByRole("group", { name: "Edit kind" }).getByRole("button", { name: "Steer" }).click();
  await page.getByRole("group", { name: "Direction" }).getByRole("button", { name: "Channel" }).click();
  await page.getByRole("spinbutton", { name: "Channel", exact: true }).fill("3");
  await page.getByRole("spinbutton", { name: "Alpha" }).fill("20");
  await page.getByRole("group", { name: "Positions", exact: true }).getByRole("button", { name: "Last" }).click();
  await page.getByRole("button", { name: "Add to set" }).click();
  await page.getByRole("button", { name: "Run interventions" }).click();
  await expect(page.getByRole("region", { name: "Intervention result" })).toBeVisible({ timeout: 30_000 });
  await page.getByRole("spinbutton", { name: "Tokens" }).fill("8");
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(page.getByRole("group", { name: "Clean continuation" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("group", { name: "Intervened continuation" })).toBeVisible();
  await expect(page.getByRole("table", { name: "Per-step top tokens" })).toBeVisible();
});
