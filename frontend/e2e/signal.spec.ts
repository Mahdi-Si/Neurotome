import { expect, test } from "@playwright/test";
import type { RunMeta } from "../src/lib/types";
import { createRun, getJson, url } from "./util";

// a time-series model with a dataset (list_samples / load_sample): 2 raw channels at 4 Hz → a (1, 74, 10) band-power input
const SIGNAL = "py:examples.signal_model:build";

let run: RunMeta;
test.beforeAll(async ({ playwright }) => {
  const request = await playwright.request.newContext();
  run = await createRun(request, { model: SIGNAL, sample: { kind: "ref", ref: "rec02:1" } });
  await request.dispose();
});

test("a dataset sample shows its raw signal and the feature blocks it became, on one clock", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/tensors?node=input_1`));
  await expect(page.getByRole("heading", { name: "Input" })).toBeVisible();
  const sample = page.getByRole("group", { name: "Sample" });
  await expect(sample.getByText("rec02:1", { exact: true })).toBeVisible();
  await expect(sample.getByText("no dips", { exact: true })).toBeVisible();
  const raw = page.getByRole("group", { name: "Raw input" });
  await expect(raw.getByText("rec02 · segment 1 · 2 channels · 2400 samples at 4 Hz")).toBeVisible();
  for (const ch of ["hr", "pressure"]) await expect(raw.getByRole("img", { name: `Raw ${ch}` })).toBeVisible();
  await expect(raw.getByText("hr · bpm")).toBeVisible();
  // both cards: x = minutes from recording start, 10 … 20 (the 600–1200 s segment)
  const model = page.getByRole("group", { name: "Model input" });
  for (const g of [raw, model]) {
    await expect(g.getByText("min from recording start").first()).toBeVisible();
    await expect(g.getByText("10", { exact: true }).first()).toBeVisible();
    await expect(g.getByText("20", { exact: true }).first()).toBeVisible();
  }
  await expect(model.getByText(/1×74×10 · float32/)).toBeVisible();
  const hr = model.getByRole("img", { name: "hr · band power" });
  await expect(hr).toBeVisible();
  await expect(model.getByRole("img", { name: "pressure · band power" })).toBeVisible();
  await expect(model.getByText("0.0625–0.125 Hz").first()).toBeVisible();   // row labels
  const box = (await hr.boundingBox())!;
  await page.mouse.move(box.x + 3, box.y + 3);
  await expect(model.getByText(/t 10\.\d+ min · step 0 · "0\.0625–0\.125 Hz" · /)).toBeVisible();
  // the raw plot starts at 600 s, the first model step (a 16 s frame) at 604 s: the heatmap sits a little right
  const rawBox = (await raw.getByRole("img", { name: "Raw hr" }).boundingBox())!;
  expect(box.x).toBeGreaterThan(rawBox.x);
  expect(box.x - rawBox.x).toBeLessThan(0.02 * rawBox.width);
  await expect(page.getByRole("group", { name: "Preprocessing" }).getByText(/^Cut segment 1 of rec02/)).toBeVisible();
  await expect(page.getByRole("table", { name: "Model input values" }).getByText("pressure · band power")).toBeVisible();
});

test("the output of a values model shows scores, not probabilities", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/trace`));
  const first = run.output[0];
  expect(first.prob).toBeNull();
  // the top-5 stay quick picks, labelled by the module's CATEGORIES ("t40 · +5:28")
  await expect(page.getByRole("option", { name: new RegExp(`^logit\\[${first.index}\\] · t${first.index} · \\+\\d+:\\d\\d · value `) })).toBeAttached();
});

test("any step can be the target; its input windows sit on the map with where the attribution is", async ({ page }) => {
  const step = Array.from({ length: 74 }, (_, i) => i).find((i) => i >= 30 && !run.output.some((o) => o.index === i))!;
  await page.goto(url(`#/runs/${run.id}/trace?m=gxa`));
  const target = page.getByRole("group", { name: "Target" });
  await target.getByLabel("Output index", { exact: true }).fill(String(step));
  await expect(page).toHaveURL(new RegExp(`[?&]t=${step}(&|$)`));
  await expect(target.getByText(new RegExp(`^t${step} · \\+\\d+:\\d\\d · value -?\\d\\.\\d{4} · of 74$`))).toBeVisible();
  await expect(target.getByLabel("Output index slider")).toHaveValue(String(step));
  await page.getByRole("button", { name: "Run attribution" }).first().click();
  const ts = page.getByRole("group", { name: "Time-series attribution" });
  await expect(ts.getByRole("img", { name: "Input attribution" })).toBeVisible({ timeout: 30_000 });
  const legend = ts.getByRole("list", { name: "Input windows" });
  await expect(legend.getByText(`Step ${step}`, { exact: true })).toBeVisible();
  await expect(legend.getByText(`pressure context (12 steps) · steps ${step - 12}–${step - 1} · pressure · band power`)).toBeVisible();
  const where = ts.getByRole("table", { name: "Where the attribution is" });
  await expect(where.getByRole("row")).toHaveCount(5);   // header, target, context, horizon, outside
  await expect(where.getByText("outside all windows")).toBeVisible();
  const shares = (await where.locator("tbody tr td:nth-child(3)").allTextContents()).map(parseFloat);
  expect(shares).toHaveLength(4);
  expect(shares.every((v) => v >= 0 && v <= 100)).toBe(true);
  // zoom around the step: the x axis narrows from 608 … 1192 s to the windows ± a margin
  const xFrom = ts.getByText("608", { exact: true }).first();
  await expect(xFrom).toBeVisible();
  await ts.getByRole("group", { name: "Zoom" }).getByRole("button", { name: "Around step" }).click();
  await expect(ts.getByText(/zoomed to steps \d+–\d+/)).toBeVisible();
  await expect(ts.getByText("608", { exact: true })).toHaveCount(0);
  // a window's values: the input node in the Tensors view, the Values panel prefilled with its slice
  await where.getByRole("link", { name: /^Show input values · pressure context/ }).click();
  await expect(page).toHaveURL(/#\/runs\/run_[^/]+\/tensors\?node=input_1&s=/);
  await expect(page.getByLabel("Slice expression")).toHaveValue(`[0, ${step - 12}:${step}, 5:10]`);
  await expect(page.getByText(/· shape 12×5/)).toBeVisible();
});

test("a clicked map cell shows its input values, and Show on input draws the step on the input", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/trace?m=gxa&t=40&go=1`));
  const ts = page.getByRole("group", { name: "Time-series attribution" });
  const map = ts.getByRole("img", { name: "Input attribution" });
  await expect(map).toBeVisible({ timeout: 30_000 });
  const box = (await map.boundingBox())!;
  await page.mouse.click(box.x + 3, box.y + box.height - 3);   // step 0, the last pressure row
  await expect(ts.getByText(/^Picked · pressure · band power · "1–2 Hz" · t=608\.00 s/)).toBeVisible();
  const cellValues = await ts.getByRole("link", { name: "Show input values →" }).getAttribute("href");
  expect(new URLSearchParams(cellValues!.split("?")[1]).get("s")).toBe("[0, 0:1, 5:10]");
  await ts.getByRole("link", { name: "Show on input →" }).click();
  await expect(page).toHaveURL(/tensors\?node=input_1&t=40/);
  const model = page.getByRole("group", { name: "Model input" });
  const legend = model.getByRole("list", { name: "Input windows" });
  await expect(legend.getByText("Step 40", { exact: true })).toBeVisible();
  await expect(legend.getByText(/^horizon t\+1…t\+3 · steps 41–43$/)).toBeVisible();
  // the context band sits on the pressure map only; the raw plots show every window's time span and the step
  const bands = (name: string) => model.getByRole("img", { name }).locator("xpath=..").locator("div[aria-hidden]");
  await expect(bands("pressure · band power")).toHaveCount(4);   // target, context, horizon, step rule
  await expect(bands("hr · band power")).toHaveCount(3);
  await expect(page.getByRole("group", { name: "Raw input" }).locator("div[aria-hidden]")).toHaveCount(8);   // 2 channels × (3 + rule)
});

test("input attribution of a signal run is a channel × time map with block shares and a time line", async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/trace?m=gxa`));
  await page.getByRole("button", { name: "Run attribution" }).first().click();
  const ts = page.getByRole("group", { name: "Time-series attribution" });
  const map = ts.getByRole("img", { name: "Input attribution" });
  await expect(map).toBeVisible({ timeout: 30_000 });
  // x = seconds from recording start (steps 608 … 1192 s), y = the 10 channels, block names beside the map
  await expect(ts.getByText("s from recording start").first()).toBeVisible();
  await expect(ts.getByText("608", { exact: true }).first()).toBeVisible();
  await expect(ts.getByText("channel · by block")).toBeVisible();
  await expect(ts.getByText("hr · band power", { exact: true }).first()).toBeVisible();
  const box = (await map.boundingBox())!;
  await page.mouse.move(box.x + 3, box.y + 3);
  await expect(ts.getByText(/^hr · band power · "0\.0625–0\.125 Hz" · t=608\.00 s · rel /)).toBeVisible();
  await page.mouse.move(box.x + box.width - 3, box.y + box.height - 3);
  await expect(ts.getByText(/^pressure · band power · "1–2 Hz" · t=1192\.00 s · rel /)).toBeVisible();
  // block shares add up to 100 %; the time line reads its peak column
  const shares = ts.getByRole("img", { name: "Block shares" });
  await expect(shares).toBeVisible();
  const pcts = (await shares.locator("text").allTextContents()).filter((t) => t.endsWith("%")).map(parseFloat);
  expect(pcts).toHaveLength(2);
  expect(Math.abs(pcts[0] + pcts[1] - 100)).toBeLessThanOrEqual(1);
  await expect(ts.getByRole("img", { name: "Attribution over time" })).toBeVisible();
  await expect(ts.getByText(/^Σ per column · peak t=\d+\.\d+ s · Σ /)).toBeVisible();
  await expect(page.getByRole("button", { name: "Export Block shares as PNG" })).toBeVisible();
  // the map widens the left panel; Narrow puts it back to 400 px
  const aside = page.locator("aside").first();
  expect((await aside.boundingBox())!.width).toBeGreaterThan(600);
  await page.getByRole("button", { name: /^Narrow/ }).click();
  expect((await aside.boundingBox())!.width).toBeCloseTo(400, 0);
});

test("Home offers the model's dataset and captures the picked sample", async ({ page, request }) => {
  await page.goto(url("#/"));
  await page.getByLabel("Model", { exact: true }).fill(SIGNAL);
  const kinds = page.getByRole("group", { name: "Sample kind" });
  await expect(kinds.getByRole("button", { name: "Dataset" })).toHaveAttribute("aria-pressed", "true");
  await page.getByLabel("Filter samples").fill("rec03");
  const pick = page.getByLabel("Dataset sample");
  await expect(pick.locator("option")).toHaveCount(3);
  await pick.selectOption("rec03:2");
  await page.getByRole("button", { name: /Capture forward pass/ }).click();
  await expect(page).toHaveURL(/#\/runs\/run_[^/]+\/graph/);
  const id = /runs\/(run_[^/]+)\//.exec(page.url())![1];
  const meta = await getJson<RunMeta>(request, `/api/runs/${id}`);
  expect(meta.sample.ref).toBe("rec03:2");
});
