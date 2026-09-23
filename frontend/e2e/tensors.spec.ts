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

test.beforeEach(async ({ page }) => {
  await page.goto(url(`#/runs/${run.id}/tensors?node=${encodeURIComponent(conv)}`));
  await expect(page.getByRole("heading", { name: "Channels" })).toBeVisible();
});

test("channel field and dot matrix set the URL state", async ({ page }) => {
  await page.getByRole("spinbutton", { name: /^Channel \(/ }).fill("3");
  await expect(page).toHaveURL(/[?&]c=3/);
  await page.getByRole("img", { name: /dot matrix$/ }).click({ position: { x: 4, y: 4 } });
  await expect(page).toHaveURL(/[?&]py=\d+/);
  await expect(page).toHaveURL(/[?&]px=\d+/);
});

test("a slice expression shows a 4×4 table", async ({ page }) => {
  const expr = page.getByLabel("Slice expression");
  await expr.fill("[0, 0, :4, :4]");
  await expr.press("Enter");
  const rows = page.locator("table tbody tr");
  await expect(rows).toHaveCount(4);
  await expect(rows.first().locator("td")).toHaveCount(4);
});

test("compare mode opens and exits", async ({ page }) => {
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await expect(page.getByText("Δ = B − A")).toBeVisible();
  await expect(page).toHaveURL(/[?&]cmp=/);
  await page.getByRole("button", { name: "Exit compare" }).click();
  await expect(page.getByRole("button", { name: "Compare", exact: true })).toBeVisible();
  await expect(page).not.toHaveURL(/[?&]cmp=/);
});
