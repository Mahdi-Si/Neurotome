import { expect, test } from "@playwright/test";
import type { RunMeta } from "../src/lib/types";
import { createRun, url } from "./util";

let run: RunMeta;
test.beforeAll(async ({ playwright }) => {
  const request = await playwright.request.newContext();
  run = await createRun(request);
  await request.dispose();
});

const graph = (q: string) => url(`#/runs/${run.id}/graph?${q}`);

test("expand and collapse one module at depth 1", async ({ page }) => {
  await page.goto(graph("depth=1"));
  await page.getByRole("button", { name: "Expand blocks.0", exact: true }).click();
  await expect(page).toHaveURL(/expand=/);
  await page.getByRole("button", { name: "Collapse blocks.0", exact: true }).click();
  await expect(page).not.toHaveURL(/expand=/);
});

test("search with / and keyboard navigation on the canvas", async ({ page }) => {
  await page.goto(graph("depth=1"));
  await expect(page.getByRole("button", { name: "Expand blocks.0", exact: true })).toBeVisible();   // laid out
  await page.keyboard.press("/");
  const search = page.getByLabel("Search graph");
  await expect(search).toBeFocused();
  await search.fill("conv");
  await expect(page.getByRole("option").first()).toBeVisible();
  await search.press("Enter");
  await expect(page).toHaveURL(/node=/);

  const before = page.url();
  await page.getByRole("application", { name: /Computation graph/ }).focus();
  await page.keyboard.press("ArrowDown");
  await expect.poll(() => page.url()).not.toBe(before);
  await expect(page).toHaveURL(/node=/);
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/tensors/);
});

test("Norm colour mode shows its legend", async ({ page }) => {
  await page.goto(graph("depth=1"));
  const norm = page.getByRole("group", { name: "Node colour" }).getByRole("button", { name: "Norm", exact: true });
  await norm.click();
  await expect(norm).toHaveAttribute("aria-pressed", "true");
  await expect(page).toHaveURL(/color=norm/);
  await expect(page.getByText(/^RMS = /)).toBeVisible();
});

test("the right panel widens by dragging its edge, keeps the width, and resets on double-click", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await page.goto(graph("depth=1"));
  const edge = page.getByRole("separator", { name: "Resize panel" });
  await expect(edge).toHaveAttribute("aria-valuenow", "400");
  const box = (await edge.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + 200);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 150, box.y + 200, { steps: 5 });
  await page.mouse.up();
  await expect(edge).toHaveAttribute("aria-valuenow", "550");
  await page.reload();
  await expect(edge).toHaveAttribute("aria-valuenow", "550");
  await edge.press("ArrowRight");
  await expect(edge).toHaveAttribute("aria-valuenow", "526");
  await edge.dblclick();
  await expect(edge).toHaveAttribute("aria-valuenow", "400");
});
