import { expect, test } from "@playwright/test";
import type { RunMeta } from "../src/lib/types";
import { createRun, url } from "./util";

// a 32×24 RGB PNG (a colour ramp), so the upload needs no file on disk
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAIAAAAUMWhjAAAAfElEQVR4nLXNywmAMBAA0RE8bP/VWFpuCgERJSb7hTkOvI3jFGl17dKa8FPrE6GmQIaxAogaCoCQoQPwG2oAp2EB8BhGALNhB7AZLgCD4QXQGgEAlREDWBthgIWRATAzkgB+jTyAsZEKMDCyAb5GAcDLqAF4jDLgNiqBblz90S96C+oOqwAAAABJRU5ErkJggg==",
  "base64");

let image: RunMeta;
let text: RunMeta;
test.beforeAll(async ({ playwright }) => {
  const request = await playwright.request.newContext();
  const up = await request.post(url("api/uploads"), {
    multipart: { file: { name: "ramp.png", mimeType: "image/png", buffer: PNG } },
  });
  expect(up.ok(), await up.text()).toBe(true);
  const { upload_id } = await up.json();
  image = await createRun(request, { sample: { kind: "upload", upload_id, name: "ramp.png" } });
  text = await createRun(request, { model: "py:examples.tiny_lm:build", sample: { kind: "text", text: "the cat" } });
  await request.dispose();
});

test("the input node shows the raw image next to the model input, with the preprocessing", async ({ page }) => {
  await page.goto(url(`#/runs/${image.id}/tensors?node=input_1`));
  await expect(page.getByRole("heading", { name: "Input" })).toBeVisible();
  const raw = page.getByRole("group", { name: "Raw input" });
  await expect(raw.getByText(/ramp\.png · PNG · 32×24 px/)).toBeVisible();
  await expect(raw.getByRole("img", { name: "Raw input" })).toBeVisible();
  await expect(raw.getByText("31", { exact: true })).toBeVisible();   // the x range ends at W − 1
  const model = page.getByRole("group", { name: "Model input" });
  await expect(model.getByText(/1×1×28×28 · float32/)).toBeVisible();
  await expect(page.getByRole("group", { name: "Preprocessing" }).getByText("Custom: examples.custom_model.preprocess")).toBeVisible();
  await expect(page.getByRole("table", { name: "Model input values" }).locator("tbody tr")).toHaveCount(1);
  // "Tensor" switches to the input tensor's own channel views
  await page.getByRole("group", { name: "Show" }).getByRole("button", { name: "Tensor" }).click();
  await expect(page.getByRole("heading", { name: "Channels" })).toBeVisible();
});

test("a text run shows its text and the token ids it became", async ({ page }) => {
  await page.goto(url(`#/runs/${text.id}/tensors?node=input_1`));
  await expect(page.getByRole("group", { name: "Raw input" }).getByText("the cat", { exact: true })).toBeVisible();
  await expect(page.getByRole("list", { name: "Model input tokens" }).getByRole("listitem")).toHaveCount(7);
  await expect(page.getByText("Tokenize with CharTokenizer → token ids")).toBeVisible();
});
