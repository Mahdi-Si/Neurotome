// Starts `neurotome serve` (CPU, temp runs dir, free port) for the e2e suite; the returned function stops it.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../..", import.meta.url));

const freePort = () => new Promise<number>((resolve, reject) => {
  const s = createServer().once("error", reject);
  s.listen(0, "127.0.0.1", () => {
    const port = (s.address() as { port: number }).port;
    s.close(() => resolve(port));
  });
});

const python = () => process.env.NEUROTOME_PYTHON
  ?? [join(root, ".venv/Scripts/python.exe"), join(root, ".venv/bin/python")].find(existsSync) ?? "python";

export default async function globalSetup() {
  if (!existsSync(join(root, "backend/neurotome/static/index.html")))
    throw new Error("backend/neurotome/static/index.html is missing: run `npm run build` first (npm run e2e does)");
  const port = await freePort();
  const runs = mkdtempSync(join(tmpdir(), "neurotome-e2e-"));
  const base = `http://127.0.0.1:${port}`;
  const win = process.platform === "win32";
  const proc = spawn(python(), ["-m", "neurotome.cli", "serve", "--host", "127.0.0.1", "--port", String(port), "--runs-dir", runs, "--device", "cpu"],
    { cwd: root, detached: !win, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, PYTHONUNBUFFERED: "1" } });
  let log = "";
  const keep = (b: Buffer) => { log = (log + b.toString()).slice(-4000); };
  proc.stdout!.on("data", keep);
  proc.stderr!.on("data", keep);

  const stop = () => {
    if (proc.pid && proc.exitCode === null) {
      if (win) spawnSync("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { stdio: "ignore" });
      else try { process.kill(-proc.pid, "SIGKILL"); } catch { /* already gone */ }
    }
    rmSync(runs, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
  };

  const deadline = Date.now() + 60_000;
  for (;;) {
    if (proc.exitCode !== null) { stop(); throw new Error(`neurotome serve exited with ${proc.exitCode}:\n${log}`); }
    try {
      const r = await fetch(`${base}/api/health`);
      if (r.ok && (await r.json()).ok) break;
    } catch { /* not up yet */ }
    if (Date.now() > deadline) { stop(); throw new Error(`neurotome serve did not answer /api/health within 60 s:\n${log}`); }
    await new Promise((r) => setTimeout(r, 250));
  }
  process.env.NEUROTOME_E2E_URL = base;
  return stop;
}
