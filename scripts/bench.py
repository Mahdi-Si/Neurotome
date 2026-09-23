"""Benchmarks for the MVP success criteria (docs/PLAN.md §5), in a temporary runs dir.

    python scripts/bench.py [--device cpu]

capture_resnet50_s   create_from_spec("torchvision:resnet50"), random weights, 1x3x224x224, model build incl.
tensor_view_p95_ms   p95 of GET /api/runs/{id}/ops/{op}/tensor over every saved op (TestClient, defaults)
ig32_s               manager.attribution(id, "ig", None, 32)
Exits 1 when any metric misses its budget.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

BUDGETS = {"capture_resnet50_s": 10.0, "tensor_view_p95_ms": 300.0, "ig32_s": 30.0}


def p95(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, -(-95 * len(s) // 100) - 1))]   # nearest rank


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            import subprocess

            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        m = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.M)
        if m:
            return m.group(1).strip()
    except Exception:  # noqa: BLE001 - the name is informational only
        pass
    return platform.processor() or "unknown CPU"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--device", default="cpu", help="budgets are set for cpu")
    a = p.parse_args(argv)

    import torch
    import torchvision  # noqa: F401 - imported before timing, like a running server
    from fastapi.testclient import TestClient

    from neurotome.api import create_app
    from neurotome.runs import RunManager
    from neurotome.store import RunStore

    with tempfile.TemporaryDirectory(prefix="neurotome-bench-", ignore_cleanup_errors=True) as tmp:
        mgr = RunManager(RunStore(Path(tmp)), a.device, 2048 * 1024 * 1024)
        got: dict[str, float] = {}

        t = time.perf_counter()
        meta = mgr.create_from_spec("torchvision:resnet50", None, {"kind": "random"})
        got["capture_resnet50_s"] = time.perf_counter() - t

        run_id = meta["id"]
        ops = [o["id"] for o in mgr.store.ir(run_id)["ops"] if o.get("saved")]
        client = TestClient(create_app(mgr))
        times = []
        for op in ops:
            t = time.perf_counter()
            r = client.get(f"/api/runs/{run_id}/ops/{quote(op, safe='')}/tensor")
            times.append((time.perf_counter() - t) * 1000)
            if r.status_code != 200:
                print(f"GET tensor {op}: {r.status_code} {r.text[:200]}", file=sys.stderr)
                return 1
        got["tensor_view_p95_ms"] = p95(times)

        t = time.perf_counter()
        mgr.attribution(run_id, "ig", None, 32)
        got["ig32_s"] = time.perf_counter() - t

    print(f"{'metric':<20} {'value':>9} {'budget':>9}  pass")
    for k, budget in BUDGETS.items():
        print(f"{k:<20} {got[k]:>9.2f} {budget:>9.0f}  {'yes' if got[k] < budget else 'NO'}")
    print(f"{len(ops)} saved ops, device {a.device}, {cpu_name()}, {torch.get_num_threads()} threads "
          f"({os.cpu_count()} logical CPUs), torch {torch.__version__}, Python {platform.python_version()}, "
          f"{platform.system()} {platform.release()}")
    return 0 if all(got[k] < b for k, b in BUDGETS.items()) else 1


if __name__ == "__main__":
    assert p95([float(i) for i in range(1, 101)]) == 95.0 and p95([3.0]) == 3.0
    raise SystemExit(main())
