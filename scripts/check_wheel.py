"""Check a built neurotome wheel and sdist: UI and license inside; optionally a clean install serves the UI.

    python scripts/check_wheel.py backend/dist/*.whl [--sdist backend/dist/*.tar.gz] [--install]
    python scripts/check_wheel.py --self-test

--install makes a temporary venv with uv (CPU torch first, then the wheel), runs `neurotome --help`, starts
`python -m neurotome.cli serve` from the temp dir and checks /api/health, / and one asset.
Exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import configparser
import glob
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

TORCH_CPU = "https://download.pytorch.org/whl/cpu"
WHEEL_NEEDS = {
    "neurotome/static/index.html": r"neurotome/static/index\.html",
    "neurotome/static/assets/*.js": r"neurotome/static/assets/[^/]+\.js",
    "bundled .woff2 fonts": r"neurotome/static/assets/[^/]+\.woff2",
    "dist-info licenses/LICENSE": r"[^/]+\.dist-info/licenses/LICENSE",
}


def wheel_problems(names: list[str], entry_points: str) -> list[str]:
    bad = [what for what, pat in WHEEL_NEEDS.items() if not any(re.fullmatch(pat, n) for n in names)]
    ep = configparser.ConfigParser()
    ep.read_string(entry_points)
    if ep.get("console_scripts", "neurotome", fallback=None) != "neurotome.cli:main":
        bad.append("neurotome console script in entry_points.txt")
    return bad


def check_wheel(path: str) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        ep = next((n for n in names if re.fullmatch(r"[^/]+\.dist-info/entry_points\.txt", n)), None)
        return wheel_problems(names, z.read(ep).decode() if ep else "")


def check_sdist(path: str) -> list[str]:
    with tarfile.open(path) as t:
        ok = any(n.endswith("/neurotome/static/index.html") for n in t.getnames())
    return [] if ok else ["neurotome/static/index.html in the sdist"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode("utf-8", "replace")


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/pid", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(10)


def check_install(wheel: str) -> list[str]:
    tmp = Path(tempfile.mkdtemp(prefix="neurotome-wheel-"))
    venv, log = tmp / "venv", tmp / "serve.log"
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    py, exe = bindir / ("python.exe" if os.name == "nt" else "python"), bindir / "neurotome"
    proc = None
    step = "uv venv"
    try:
        run = lambda *a: subprocess.run([str(x) for x in a], check=True, capture_output=True, text=True)  # noqa: E731
        run("uv", "venv", venv, "--python", sys.executable)
        step = "CPU torch"
        run("uv", "pip", "install", "--python", py, "torch", "torchvision", "--index-url", TORCH_CPU)
        step = "install wheel"
        run("uv", "pip", "install", "--python", py, Path(wheel).resolve())
        step = "neurotome --help"
        run(exe, "--help")
        step = "serve"
        port = _free_port()
        with open(log, "w") as out:
            proc = subprocess.Popen(
                [str(py), "-m", "neurotome.cli", "serve", "--port", str(port),
                 "--runs-dir", str(tmp / "runs"), "--device", "cpu"],
                cwd=tmp, stdout=out, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        base, deadline = f"http://127.0.0.1:{port}", time.monotonic() + 60
        step = "/api/health within 60 s"
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            try:
                if '"ok":true' in _get(base + "/api/health")[1].replace(" ", ""):
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("no answer")
            time.sleep(0.5)
        step = "GET / serves the UI"
        status, html = _get(base + "/")
        if status != 200 or '<div id="root">' not in html:
            raise RuntimeError(f"status {status}, or no root div in the page")
        step = "GET one asset"
        asset = re.search(r'(?:src|href)="/?(assets/[^"]+)"', html)
        if not asset:
            raise RuntimeError("index.html references no assets/ file")
        if _get(f"{base}/{asset.group(1)}")[0] != 200:
            raise RuntimeError(asset.group(1))
        return []
    except Exception as e:  # report the failing step with the tool's own output
        detail = getattr(e, "stderr", None) or ""
        if proc is not None and log.exists():
            detail += log.read_text(errors="replace")[-2000:]
        return [f"install: {step} failed: {type(e).__name__}: {e}\n{detail}".rstrip()]
    finally:
        if proc is not None:
            _kill_tree(proc)
        shutil.rmtree(tmp, ignore_errors=True)


def self_test() -> None:
    good = ["neurotome/static/index.html", "neurotome/static/assets/index-a1.js",
            "neurotome/static/assets/g.woff2", "neurotome-0.1.0.dist-info/licenses/LICENSE"]
    ep = "[console_scripts]\nneurotome = neurotome.cli:main\n"
    assert wheel_problems(good, ep) == []
    assert wheel_problems(good[1:], ep) == ["neurotome/static/index.html"]
    assert wheel_problems(good, "[gui_scripts]\nneurotome = neurotome.cli:main\n") == [
        "neurotome console script in entry_points.txt"]
    assert wheel_problems([], "") == [*WHEEL_NEEDS, "neurotome console script in entry_points.txt"]
    print("self-test ok")


def _one(pattern: str) -> str:
    hits = sorted(glob.glob(pattern)) or [pattern]   # PowerShell does not expand globs
    if len(hits) > 1:
        sys.exit(f"{pattern} matches {len(hits)} files; remove old builds from backend/dist")
    return hits[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("wheel", nargs="?")
    p.add_argument("--sdist")
    p.add_argument("--install", action="store_true", help="install into a fresh venv and serve the UI")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    if not a.wheel:
        p.error("the wheel path is required")
    wheel = _one(a.wheel)
    results = [(f"wheel {Path(wheel).name}", check_wheel(wheel))]
    if a.sdist:
        sdist = _one(a.sdist)
        results.append((f"sdist {Path(sdist).name}", check_sdist(sdist)))
    if a.install:
        results.append(("install + serve (clean venv, CPU torch)", check_install(wheel)))
    for what, bad in results:
        print(f"{'ok  ' if not bad else 'FAIL'}  {what}")
        for b in bad:
            print(f"      missing/failed: {b}")
    return 1 if any(bad for _, bad in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
