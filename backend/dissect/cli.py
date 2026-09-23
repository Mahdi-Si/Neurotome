"""Command line: ``dissect serve`` and ``dissect capture``."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dissect", description="Open a PyTorch model and see every signal.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the local server (API + UI)")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--runs-dir", default=None)
    s.add_argument("--device", default=None, help="cpu, cuda, cuda:1 ...")
    s.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")

    c = sub.add_parser("capture", help="capture a run without starting the server")
    c.add_argument("model", help="torchvision:resnet18 or py:package.module:factory")
    c.add_argument("--weights", default=None, help="DEFAULT or a torchvision weights enum name")
    c.add_argument("--image", default=None, help="path to an image to use as the sample")
    c.add_argument("--runs-dir", default=None)

    args = p.parse_args(argv)
    if getattr(args, "runs_dir", None):
        os.environ["DISSECT_RUNS_DIR"] = args.runs_dir
    if getattr(args, "device", None):
        os.environ["DISSECT_DEVICE"] = args.device

    if args.cmd == "serve":
        import uvicorn

        from .config import get_settings

        st = get_settings()
        host, port = args.host or st.host, args.port or st.port
        print(f"Dissect -> http://{host}:{port}   runs: {st.runs_dir}")
        uvicorn.run("dissect.api:create_app", factory=True, host=host, port=port, reload=args.reload)
        return 0

    if args.cmd == "capture":
        from .config import get_settings
        from .runs import RunManager
        from .store import RunStore

        st = get_settings()
        mgr = RunManager(RunStore(st.runs_dir), st.device, st.max_capture_bytes)
        sample = {"kind": "random"}
        if args.image:
            with open(args.image, "rb") as f:
                sample = {
                    "kind": "upload",
                    "upload_id": mgr.save_upload(args.image, f.read()),
                    "name": os.path.basename(args.image),
                }
        meta = mgr.create_from_spec(args.model, args.weights, sample)
        json.dump({k: meta[k] for k in ("id", "model", "sample", "output")}, sys.stdout, indent=2)
        print()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
