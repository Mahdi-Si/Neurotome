"""Command line: ``neurotome serve`` and ``neurotome capture``."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="neurotome", description="Open a PyTorch model and see every signal.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the local server (API + UI)")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--runs-dir", default=None)
    s.add_argument("--device", default=None, help="cpu, cuda, cuda:1 ...")
    s.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")

    c = sub.add_parser("capture", help="capture a run without starting the server")
    c.add_argument("model", help="torchvision:resnet18, hf:gpt2 or py:package.module:factory")
    c.add_argument("--weights", default=None, help="DEFAULT (pretrained) or a torchvision weights enum name")
    source = c.add_mutually_exclusive_group()
    source.add_argument("--image", default=None, help="path to an image to use as the sample")
    source.add_argument("--text", default=None, help="text sample for a model with a tokenizer")
    c.add_argument("--backend", choices=["torchlens", "nnsight"], default=None,
                   help="default: auto (NNsight above NEUROTOME_NNSIGHT_PARAMS)")  # fmt: skip
    c.add_argument("--runs-dir", default=None)
    c.add_argument("--modules", default=None, help="save tensors only inside these modules: layer3,layer4")
    c.add_argument("--max-capture-mb", type=float, default=None, help="capture budget for this run (MB)")

    args = p.parse_args(argv)
    if getattr(args, "runs_dir", None):
        os.environ["NEUROTOME_RUNS_DIR"] = args.runs_dir
    if getattr(args, "device", None):
        os.environ["NEUROTOME_DEVICE"] = args.device

    if args.cmd == "serve":
        import uvicorn

        from .config import get_settings

        st = get_settings()
        host, port = args.host or st.host, args.port or st.port
        print(f"Neurotome -> http://{host}:{port}   runs: {st.runs_dir}")
        uvicorn.run("neurotome.api:create_app", factory=True, host=host, port=port, reload=args.reload)
        return 0

    if args.cmd == "capture":
        from .config import get_settings
        from .runs import RunManager

        st = get_settings()
        mgr = RunManager.from_settings(st)
        sample: dict = {"kind": "random"}
        if args.text is not None:
            sample = {"kind": "text", "text": args.text}
        if args.image:
            with open(args.image, "rb") as f:
                sample = {
                    "kind": "upload",
                    "upload_id": mgr.save_upload(args.image, f.read()),
                    "name": os.path.basename(args.image),
                }
        modules = args.modules.split(",") if args.modules else None
        meta = mgr.create_from_spec(args.model, args.weights, sample, modules, args.max_capture_mb,
                                    backend=args.backend)  # fmt: skip
        json.dump({k: meta[k] for k in ("id", "model", "sample", "output")}, sys.stdout, indent=2)
        print()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
