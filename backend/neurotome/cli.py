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
    source.add_argument("--sample", default=None, metavar="REF",
                        help="a dataset sample of a py: module (its load_sample(ref))")  # fmt: skip
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

    from .config import get_settings

    try:
        st = get_settings()
    except ValueError as e:  # a bad NEUROTOME_* value: a plain message, never a traceback
        print(f"neurotome: {e}", file=sys.stderr)
        return 1

    if args.cmd == "serve":
        import uvicorn

        host, port = args.host or st.host, args.port or st.port
        os.environ["NEUROTOME_HOST"] = host  # the app (uvicorn's factory) answers this host name too
        print(f"Neurotome -> http://{host}:{port}   runs: {st.runs_dir}")
        uvicorn.run("neurotome.api:create_app", factory=True, host=host, port=port, reload=args.reload)
        return 0

    if args.cmd == "capture":
        from .runs import RunManager

        mgr = RunManager.from_settings(st)
        sample: dict = {"kind": "random"}
        if args.text is not None:
            sample = {"kind": "text", "text": args.text}
        if args.sample is not None:
            sample = {"kind": "ref", "ref": args.sample}
        if args.image:
            with open(args.image, "rb") as f:
                sample = {
                    "kind": "upload",
                    "upload_id": mgr.save_upload(args.image, f.read()),
                    "name": os.path.basename(args.image),
                }
        modules = args.modules.split(",") if args.modules else None
        try:
            meta = mgr.create_from_spec(args.model, args.weights, sample, modules, args.max_capture_mb,
                                        backend=args.backend)  # fmt: skip
        except (ValueError, RuntimeError) as e:  # a plain message, as the API gives, never a traceback
            from .errors import describe

            print(f"neurotome capture: {describe(e)[1]}", file=sys.stderr)
            return 1
        json.dump({k: meta[k] for k in ("id", "model", "sample", "output")}, sys.stdout, indent=2)
        print()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
