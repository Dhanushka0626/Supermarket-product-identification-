"""Shared command-line handling for all scripts (adds src/ to the import path)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smartcheckout.config import setup  # noqa: E402


def parse_args(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"), help="YAML settings file")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override settings, e.g. epochs=10")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg, paths = setup(args.config, args.set)
    return args, cfg, paths


def device_of(cfg):
    import torch
    return "cpu" if cfg["device"] == "cpu" or not torch.cuda.is_available() else "cuda"
