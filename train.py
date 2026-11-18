"""Backward-compatible entry point. Prefer: ``uv run helmet-safety train [OPTIONS]``."""

import sys

from helmet_safety.interfaces.cli.main import run

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train", *sys.argv[1:]]
    run()
