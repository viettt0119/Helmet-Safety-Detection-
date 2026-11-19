"""Backward-compatible entry point. Prefer: ``uv run helmet-safety predict SOURCE [OPTIONS]``.

Without arguments it opens the webcam and shows results in a window, like the original script.
"""

import sys

from helmet_safety.interfaces.cli.main import run

if __name__ == "__main__":
    args = sys.argv[1:] or ["0", "--show"]
    sys.argv = [sys.argv[0], "predict", *args]
    run()
