"""Compare two phone recordings of a room and report what changed.

    python run.py --a videos/before.mp4 --b videos/after.mp4 --out output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, type=Path, help="first recording, the before")
    parser.add_argument("--b", required=True, type=Path, help="second recording, the after")
    parser.add_argument("--out", default=Path("output"), type=Path, help="where results go")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--no-semantics", action="store_true",
                        help="skip CLIP labelling, geometry only, much faster")
    args = parser.parse_args()

    for video in (args.a, args.b):
        if not video.exists():
            print(f"error: {video} not found", file=sys.stderr)
            return 1

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(args.a, args.b, args.out, cfg, skip_semantics=args.no_semantics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
