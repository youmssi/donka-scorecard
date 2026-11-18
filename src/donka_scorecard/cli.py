from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .builder import build


def main() -> int:
    parser = argparse.ArgumentParser(prog="donka-scorecard")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("build", help="fit and write a Donka format-v1 scorecard pack")
    command.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        result = build(args.config)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"donka-scorecard: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
