#!/usr/bin/env python3
"""Run the manuscript consistency subset of the submission readiness audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import audit_submission_ready as audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    issues = audit.audit_manuscript_consistency()
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    if args.json:
        print(json.dumps({"errors": [i.__dict__ for i in errors], "warnings": [i.__dict__ for i in warnings]}, indent=2))
    else:
        for issue in issues:
            print(f"{issue.severity.upper()}: {issue.where}: {issue.message}")
        print(f"consistency audit complete: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
