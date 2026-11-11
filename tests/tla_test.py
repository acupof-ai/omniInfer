#!/usr/bin/env python3
"""Run the finite request-lifecycle model with the official TLC tools."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "tla" / "RequestLifecycle.tla"
CONFIG = ROOT / "tla" / "RequestLifecycle.cfg"


def main():
    if "TLA2TOOLS_JAR" not in os.environ:
        print("SKIP: TLA2TOOLS_JAR is not set; official tla2tools.jar is required.")
        return 0

    jar = Path(os.environ["TLA2TOOLS_JAR"]).expanduser().resolve()
    if not jar.is_file():
        print(f"FAIL: TLA2TOOLS_JAR is not a file: {jar}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="umminfer-tla-") as workdir:
        command = [
            "java", "-cp", str(jar), "tlc2.TLC", "-workers", "1",
            "-metadir", workdir, "-teSpecOutDir", workdir,
            "-config", str(CONFIG), str(SPEC),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print("FAIL: TLC exceeded the 120-second timeout.", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"FAIL: could not run TLC: {exc}", file=sys.stderr)
            return 1

    print(result.stdout, end="")
    completed = re.search(
        r"^Model checking completed\. No error has been found\.$",
        result.stdout,
        re.MULTILINE,
    )
    if result.returncode != 0 or completed is None:
        print("FAIL: TLC did not confirm a successful model check.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
