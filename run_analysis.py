#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

PORT = sys.argv[1] if len(sys.argv) > 1 else "8867"

def run(cmd):
    subprocess.run(cmd, check=True)

def main():

    Path("config").mkdir(exist_ok=True)

    print(f"Launching analysis interface on port {PORT}...")

    run([
        "voila",
        "interface.ipynb",
        "--port", PORT
    ])

if __name__ == "__main__":
    main()
