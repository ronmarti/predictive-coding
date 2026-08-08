"""Fallback entry point for local debugging outside Docker.

In production the container runs:
    solara run application/app.py --host 0.0.0.0 --port 8765
"""
import subprocess
import sys


def main() -> None:
    """Launch solara dev server via subprocess."""
    subprocess.run(
        [
            sys.executable, "-m", "solara", "run",
            "application/app.py",
            "--host", "0.0.0.0",
            "--port", "8765",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
