"""`python -m tau_telecom_sft.serving [--port 8085] [--reload]`: the viewer on 127.0.0.1 only."""

from __future__ import annotations

import argparse

import uvicorn

HOST = "127.0.0.1"  # D9: the Mac only, for now
PORT = 8085


def main() -> None:
    ap = argparse.ArgumentParser(description="tau-telecom-sft viewer API (and built UI)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    uvicorn.run(
        "tau_telecom_sft.serving.app:create_app",
        factory=True,
        host=HOST,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
