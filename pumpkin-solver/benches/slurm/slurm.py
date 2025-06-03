#!/usr/bin/env python3

import argparse

from run import run
from status import check_status


def main():
    parser = argparse.ArgumentParser(description="CLI for managing Slurm experiments")
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--config", type=str, default="./experiments.toml", help="Path to experiments.toml")
    run_parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=False)
    run_parser.set_defaults(func=run)

    status_parser = subparsers.add_parser("status", help="Check running jobs")
    status_parser.add_argument("--config", type=str, default="./experiments.toml", help="Path to experiments.toml")
    status_parser.set_defaults(func=check_status)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
