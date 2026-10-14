"""Convenience entry point: python main.py --help"""
import argparse
from src import train, evaluate, predict


def main():
    parser = argparse.ArgumentParser(description="Educational chest X-ray image classification demo")
    parser.add_argument("command", choices=["train", "evaluate", "predict"])
    args, rest = parser.parse_known_args()
    import sys
    sys.argv = [sys.argv[0], *rest]
    {"train": train.main, "evaluate": evaluate.main, "predict": predict.main}[args.command]()


if __name__ == "__main__":
    main()
