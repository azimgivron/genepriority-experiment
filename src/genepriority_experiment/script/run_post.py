import argparse
from typing import Any

from genepriority_experiment.script.parser import parse_post
from genepriority_experiment.script.post import post


def main():
    parser = argparse.ArgumentParser(
        description=("Gene Prioritization Post Processing"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parse_post(parser)
    args: Any = parser.parse_args()
    post(args)
