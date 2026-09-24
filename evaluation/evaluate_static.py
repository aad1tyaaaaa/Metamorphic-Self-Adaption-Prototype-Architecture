"""Baseline A (static 12-layer GPT-2) through the standard runner.

    python -m evaluation.evaluate_static --dataset short_qa [--generate]
"""

from evaluation.runner import cli

if __name__ == "__main__":
    cli(["static"])
