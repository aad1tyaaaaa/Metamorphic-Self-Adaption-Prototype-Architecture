"""Baseline B (depth-only adaptation) through the standard runner.

    python -m evaluation.evaluate_depth_adaptive --dataset short_qa [--generate]
"""

from evaluation.runner import cli

if __name__ == "__main__":
    cli(["depth_adaptive"])
