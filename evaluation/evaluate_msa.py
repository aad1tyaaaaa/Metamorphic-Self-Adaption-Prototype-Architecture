"""Baseline E (full MSA) through the standard runner.

    python -m evaluation.evaluate_msa --dataset short_qa [--generate]
"""

from evaluation.runner import cli

if __name__ == "__main__":
    cli(["msa"])
