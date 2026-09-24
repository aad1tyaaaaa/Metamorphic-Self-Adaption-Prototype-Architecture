"""MSA-GPT-2 prototype command line (Phase 19).

    python main.py --mode simulate
    python main.py --mode calibrate
    python main.py --mode train-predictor
    python main.py --mode evaluate
    python main.py --mode benchmark
    python main.py --mode plot
    python main.py --mode tables
    python main.py --mode ui
    python main.py --mode validate
    python main.py --mode all

Extra arguments after `--` are passed through to the underlying module, e.g.

    python main.py --mode evaluate -- --suite stability
"""

import argparse
import runpy
import sys

MODES = {
    "simulate": ("run_msa", []),
    "calibrate": ("calibration.run_calibration", []),
    "train-predictor": ("controller.train_predictor", []),
    "ablate-features": ("controller.train_predictor", ["--ablation"]),
    "frontier": ("controller.policy_sweep", []),
    "evaluate": ("evaluation.evaluate", []),
    "benchmark": ("evaluation.benchmark", []),
    "plot": ("evaluation.generate_plots", []),
    "tables": ("evaluation.tables", []),
    "validate": ("test_msa", []),
    "ui": ("ui.export_ui_data", []),
}

PIPELINE = ["calibrate", "frontier", "train-predictor", "ablate-features",
            "benchmark", "evaluate", "plot", "tables", "ui", "validate"]


def run(mode, extra):
    module, defaults = MODES[mode]
    print(f"\n{'=' * 74}\n>>> {mode}  ({module})\n{'=' * 74}", flush=True)

    sys.argv = [module] + defaults + extra
    runpy.run_module(module, run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description="MSA-GPT-2 prototype")
    parser.add_argument("--mode", default="simulate",
                        choices=sorted(MODES) + ["all"])
    args, extra = parser.parse_known_args()

    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.mode == "all":
        for mode in PIPELINE:
            run(mode, [])
        print("\nFull pipeline complete.")
    else:
        run(args.mode, extra)


if __name__ == "__main__":
    main()
