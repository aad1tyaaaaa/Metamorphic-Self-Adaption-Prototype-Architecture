import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import time
import math
import torch
import pandas as pd

from models.backbone import load_model
from models.adaptive_model import AdaptiveGPT2
from analyzer.task_analyzer import TaskAnalyzer
from controller.complexity import ComplexityScorer


# --------------------------------------------------
# 1. Load model
# --------------------------------------------------

print("Loading calibration system...")

model, tokenizer = load_model()

adaptive_model = AdaptiveGPT2(model)
adaptive_model.eval()

analyzer = TaskAnalyzer()
scorer = ComplexityScorer()


# --------------------------------------------------
# 2. Calibration dataset
# --------------------------------------------------

# texts = [
#     "What is 2 + 2?",

#     "What is the capital of France?",

#     "Explain why the sky appears blue.",

#     "Calculate the percentage increase from 50 to 75.",

#     "What is the boiling point of water?",

#     "Explain the difference between RAM and ROM.",

#     "Why does the Earth have seasons?",

#     """
#     John has 5 apples. He buys 3 more apples.
#     He then gives 2 apples to Sarah.
#     How many apples does John have remaining?
#     """,

#     """
#     A train travels 120 kilometers in 2 hours.
#     What is its average speed?
#     """,

#     """
#     Explain how photosynthesis converts sunlight into chemical energy.
#     """,

#     """
#     If a shirt costs 800 rupees and is discounted by 25 percent,
#     what is the final price?
#     """,

#     """
#     Explain why increasing the temperature generally increases
#     the rate of a chemical reaction.
#     """,

#     """
#     A store has 240 items. It sells 35 percent of them.
#     How many items remain?
#     """
# ]

from data.calibration_dataset import CALIBRATION_DATASET

texts = CALIBRATION_DATASET

# --------------------------------------------------
# 3. Depth configurations
# --------------------------------------------------

DEPTH_MAP = {
    4: "shallow",
    8: "medium",
    12: "deep"
}


# --------------------------------------------------
# 4. Evaluate one input at one depth
# --------------------------------------------------

def evaluate(text, depth):

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]

    start_time = time.perf_counter()

    with torch.no_grad():

        logits = adaptive_model(
            input_ids=input_ids,
            depth=depth
        )

    end_time = time.perf_counter()

    latency = end_time - start_time

    # ----------------------------------------------
    # Next-token language modeling loss
    # ----------------------------------------------

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    loss_value = loss.item()

    perplexity = math.exp(
        min(loss_value, 20)
    )

    return loss_value, perplexity, latency


# --------------------------------------------------
# 5. Run calibration
# --------------------------------------------------

results = []

total_runs = len(texts) * len(DEPTH_MAP)

run_number = 0


for sample_id, text in enumerate(texts):

    text = text.strip()

    # Analyze task once
    features = analyzer.analyze(text)

    complexity = scorer.calculate(features)

    print("\n")
    print("=" * 70)
    print(f"SAMPLE {sample_id + 1}")
    print("=" * 70)

    print(f"Text: {text}")
    print(f"Complexity: {complexity:.3f}")

    for depth in [4, 8, 12]:

        run_number += 1

        print(
            f"\n[{run_number}/{total_runs}] "
            f"Testing depth {depth}..."
        )

        loss, perplexity, latency = evaluate(
            text,
            depth
        )

        configuration = DEPTH_MAP[depth]

        result = {
            "sample_id": sample_id,
            "text": text,

            "input_length": features["input_length"],
            "reasoning": features["reasoning"],
            "domain": features["domain"],
            "structure": features["structure"],

            "complexity": complexity,

            "depth": depth,
            "configuration": configuration,

            "loss": loss,
            "perplexity": perplexity,
            "latency_seconds": latency
        }

        results.append(result)

        print(
            f"Depth: {depth} | "
            f"Config: {configuration} | "
            f"Loss: {loss:.4f} | "
            f"PPL: {perplexity:.2f} | "
            f"Latency: {latency:.4f}s"
        )


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

df = pd.DataFrame(results)

output_path = "results/calibration_results.csv"

df.to_csv(
    output_path,
    index=False
)

print("\n")
print("=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)

print(f"Total experiments: {len(df)}")
print(f"Saved to: {output_path}")

print("\nAverage results by depth:")

summary = (
    df.groupby("depth")
    [["loss", "perplexity", "latency_seconds"]]
    .mean()
)

print(summary)