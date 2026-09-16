import time
import torch
import math

from models.backbone import load_model
from models.adaptive_model import AdaptiveGPT2


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model, tokenizer = load_model()

adaptive_model = AdaptiveGPT2(model)

adaptive_model.eval()


# ---------------------------------------------------------
# Test sentences
# ---------------------------------------------------------

texts = [
    "The capital of India is New Delhi.",
    "Machine learning is a branch of artificial intelligence.",
    "Python is widely used for software development and data science.",
    "The Earth revolves around the Sun.",
    "A neural network consists of interconnected layers of neurons.",
]


# ---------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------

def benchmark_depth(depth):

    total_loss = 0.0
    total_time = 0.0
    count = 0

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"]

        # Need at least 2 tokens for next-token prediction
        if input_ids.shape[1] < 2:
            continue

        start = time.perf_counter()

        with torch.no_grad():

            logits = adaptive_model(
                input_ids=input_ids,
                depth=depth
            )

        end = time.perf_counter()

        elapsed = end - start

        # Next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_function = torch.nn.CrossEntropyLoss()

        loss = loss_function(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        total_loss += loss.item()
        total_time += elapsed

        count += 1

    average_loss = total_loss / count
    average_time = total_time / count

    perplexity = math.exp(average_loss)

    return average_loss, perplexity, average_time


# ---------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("MSA DEPTH BENCHMARK")
print("=" * 70)


for depth in [4, 8, 12]:

    loss, perplexity, latency = benchmark_depth(depth)

    print("\n")
    print(f"DEPTH:       {depth}")
    print(f"LOSS:        {loss:.4f}")
    print(f"PERPLEXITY:  {perplexity:.4f}")
    print(f"LATENCY:     {latency:.4f} seconds")


print("\n")
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)