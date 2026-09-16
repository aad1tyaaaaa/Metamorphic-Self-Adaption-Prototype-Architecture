import torch

from models.backbone import load_model
from models.adaptive_model import AdaptiveGPT2


# Load original GPT-2
model, tokenizer = load_model()

# Create adaptive version
adaptive_model = AdaptiveGPT2(model)

adaptive_model.eval()


prompt = "The capital of India is"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


# Test different depths
for depth in [4, 8, 12]:

    print("\n" + "=" * 60)
    print(f"TESTING DEPTH: {depth}")
    print("=" * 60)

    with torch.no_grad():

        logits = adaptive_model(
            input_ids=inputs["input_ids"],
            depth=depth
        )

    print("Logits shape:", logits.shape)