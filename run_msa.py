import torch

from models.backbone import load_model
from models.adaptive_model import AdaptiveGPT2

from analyzer.task_analyzer import TaskAnalyzer
from controller.complexity import ComplexityScorer
from controller.selector import ConfigurationSelector


# =========================================================
# Load GPT-2
# =========================================================

print("Loading MSA system...")

model, tokenizer = load_model()

adaptive_model = AdaptiveGPT2(model)

adaptive_model.eval()


# =========================================================
# Initialize MSA components
# =========================================================

analyzer = TaskAnalyzer()

scorer = ComplexityScorer()

selector = ConfigurationSelector()


# =========================================================
# Configuration → depth
# =========================================================

DEPTH_MAP = {
    "shallow": 4,
    "medium": 8,
    "deep": 12
}


# =========================================================
# Run MSA
# =========================================================

def run_msa(text):

    # -----------------------------------------------------
    # 1. Analyze task
    # -----------------------------------------------------

    features = analyzer.analyze(text)

    # -----------------------------------------------------
    # 2. Calculate complexity
    # -----------------------------------------------------

    complexity = scorer.calculate(features)

    # -----------------------------------------------------
    # 3. Select configuration
    # -----------------------------------------------------

    configuration = selector.select(complexity)

    # -----------------------------------------------------
    # 4. Convert configuration → depth
    # -----------------------------------------------------

    depth = DEPTH_MAP[configuration]

    # -----------------------------------------------------
    # 5. Tokenize
    # -----------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    # -----------------------------------------------------
    # 6. Run adaptive GPT-2
    # -----------------------------------------------------

    with torch.no_grad():

        logits = adaptive_model(
            input_ids=inputs["input_ids"],
            depth=depth
        )

    return {
        "features": features,
        "complexity": complexity,
        "configuration": configuration,
        "depth": depth,
        "logits": logits
    }


# =========================================================
# Test inputs
# =========================================================

texts = [

    "What is 2 + 2?",

    "Explain why the sky appears blue.",

    "Calculate the percentage increase from 50 to 75.",

    """
    John has 5 apples. He buys 3 more apples.
    He then gives 2 apples to Sarah.
    Explain how many apples John has remaining.
    """

]


# =========================================================
# Run tests
# =========================================================

for text in texts:

    result = run_msa(text)

    print("\n")
    print("=" * 70)

    print("INPUT:")
    print(text.strip())

    print("\nFEATURES:")

    for name, value in result["features"].items():

        print(
            f"{name:20s}: {value:.3f}"
        )

    print(
        f"\nCOMPLEXITY: "
        f"{result['complexity']:.3f}"
    )

    print(
        f"CONFIGURATION: "
        f"{result['configuration'].upper()}"
    )

    print(
        f"SELECTED DEPTH: "
        f"{result['depth']}"
    )

    print(
        f"LOGITS SHAPE: "
        f"{result['logits'].shape}"
    )