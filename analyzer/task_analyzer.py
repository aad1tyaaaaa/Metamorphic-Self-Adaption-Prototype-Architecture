import re

from configs.configurations import FEATURE_SETS

REASONING_WORDS = [
    "why", "because", "therefore", "explain", "prove",
    "derive", "calculate", "reason", "compare", "evaluate",
]

MATH_WORDS = [
    "calculate", "equation", "number", "percentage", "multiply",
    "divide", "algebra", "geometry", "probability",
]

STEP_WORDS = [
    "then", "next", "after", "finally", "first", "second",
    "step", "and then", "once", "before",
]

# Question stems ordered by how much downstream work they usually imply.
QUESTION_TYPES = [
    (0.25, ["what is", "what are", "who", "when", "where", "which"]),
    (0.50, ["how many", "how much", "list", "name"]),
    (0.75, ["how do", "how does", "how would", "describe", "summarize"]),
    (1.00, ["why", "explain", "prove", "derive", "compare", "evaluate"]),
]

MATH_EXPRESSION = re.compile(r"[-+*/^=<>]|\d+\s*(?:percent|%)|\b\d+\s*[-+*/x]\s*\d+")


class TaskAnalyzer:
    """Heuristic feature extraction.

    Version 1 keeps the original four features unchanged so it remains a valid
    baseline; version 2 adds four features aimed at the signals the original set
    misses (numeric content, explicit arithmetic, question intent, step count).
    """

    def __init__(self, version="v1"):
        if version not in FEATURE_SETS:
            raise ValueError(f"version must be one of {list(FEATURE_SETS)}, got {version!r}")
        self.version = version
        self.features = FEATURE_SETS[version]

    def analyze(self, text):
        values = self.analyze_all(text)
        return {name: values[name] for name in self.features}

    def analyze_all(self, text):
        """Every feature both versions can produce, for calibration logging."""
        return {
            "input_length": self.input_length(text),
            "reasoning": self.reasoning_complexity(text),
            "domain": self.domain_score(text),
            "structure": self.structural_complexity(text),
            "numeric_density": self.numeric_density(text),
            "math_expression": self.math_expression(text),
            "question_type": self.question_type(text),
            "reasoning_steps": self.reasoning_steps(text),
        }

    def vector(self, text):
        values = self.analyze(text)
        return [values[name] for name in self.features]

    # ------------------------------------------------------------------
    # Version 1 features
    # ------------------------------------------------------------------

    def input_length(self, text):
        return min(len(text.split()) / 100, 1.0)

    def reasoning_complexity(self, text):
        lowered = text.lower()
        count = sum(lowered.count(word) for word in REASONING_WORDS)
        return min(count / 5, 1.0)

    def domain_score(self, text):
        lowered = text.lower()
        count = sum(lowered.count(word) for word in MATH_WORDS)
        return min(count / 5, 1.0)

    def structural_complexity(self, text):
        sentences = len(re.findall(r"[.!?]", text))
        commas = text.count(",")
        parentheses = text.count("(") + text.count(")")
        return min((sentences + commas + parentheses) / 10, 1.0)

    # ------------------------------------------------------------------
    # Version 2 additions
    # ------------------------------------------------------------------

    def numeric_density(self, text):
        """Share of whitespace tokens that contain a digit."""
        words = text.split()
        if not words:
            return 0.0
        numeric = sum(1 for word in words if any(ch.isdigit() for ch in word))
        return min(numeric / len(words) * 4, 1.0)

    def math_expression(self, text):
        """Density of explicit arithmetic/relational operators."""
        return min(len(MATH_EXPRESSION.findall(text)) / 3, 1.0)

    def question_type(self, text):
        lowered = text.lower()
        score = 0.0
        for value, stems in QUESTION_TYPES:
            if any(stem in lowered for stem in stems):
                score = max(score, value)
        return score

    def reasoning_steps(self, text):
        """Rough count of sequential operations implied by the prompt."""
        lowered = text.lower()
        markers = sum(lowered.count(word) for word in STEP_WORDS)
        clauses = max(len(re.split(r"[.!?;]", text.strip())) - 1, 0)
        return min((markers + clauses) / 6, 1.0)


def demo():
    analyzer = TaskAnalyzer("v2")

    simple = analyzer.analyze("What is the capital of France?")
    hard = analyzer.analyze(
        "John has 5 apples. He buys 3 more, then gives 2 to Sarah. "
        "Explain why he has 6 apples left and calculate 15 percent of that."
    )

    assert len(TaskAnalyzer("v1").analyze("hi")) == 4
    assert len(simple) == 8
    assert all(0.0 <= value <= 1.0 for value in simple.values())
    assert all(0.0 <= value <= 1.0 for value in hard.values())
    assert hard["reasoning_steps"] > simple["reasoning_steps"]
    assert hard["numeric_density"] > simple["numeric_density"]
    assert hard["question_type"] > simple["question_type"]
    assert analyzer.analyze("") == dict.fromkeys(analyzer.features, 0.0)

    print("task_analyzer demo OK")
    for name, value in hard.items():
        print(f"  {name:18s}: {value:.3f}")


if __name__ == "__main__":
    demo()
