import re


class TaskAnalyzer:

    def __init__(self):
        pass

    def analyze(self, text):

        return {
            "input_length": self.input_length(text),
            "reasoning": self.reasoning_complexity(text),
            "domain": self.domain_score(text),
            "structure": self.structural_complexity(text),
        }

    # --------------------------------------------------
    # Feature 1: Input length
    # --------------------------------------------------

    def input_length(self, text):

        words = text.split()

        score = len(words) / 100

        return min(score, 1.0)

    # --------------------------------------------------
    # Feature 2: Reasoning complexity
    # --------------------------------------------------

    def reasoning_complexity(self, text):

        reasoning_words = [
            "why",
            "because",
            "therefore",
            "explain",
            "prove",
            "derive",
            "calculate",
            "reason",
            "compare",
            "evaluate",
        ]

        text_lower = text.lower()

        count = sum(
            text_lower.count(word)
            for word in reasoning_words
        )

        score = count / 5

        return min(score, 1.0)

    # --------------------------------------------------
    # Feature 3: Domain complexity
    # --------------------------------------------------

    def domain_score(self, text):

        math_words = [
            "calculate",
            "equation",
            "number",
            "percentage",
            "multiply",
            "divide",
            "algebra",
            "geometry",
            "probability",
        ]

        text_lower = text.lower()

        count = sum(
            text_lower.count(word)
            for word in math_words
        )

        score = count / 5

        return min(score, 1.0)

    # --------------------------------------------------
    # Feature 4: Structural complexity
    # --------------------------------------------------

    def structural_complexity(self, text):

        sentences = len(
            re.findall(
                r"[.!?]",
                text
            )
        )

        commas = text.count(",")

        parentheses = (
            text.count("(")
            + text.count(")")
        )

        score = (
            sentences
            + commas
            + parentheses
        ) / 10

        return min(score, 1.0)


if __name__ == "__main__":

    analyzer = TaskAnalyzer()

    examples = [
        "What is 2 + 2?",
        "Explain why the sky appears blue.",
        "Calculate the percentage increase from 50 to 75.",
        "John has 5 apples. He gives 2 to Tom. How many does he have left?",
    ]

    for text in examples:

        print("\nTEXT:")
        print(text)

        print("\nFEATURES:")

        features = analyzer.analyze(text)

        for name, value in features.items():

            print(
                f"{name:20s}: {value:.3f}"
            )