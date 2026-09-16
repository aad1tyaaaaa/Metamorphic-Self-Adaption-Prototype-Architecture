class ComplexityScorer:

    def __init__(self):

        self.weights = {
            "input_length": 0.25,
            "reasoning": 0.35,
            "domain": 0.20,
            "structure": 0.20,
        }

    def calculate(self, features):

        score = 0.0

        for feature, weight in self.weights.items():

            score += (
                weight * features[feature]
            )

        return score