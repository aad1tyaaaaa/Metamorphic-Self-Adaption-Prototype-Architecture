from analyzer.task_analyzer import TaskAnalyzer
from controller.complexity import ComplexityScorer
from controller.selector import ConfigurationSelector


analyzer = TaskAnalyzer()

scorer = ComplexityScorer()

selector = ConfigurationSelector()


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


for text in texts:

    features = analyzer.analyze(text)

    complexity = scorer.calculate(features)

    configuration = selector.select(
        complexity
    )

    print("\n" + "=" * 60)

    print("INPUT:")
    print(text.strip())

    print("\nFEATURES:")
    print(features)

    print(
        f"\nCOMPLEXITY: {complexity:.3f}"
    )

    print(
        f"CONFIGURATION: {configuration.upper()}"
    )