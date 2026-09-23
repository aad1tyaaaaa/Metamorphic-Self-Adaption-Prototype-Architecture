"""Calibration prompt set (Phase 4).

The prompts live in `data/calibration_prompts.json` so that calibration is
reproducible and needs no network access. Rebuild the file with:

    python -m data.calibration_dataset --build
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(HERE, "calibration_prompts.json")
SEED_PATH = os.path.join(HERE, "_seed_prompts.json")

TARGET_SIZE = 500
SEED = 42


def load():
    with open(PROMPTS_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    return [entry["text"] for entry in payload], payload


# ----------------------------------------------------------------------
# Dataset construction
# ----------------------------------------------------------------------

SCIENCE_TOPICS = [
    "photosynthesis converts sunlight into chemical energy",
    "the greenhouse effect warms the atmosphere",
    "vaccines train the immune system",
    "tectonic plates cause earthquakes",
    "the water cycle moves water between ocean and air",
    "antibiotic resistance develops in bacteria",
    "the moon causes ocean tides",
    "sound travels faster in water than in air",
    "metals conduct electricity better than plastics",
    "the human eye perceives colour",
    "enzymes speed up chemical reactions",
    "erosion reshapes river valleys",
    "a rainbow forms after rainfall",
    "birds migrate across continents each year",
    "the seasons change through the year",
]

TECH_TOPICS = [
    "TCP and UDP", "RAM and ROM", "a process and a thread",
    "a compiler and an interpreter", "HTTP and HTTPS", "SQL and NoSQL databases",
    "a stack and a queue", "authentication and authorization",
    "a virtual machine and a container", "latency and throughput",
    "a hash table and a binary search tree", "a TCP handshake and a TLS handshake",
    "horizontal and vertical scaling", "synchronous and asynchronous I/O",
    "a cache hit and a cache miss",
]

CODE_TASKS = [
    "reverses a string", "checks whether a number is prime",
    "computes the factorial of an integer", "merges two sorted lists",
    "counts word frequencies in a paragraph", "removes duplicates from a list",
    "finds the largest element in an array", "validates an email address",
    "converts Celsius to Fahrenheit", "implements binary search",
    "flattens a nested list", "computes a moving average",
    "parses a CSV row into fields", "detects a palindrome",
    "sums the digits of a number",
]

SUMMARY_SUBJECTS = [
    "the industrial revolution", "the theory of evolution",
    "the origins of the internet", "the printing press",
    "the discovery of penicillin", "the space race",
    "the invention of the transistor", "global supply chains",
    "renewable energy adoption", "the history of written language",
]

LOGIC_TEMPLATES = [
    "All {a} are {b}. Some {b} are {c}. Does it follow that some {a} are {c}? Explain why or why not.",
    "If every {a} needs a {b}, and no {c} has a {b}, can a {c} be one of the {a}? Explain your reasoning.",
    "Either the {a} are in the {b} or in the {c}. They are not in the {b}. Where are they, and why?",
]

LOGIC_NOUNS = [
    ("engineers", "problem solvers", "artists"),
    ("mammals", "warm-blooded animals", "reptiles"),
    ("squares", "rectangles", "rhombuses"),
    ("students", "readers", "athletes"),
    ("bridges", "structures", "tunnels"),
]


def _arithmetic(rng, count):
    prompts = []
    for _ in range(count):
        kind = rng.choice(["add", "percent", "rate", "chain"])
        if kind == "add":
            a, b = rng.randint(12, 999), rng.randint(12, 999)
            prompts.append((f"Calculate {a} + {b}.", "basic_arithmetic"))
        elif kind == "percent":
            pct = rng.choice([5, 10, 15, 20, 25, 40, 60])
            total = rng.randint(40, 2000)
            prompts.append((f"What is {pct} percent of {total}?", "basic_arithmetic"))
        elif kind == "rate":
            dist, hours = rng.randint(60, 600), rng.randint(2, 9)
            prompts.append((
                f"A train travels {dist} kilometers in {hours} hours. "
                "What is its average speed?", "multi_step_arithmetic"))
        else:
            start = rng.randint(5, 60)
            bought = rng.randint(2, 30)
            sold = rng.randint(1, 10)
            prompts.append((
                f"A shop starts with {start} boxes, receives {bought} more, "
                f"then sells {sold}. How many boxes remain, and what is "
                "20 percent of that number?", "multi_step_arithmetic"))
    return prompts


def _gsm8k(count):
    """Real multi-step word problems. Cached locally after the first build."""
    import warnings
    warnings.filterwarnings("ignore")
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split=f"train[:{count}]")
    return [(row["question"].strip(), "hard_reasoning") for row in data]


def build():
    rng = random.Random(SEED)
    entries = []

    with open(SEED_PATH, encoding="utf-8") as handle:
        for text in json.load(handle):
            entries.append({"text": " ".join(text.split()), "category": "seed_handwritten"})

    def add(pairs):
        for text, category in pairs:
            entries.append({"text": " ".join(text.split()), "category": category})

    add(_arithmetic(rng, 60))

    add([(f"Explain why {topic}.", "scientific_explanation") for topic in SCIENCE_TOPICS])
    add([
        (f"Explain in detail why {topic}, describe the main stages involved, "
         f"and give one everyday example.", "scientific_explanation")
        for topic in SCIENCE_TOPICS
    ])

    add([(f"What is the difference between {topic}?", "technical") for topic in TECH_TOPICS])
    add([
        (f"Compare {topic}. Explain one advantage of each, and describe a "
         f"situation where you would choose one over the other.", "technical")
        for topic in TECH_TOPICS
    ])

    add([(f"Write a Python function that {task}.", "code") for task in CODE_TASKS])
    add([
        (f"Write a Python function that {task}. Explain how it works, "
         f"describe its time complexity, and list one edge case it must handle.", "code")
        for task in CODE_TASKS
    ])

    add([(f"Summarize {subject} in two sentences.", "summarization")
         for subject in SUMMARY_SUBJECTS])
    add([
        (f"Summarize {subject}. Cover the main causes, the most important "
         f"consequences, and explain why it still matters today.", "summarization")
        for subject in SUMMARY_SUBJECTS
    ])

    add([
        (template.format(a=a, b=b, c=c), "logical_reasoning")
        for template in LOGIC_TEMPLATES
        for a, b, c in LOGIC_NOUNS
    ])

    add([
        (f"You are given a report about {subject}. First, identify the three "
         f"most important claims. Second, explain which claim is best supported "
         f"by evidence. Third, describe what additional information would be "
         f"needed to evaluate the weakest claim.", "long_structured")
        for subject in SUMMARY_SUBJECTS
    ])
    add([
        (f"Consider a system that handles {topic}. Describe the components "
         f"involved, explain how they interact, and evaluate one failure mode.",
         "long_structured")
        for topic in TECH_TOPICS
    ])

    remaining = TARGET_SIZE - len(entries)
    if remaining > 0:
        add(_gsm8k(remaining))

    # De-duplicate while preserving order, then trim to the target size.
    seen, unique = set(), []
    for entry in entries:
        if entry["text"] not in seen:
            seen.add(entry["text"])
            unique.append(entry)

    unique = unique[:TARGET_SIZE]

    with open(PROMPTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(unique, handle, indent=1, ensure_ascii=False)

    return unique


if __name__ == "__main__":
    import sys
    from collections import Counter

    if "--build" in sys.argv:
        entries = build()
        print(f"Built {len(entries)} prompts -> {PROMPTS_PATH}")
    else:
        _, entries = load()
        print(f"Loaded {len(entries)} prompts")

    for category, count in Counter(e["category"] for e in entries).most_common():
        print(f"  {category:24s} {count}")
