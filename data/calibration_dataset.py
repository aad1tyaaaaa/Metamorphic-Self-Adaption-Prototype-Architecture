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
# The original 499-prompt corpus, kept unchanged as a historical version.
V1_PROMPTS_PATH = os.path.join(HERE, "calibration_prompts_v1_499.json")

TARGET_SIZE = 800
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
    "ice floats on liquid water",
    "the sky turns red at sunset",
    "stars appear to twinkle at night",
    "salt lowers the freezing point of water",
    "muscles become sore after exercise",
    "volcanoes erupt",
    "magnets attract iron",
    "leaves change colour in autumn",
    "the human heart has four chambers",
    "hot air balloons rise",
]

TECH_TOPICS = [
    "TCP and UDP", "RAM and ROM", "a process and a thread",
    "a compiler and an interpreter", "HTTP and HTTPS", "SQL and NoSQL databases",
    "a stack and a queue", "authentication and authorization",
    "a virtual machine and a container", "latency and throughput",
    "a hash table and a binary search tree", "a TCP handshake and a TLS handshake",
    "horizontal and vertical scaling", "synchronous and asynchronous I/O",
    "a cache hit and a cache miss",
    "IPv4 and IPv6", "a mutex and a semaphore", "REST and GraphQL",
    "symmetric and asymmetric encryption", "a linked list and an array",
    "static and dynamic typing", "a CPU and a GPU", "git merge and git rebase",
    "unit tests and integration tests", "a router and a switch",
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
    "rotates a matrix by ninety degrees", "finds the second largest number in a list",
    "counts the vowels in a string", "checks whether two strings are anagrams",
    "computes the greatest common divisor of two integers",
    "generates the first n Fibonacci numbers", "transposes a list of lists",
    "groups a list of words by their first letter",
    "converts a Roman numeral to an integer", "checks whether brackets are balanced",
]

SUMMARY_SUBJECTS = [
    "the industrial revolution", "the theory of evolution",
    "the origins of the internet", "the printing press",
    "the discovery of penicillin", "the space race",
    "the invention of the transistor", "global supply chains",
    "renewable energy adoption", "the history of written language",
    "the fall of the Roman Empire", "the French Revolution",
    "the development of vaccines", "the invention of the steam engine",
    "the growth of cities", "the history of aviation",
    "the discovery of DNA", "the rise of social media",
    "the Green Revolution in agriculture", "the history of the Olympic Games",
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
    ("pilots", "trained professionals", "passengers"),
    ("oak trees", "plants", "fungi"),
    ("violins", "instruments", "drums"),
    ("novels", "books", "magazines"),
    ("cats", "pets", "wild animals"),
    ("triangles", "polygons", "circles"),
    ("doctors", "graduates", "patients"),
]

# Short factual questions. Deliberately disjoint from the short_qa evaluation
# set in data/load_data.py so calibration does not see evaluation prompts.
FACTUAL_QUESTIONS = [
    "What is the capital of Australia?", "What is the capital of Brazil?",
    "What is the capital of Argentina?", "What is the capital of Norway?",
    "What is the capital of Sweden?", "What is the capital of Portugal?",
    "What is the capital of Greece?", "What is the capital of Turkey?",
    "What is the capital of Mexico?", "What is the capital of Thailand?",
    "What is the capital of South Korea?", "What is the capital of Nigeria?",
    "What is the capital of Peru?", "What is the capital of Ireland?",
    "What is the capital of Poland?", "What is the capital of Vietnam?",
    "Who invented the telephone?", "Who invented the light bulb?",
    "Who discovered penicillin?", "Who discovered gravity?",
    "Who proposed the theory of evolution?", "Who invented the printing press?",
    "Who was the first president of the United States?",
    "Who composed the Fifth Symphony?", "Who wrote Pride and Prejudice?",
    "Who wrote The Odyssey?", "Who painted The Starry Night?",
    "Who discovered radium?", "Who built the first airplane?",
    "What is the smallest planet in the solar system?",
    "What is the hottest planet in the solar system?",
    "What is the largest mammal on Earth?", "What is the fastest land animal?",
    "What is the tallest animal?", "What is the largest bird?",
    "What is the chemical symbol for sodium?", "What is the chemical symbol for silver?",
    "What is the chemical symbol for carbon?", "What is the chemical symbol for helium?",
    "What is the atomic number of hydrogen?", "What is the most abundant gas in air?",
    "What is the largest country by area?", "What is the smallest country in the world?",
    "What is the longest river in South America?", "What is the largest lake in Africa?",
    "What is the highest waterfall in the world?", "What is the deepest ocean trench?",
    "What is the currency of the United States?", "What is the currency of Mexico?",
    "What is the currency of the European Union?", "What language is spoken in Argentina?",
    "What language is spoken in Egypt?", "What is the official language of Japan?",
    "How many planets are in the solar system?", "How many bones are in the adult human body?",
    "How many teeth does an adult human have?", "How many strings does a standard guitar have?",
    "How many minutes are in an hour?", "How many weeks are in a year?",
    "How many sides does an octagon have?", "How many colours are in a rainbow?",
    "When did the Second World War end?", "When did humans first land on the Moon?",
    "When did the Berlin Wall fall?", "Where is the Colosseum located?",
    "Where is Mount Kilimanjaro located?", "Where is the Sahara Desert?",
    "Which gas do humans exhale?", "Which organ filters blood in the body?",
    "Which metal is liquid at room temperature?",
]

# Requests for long-form explanation, distinct from the short "Explain why"
# science prompts and from the multi-part structured prompts.
LONG_EXPLANATION_SUBJECTS = [
    "how the internet routes a message from one computer to another",
    "how the human immune system fights an infection",
    "how a car engine converts fuel into motion",
    "how electricity is generated and delivered to homes",
    "how a search engine ranks web pages",
    "how vaccines are developed and tested",
    "how the stock market sets prices",
    "how climate change affects ocean ecosystems",
    "how a compiler turns source code into machine instructions",
    "how the heart and lungs work together",
    "how a neural network learns from data",
    "how bees communicate the location of food",
    "how banks create money through lending",
    "how earthquakes are measured and predicted",
    "how public key cryptography keeps messages secret",
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

    add(_arithmetic(rng, 100))

    add([(question, "factual") for question in FACTUAL_QUESTIONS])

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
        (f"Write a detailed explanation of {subject}. Start from the basic "
         f"principles, walk through each stage of the process in order, explain "
         f"why each stage is necessary, and finish with a common misconception.",
         "long_explanation")
        for subject in LONG_EXPLANATION_SUBJECTS
    ])
    add([
        (f"Explain {subject} to a curious high-school student, using an analogy "
         f"and at least two concrete examples.", "long_explanation")
        for subject in LONG_EXPLANATION_SUBJECTS
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
