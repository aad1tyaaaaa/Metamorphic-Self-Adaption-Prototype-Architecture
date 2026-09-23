"""Downstream evaluation datasets (Phase 14).

Each loader returns records of {question, answer, category}. Evaluation scores
the answer tokens conditioned on the question, which is a downstream metric
rather than the prompt loss used for calibration.

    gsm8k       real multi-step word problems (HuggingFace openai/gsm8k)
    short_qa    short factual questions with one-word answers (local)
"""

import re

# Short factual QA. Held locally so the easy-task evaluation needs no network
# and the reference answers are unambiguous single tokens where possible.
SHORT_QA = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of China?", "Beijing"),
    ("What is the capital of Kenya?", "Nairobi"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the closest planet to the Sun?", "Mercury"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for oxygen?", "O"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What gas do plants absorb from the air?", "Carbon"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many days are in a week?", "7"),
    ("How many months are in a year?", "12"),
    ("How many hours are in a day?", "24"),
    ("How many continents are there?", "7"),
    ("How many legs does a spider have?", "8"),
    ("How many sides does a triangle have?", "3"),
    ("How many sides does a hexagon have?", "6"),
    ("How many players are on a soccer team on the field?", "11"),
    ("What is 2 plus 2?", "4"),
    ("What is 5 times 6?", "30"),
    ("What is 100 divided by 4?", "25"),
    ("What is the square root of 81?", "9"),
    ("What is 7 squared?", "49"),
    ("Who wrote the play Romeo and Juliet?", "Shakespeare"),
    ("Who developed the theory of relativity?", "Einstein"),
    ("Who was the first person to walk on the Moon?", "Armstrong"),
    ("Who painted the Mona Lisa?", "Leonardo"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the longest river in Africa?", "Nile"),
    ("What is the tallest mountain in the world?", "Everest"),
    ("What is the largest desert in the world?", "Antarctica"),
    ("What language is primarily spoken in Brazil?", "Portuguese"),
    ("What language is primarily spoken in Austria?", "German"),
    ("What currency is used in Japan?", "Yen"),
    ("What currency is used in the United Kingdom?", "Pound"),
    ("What is the currency of India?", "Rupee"),
    ("In which country is the Eiffel Tower?", "France"),
    ("In which country is the Great Wall?", "China"),
    ("In which country is the Taj Mahal?", "India"),
    ("What colour is the sky on a clear day?", "Blue"),
    ("What colour do you get by mixing red and white?", "Pink"),
    ("What is the opposite of hot?", "Cold"),
    ("What is the opposite of ancient?", "Modern"),
    ("What is the first month of the year?", "January"),
    ("What is the last month of the year?", "December"),
    ("What season comes after winter?", "Spring"),
    ("What organ pumps blood around the body?", "Heart"),
    ("What organ is used for breathing?", "Lungs"),
    ("What is the hardest natural substance?", "Diamond"),
    ("What do bees produce?", "Honey"),
    ("What is the young of a cat called?", "Kitten"),
]


def load_short_qa(limit=None):
    records = [
        {"question": question, "answer": answer, "category": "short_qa"}
        for question, answer in SHORT_QA
    ]
    return records[:limit] if limit else records


def load_gsm8k(limit=100, split="test"):
    import warnings
    warnings.filterwarnings("ignore")
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split=f"{split}[:{limit}]")

    records = []
    for row in data:
        # GSM8K answers end with "#### <final answer>".
        final = row["answer"].split("####")[-1].strip()
        records.append({
            "question": row["question"].strip(),
            "answer": final,
            "reasoning": row["answer"].split("####")[0].strip(),
            "category": "gsm8k",
        })

    return records


DATASETS = {"short_qa": load_short_qa, "gsm8k": load_gsm8k}


def load_dataset_by_name(name, limit=None):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(DATASETS)}")
    return DATASETS[name](limit) if limit else DATASETS[name]()


NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def exact_match(generated, answer):
    """Loose exact match: numeric answers compare numerically, text case-folded."""
    generated = generated.strip()

    if NUMBER.fullmatch(answer.replace(",", "")):
        found = NUMBER.findall(generated.replace(",", ""))
        if not found:
            return False
        try:
            return abs(float(found[0]) - float(answer.replace(",", ""))) < 1e-6
        except ValueError:
            return False

    return answer.lower() in generated.lower()


if __name__ == "__main__":
    qa = load_short_qa()
    print(f"short_qa: {len(qa)} records, e.g. {qa[0]}")

    assert exact_match("The answer is 42", "42")
    assert exact_match(" Paris is the capital", "Paris")
    assert not exact_match("London", "Paris")
    assert not exact_match("no digits here", "42")

    try:
        gsm = load_gsm8k(limit=3)
        print(f"gsm8k: {len(gsm)} records, answer of first = {gsm[0]['answer']!r}")
    except Exception as error:
        print(f"gsm8k unavailable ({type(error).__name__}); short_qa still usable")

    print("load_data demo OK")
