# run_msa.py

```
python run_msa.py
```

```
INPUT: What is 2 + 2?
FEATURES: input_length 0.050 | reasoning 0.000 | domain 0.000 | structure 0.100
COMPLEXITY: 0.033
CONFIGURATION: SHALLOW
SELECTED DEPTH: 4
LOGITS SHAPE: torch.Size([1, 6, 50257])

INPUT: Explain why the sky appears blue.
FEATURES: input_length 0.060 | reasoning 0.400 | domain 0.000 | structure 0.100
COMPLEXITY: 0.175
CONFIGURATION: SHALLOW
SELECTED DEPTH: 4
LOGITS SHAPE: torch.Size([1, 8, 50257])

INPUT: Calculate the percentage increase from 50 to 75.
FEATURES: input_length 0.080 | reasoning 0.200 | domain 0.400 | structure 0.100
COMPLEXITY: 0.190
CONFIGURATION: SHALLOW
SELECTED DEPTH: 4
LOGITS SHAPE: torch.Size([1, 11, 50257])

INPUT:
John has 5 apples. He buys 3 more apples.
He then gives 2 apples to Sarah.
Explain how many apples John has remaining.
FEATURES: input_length 0.230 | reasoning 0.200 | domain 0.000 | structure 0.400
COMPLEXITY: 0.208
CONFIGURATION: SHALLOW
SELECTED DEPTH: 4
LOGITS SHAPE: torch.Size([1, 44, 50257])
```
