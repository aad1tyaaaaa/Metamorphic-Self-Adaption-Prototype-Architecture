# test_adaptive.py

```
python test_adaptive.py
```

Initial attempts failed with:
```
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```
at `models/adaptive_model.py`, calling `GPT2Block` with only `hidden_states` (transformers 5.17.0 changed the block's call signature). Fixed by passing `attention_mask=attention_mask` explicitly and unwrapping the tuple return.

## Successful run

```
TESTING DEPTH: 4
Logits shape: torch.Size([1, 5, 50257])

TESTING DEPTH: 8
Logits shape: torch.Size([1, 5, 50257])

TESTING DEPTH: 12
Logits shape: torch.Size([1, 5, 50257])
```
