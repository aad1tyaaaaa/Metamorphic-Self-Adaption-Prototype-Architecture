# main.py

```
python main.py
```

## Run 1

```
PROMPT:
The capital of India is

MODEL OUTPUT:
The capital of India is the capital of the world. It is the capital of the world. It is the capital of the world. It is the capital of the world.
```

## Run 2

```
PROMPT:
The capital of France is

MODEL OUTPUT:
The capital of France is France, the capital of the West, and it is the capital of the West, and it is

PROMPT:
2 + 2 equals

MODEL OUTPUT:
2 + 2 equals +1.

(2 + 2 equals +1. + 1 + 1)

PROMPT:
The largest planet in our solar system is

MODEL OUTPUT:
The largest planet in our solar system is about 30 billion miles (43.5 billion kilometers) in diameter, and its orbit is elliptical

PROMPT:
Python is a programming language used for

MODEL OUTPUT:
Python is a programming language used for programming in C++.

What is the difference between C and C++?

C
```

## Run 3 (model architecture printout)

```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```
