from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODEL_NAME = "gpt2"


def load_model():

    print("Loading GPT-2...")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    print("GPT-2 loaded successfully.")

    return model, tokenizer


if __name__ == "__main__":

    model, tokenizer = load_model()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {parameters:,}")
    print(f"Transformer layers: {len(model.transformer.h)}")