from transformers import GPT2Tokenizer


def init_gpt2_tokenizer(dataset: str):
    tokenizer = GPT2Tokenizer.from_pretrained(f"../token/bbpe/{dataset}")
    tokenizer.pad_token = "[PAD]"

    return tokenizer
