from tokenizers import ByteLevelBPETokenizer


def init_bbpe_tokenizer(dataset: str, size: int, special_tokens=list[str]):
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[f"../dataset/{dataset}-train.txt", f"../dataset/{dataset}-test.txt"],
        vocab_size=size,
        special_tokens=special_tokens
    )

    return tokenizer


def save_tokenized_dataset(dataset: str, tokenizer: ByteLevelBPETokenizer):
    tokenizer.save_model(f"../token/bbpe/{dataset}")

