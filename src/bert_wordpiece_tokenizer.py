from tokenizers import BertWordPieceTokenizer


def init_bert_wordpiece_tokenizer(dataset: str, size: int):
    tokenizer = BertWordPieceTokenizer()

    tokenizer.train(
        files=[f"../dataset/{dataset}-train.txt", f"../dataset/{dataset}-test.txt"],
        vocab_size=size
    )

    return tokenizer


def save_tokenized_dataset(dataset: str, tokenizer: BertWordPieceTokenizer):
    tokenizer.save_model(f"../token/wordpiece/{dataset}")