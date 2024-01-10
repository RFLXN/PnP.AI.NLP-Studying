from transformers import BertTokenizer


def init_bert_tokenizer_ds(dataset: str):
    tokenizer = BertTokenizer.from_pretrained(
        f"../token/wordpiece/{dataset}",
        do_lower_case=False
    )

    return tokenizer


def init_bert_tokenizer_repo(repo: str):
    tokenizer = BertTokenizer.from_pretrained(
        repo,
        do_lower_case=False
    )

    return tokenizer
