from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset, ClassificationTrainArguments


def __init_options(args: ClassificationTrainArguments, tokenizer):
    corpus = NsmcCorpus()
    return {
        "args": args,
        "corpus": corpus,
        "tokenizer": tokenizer,
    }


def init_ratsnlp_train_dataset(args: ClassificationTrainArguments, tokenizer):
    option = __init_options(args, tokenizer)
    return ClassificationDataset(
        **option,
        mode="train"
    )


def init_ratsnlp_test_dataset(args: ClassificationTrainArguments, tokenizer):
    option = __init_options(args, tokenizer)
    return ClassificationDataset(
        **option,
        mode="test"
    )
