from transformers import BertConfig, BertForSequenceClassification, PretrainedConfig
from ratsnlp.nlpbook.classification import ClassificationTrainArguments, NsmcCorpus


def init_config(args: ClassificationTrainArguments):
    return BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=NsmcCorpus().num_labels
    )


def init_model(args: ClassificationTrainArguments, cfg: PretrainedConfig):
    return BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=cfg
    )
