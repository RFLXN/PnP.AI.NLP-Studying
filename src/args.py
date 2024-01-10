from ratsnlp.nlpbook.classification import ClassificationTrainArguments, NsmcCorpus
from ratsnlp.nlpbook import set_seed, set_logger
import torch


def init_args():
    args = ClassificationTrainArguments(
        pretrained_model_name="beomi/kcbert-base",
        downstream_corpus_name="nsmc",
        downstream_model_dir="/content/drive/MyDrive/nlp/model/nlpbook/checkpoint-doccls",
        batch_size=2 if torch.cuda.is_available() else 4,
        learning_rate=5e-5,
        max_seq_length=128,
        epochs=3,
        tpu_cores=0 if torch.cuda.is_available() else 8,
        seed=7,
        downstream_corpus_root_dir="/content/drive/MyDrive/nlp/dataset/korpora"
    )

    set_seed(args)
    set_logger(args)
    return args
