from ratsnlp.nlpbook import get_trainer as rntrainer
from lightning import Trainer


def get_trainer(args):
    return rntrainer(args)
