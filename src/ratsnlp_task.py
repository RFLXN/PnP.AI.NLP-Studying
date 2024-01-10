from ratsnlp.nlpbook.classification import ClassificationTask


def init_task(model, args):
    return ClassificationTask(model, args)
