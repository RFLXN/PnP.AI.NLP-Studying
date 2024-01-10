from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
from ratsnlp.nlpbook import data_collator


def init_train_dataloader(args: ClassificationTrainArguments, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset, replacement=False),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=args.cpu_workers
    )


def init_test_dataloader(args: ClassificationTrainArguments, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=args.cpu_workers
    )
