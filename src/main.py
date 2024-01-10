from multiprocessing import freeze_support
from platform import processor
from torch.backends.mps import is_available
# from torch_config import set_torch_config
from args import init_args
from dataset_korpora import download_ds
from bert_tokenizer import init_bert_tokenizer_repo
from ratsnlp_dataset import init_ratsnlp_train_dataset, init_ratsnlp_test_dataset
from bert_dataloader import init_test_dataloader, init_train_dataloader
from bert_model import init_model, init_config
from ratsnlp_task import init_task
from ratsnlp_trainer import get_trainer


def main():
    print("Arch: " + processor())
    print("MPS Support: " + str(is_available()))

    # init arguments
    args = init_args()
    print(args)

    # set torch global config
    """
    this book's tutorials make Apple Silicon processor unusable...fuck.
    actually pytorch and pytorch lightning support MPS devices.
    but this tutorial's stupid ratsnlp library make shitty situation...
    (edit): I think pytorch has bug too.
    only 2 weeks ago, someone talk about same error at torch github repo issue. 
    """
    # set_torch_config()

    # download Korpora datasets
    download_ds(args)

    # init tokenizer
    tokenizer = init_bert_tokenizer_repo(args.pretrained_model_name)

    # load dataset
    train_dataset = init_ratsnlp_train_dataset(args, tokenizer)
    test_dataset = init_ratsnlp_test_dataset(args, tokenizer)

    # init data loaders
    train_data_loader = init_train_dataloader(args, train_dataset)
    test_data_loader = init_test_dataloader(args, test_dataset)

    # init model
    model_config = init_config(args)
    model = init_model(args, model_config)

    # fit model
    task = init_task(model, args)
    trainer = get_trainer(args)
    trainer.fit(task, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)


if __name__ == '__main__':
    freeze_support()
    main()
