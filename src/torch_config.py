import torch


# set default tensor for Apple Silicon Metal support
def __set_apple_silicon_mps():
    if torch.backends.mps.is_available():
        torch.set_default_device("mps")
        torch.backends.quantized.engine = "qnnpack"


def set_torch_config():
    __set_apple_silicon_mps()
