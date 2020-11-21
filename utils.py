import numpy as np
import random
import logging
import subprocess

import torch


def set_deterministic_mode(mode=1234):
    random.seed(mode)
    np.random.seed(mode)
    torch.manual_seed(mode)
    torch.cuda.manual_seed(mode)
    torch.backends.cudnn.deterministic = True


def get_available_device():
    if torch.cuda.is_available():
        logging.debug('using CUDA')
        return torch.device('cuda')
    else:
        logging.debug('CUDA not available, using CPU')
        return 'cpu'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_temperature():
    p = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    (output, _) = p.communicate()
    return output.splitlines()[8].split(b' ')[4]
