import math
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from utils import set_deterministic_mode, get_available_device, count_parameters
from datasets.dataset import load_or_generate_dataset
from networks.network import create_seq2seq
from models.model import Model


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')

    set_deterministic_mode()
    device = get_available_device()
    dataset = load_or_generate_dataset()

    network_params = {
      'INPUT_DIM': dataset['n_src_words'],
      'OUTPUT_DIM': dataset['n_trg_words'],
      'ENC_EMB_DIM': 256,
      'DEC_EMB_DIM': 256,
      'ENC_HID_DIM': 512,
      'DEC_HID_DIM': 512,
      'ENC_DROPOUT': 0.5,
      'DEC_DROPOUT': 0.5
    }
    network = create_seq2seq(network_params, device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    network.apply(init_weights)

    logging.debug(f'The model has {count_parameters(network):,} trainable parameters')

    optimizer = optim.Adam(network.parameters())

    # ignore <pad> token at the end of sentences when calculating loss
    criterion = nn.CrossEntropyLoss(ignore_index=dataset['trg_pad_idx'],)

    model = Model(network, optimizer, criterion)
    model.train_epochs(dataset['train_data'], dataset['valid_data'])

    network.load_state_dict(torch.load('weights/tut1-model.pt'))

    test_loss = model.evaluate(dataset['test_data'])

    logging.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    # model.predict(dataset['src_vocab'].)
