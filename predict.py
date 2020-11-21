import torch
from torchtext.data.example import Example

from datasets.dataset import load_or_generate_dataset
from utils import get_available_device
from networks.network import create_seq2seq
import logging
logging.basicConfig(level='DEBUG')


if __name__ == '__main__':
    device = get_available_device()
    dataset = load_or_generate_dataset()
    src_field = dataset['src_field']
    trg_field = dataset['trg_field']

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
    network.load_state_dict(torch.load('weights/tut1-model.pt'))

    # sentence = input('Enter sentence in german: ')
    sentence = 'Ein Hund rennt im Schnee.'
    while sentence is not 'exit':
        # Convert custom sentence to tensor

        example = Example.fromlist([sentence], [('de', src_field)])
        batch = [example.de]
        idx_input = src_field.process(batch).to(device)

        # Translate this tensor
        output_probs = network(idx_input, None, 0)
        idx_output = output_probs.squeeze(1).argmax(axis=1)
        # TODO is actually probs, not idx

        # Convert back
        output_sentence = ' '.join([trg_field.vocab.itos[idx] for idx in idx_output])

        print(output_sentence)
        sentence = input('Enter sentence in german: ')

