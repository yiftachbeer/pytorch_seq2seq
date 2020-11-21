from functools import partial

import spacy
import logging

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from utils import get_available_device

END_TOKEN = '<eos>'
START_TOKEN = '<sos>'


def download_spacy_models():
    import os
    os.system('python -m spacy download en')
    os.system('python -m spacy download de')


def load_tokenize_models():
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    return spacy_en, spacy_de


def tokenize_en(text, spacy_en):
    """
    Tokenizes English text from a string into a list of strings (tokens).
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_de(text, spacy_de):
    """
    Tokenizes German text from a string into a list of strings (tokens).
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def load_or_generate_dataset(batch_size=64):
    # download_spacy_models()  # TODO check if need to download then do it
    spacy_en, spacy_de = load_tokenize_models()

    SRC = Field(tokenize=partial(tokenize_de, spacy_de=spacy_de),
                init_token=START_TOKEN,
                eos_token=END_TOKEN,
                lower=True)
    TRG = Field(tokenize=partial(tokenize_en, spacy_en=spacy_en),
                init_token=START_TOKEN,
                eos_token=END_TOKEN,
                lower=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

    logging.debug(f"Number of training examples: {len(train_data.examples)}")
    logging.debug(f"Number of validation examples: {len(valid_data.examples)}")
    logging.debug(f"Number of testing examples: {len(test_data.examples)}")

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    logging.debug(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    logging.debug(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=get_available_device())

    # TODO class
    return {
           'train_data': train_iterator,
           'valid_data': valid_iterator,
           'test_data': test_iterator,
           'n_src_words': len(SRC.vocab),
           'n_trg_words': len(TRG.vocab),
           'trg_pad_idx': TRG.vocab.stoi[TRG.pad_token],
           'src_vocab': SRC.vocab,
           'trg_vocab': TRG.vocab,
           'src_field': SRC,
           'trg_field': TRG,
    }


