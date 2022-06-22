#!/usr/bin/env python2.7
from __future__ import print_function

import os
import argparse

import numpy as np

from tqdm import tqdm

import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from metrics import *
from datasets import *
from utils import *


# Constants
DATA_NAME = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def filter_dict(x, vocab):
    out = []
    for k in x:
        k_split = k.split(",")
        k = k_split[0]
        k_split = [y for y in k_split if y in vocab]
        if len(k_split) > 0:
            k = ",".join(k_split)
        out.append(k)
    return out


def main(args):
    data_dir = args.data_dir

    idx2word, word2idx = load_vocab(os.path.join(data_dir, "words_vocab.txt"))
    idx2cls, cls2idx = load_vocab(os.path.join(data_dir, "objects_vocab.txt"))
    idx2attr, attr2idx = load_vocab(
        os.path.join(data_dir, "attributes_vocab.txt"))
    idx2rel, rel2idx = load_vocab(
        os.path.join(data_dir, "relations_vocab.txt"))

    idx2cls = filter_dict(idx2cls, word2idx)
    idx2attr = filter_dict(idx2attr, word2idx)
    idx2rel = filter_dict(idx2rel, word2idx)

    with open("{}/objects_vocab_filtered.txt".format(data_dir), "w") as fh:
        for w in idx2cls:
            print(w, file=fh)
    with open("{}/attributes_vocab_filtered.txt".format(data_dir), "w") as fh:
        for w in idx2attr:
            print(w, file=fh)
    with open("{}/relations_vocab_filtered.txt".format(data_dir), "w") as fh:
        for w in idx2rel:
            print(w, file=fh)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing feature vectors")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
