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


def main(args):
    data_dir = args.data_dir

    idx2word, word2idx = load_vocab("{}/words_vocab.txt".format(data_dir))
    idx2cls, cls2idx = load_vocab("{}/objects_vocab.txt".format(data_dir))
    idx2attr, attr2idx = load_vocab("{}/attributes_vocab.txt".format(data_dir))
    idx2rel, rel2idx = load_vocab("{}/relations_vocab.txt".format(data_dir))

    print("Vocab size: {}".format(len(idx2word)))
    print("Classes size: {}".format(len(idx2cls)))
    print("Attributes size: {}".format(len(idx2attr)))
    print("Relations size: {}".format(len(idx2rel)))
    print("Classes overlap: {}".format(len([w for w in cls2idx if w in word2idx])))
    print("Attributes overlap: {}".format(len([w for w in attr2idx if w in word2idx])))
    print("Relations overlap: {}".format(len([w for w in rel2idx if w in word2idx])))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing feature vectors")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
