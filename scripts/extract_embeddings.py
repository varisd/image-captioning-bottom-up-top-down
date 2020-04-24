#!/usr/bin/env python2.7
from __future__ import print_function

import os
import argparse

import numpy as np
import torch

import torch.backends.cudnn as cudnn

from datasets import load_vocab

# Constants
DATA_NAME = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main(args):
    torch.nn.Module.dump_patches = True
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    decoder = checkpoint["decoder"]
    decoder = decoder.to(DEVICE)
    decoder.eval()

    idx2word, word2idx = load_vocab(os.path.join(args.data_dir, "words_vocab.txt"))

    embeddings = decoder.embedding.weight.detach().numpy()
    if args.format == "plaintext":
        with open("{}/embeddings.txt".format(args.output_dir), "w") as fh:
            for i in range(embeddings.shape[0]):
                print(
                    "{} {}".format(
                        idx2word[i],
                        " ".join(embeddings[i].astype(np.str))),
                    file=fh)
    elif args.format == "numpy":
        np.save("{}/embeddings".format(args.output_dir), embeddings)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="TODO")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="TODO")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="TODO")
    parser.add_argument(
        "--format", type=str, default="plaintext",
        help="TODO")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
