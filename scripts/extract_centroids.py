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
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_dir, DATA_NAME, 'TEST'),
        batch_size=100, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available())

    idx2word, word2idx = load_vocab(
        os.path.join(args.data_dir, "words_vocab.txt"))
    idx2cls, cls2idx = load_vocab(
        os.path.join(args.data_dir, "objects_vocab.txt"))

    torch.nn.Module.dump_patches = True
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    decoder = checkpoint["decoder"]
    decoder = decoder.to(DEVICE)
    decoder.eval()

    # Extract vectors
    orig_vectors = {}
    proj_vectors = {}
    for i, (image_features, caps, caplens, classes, attributes, allcaps) in enumerate(
            tqdm(loader, desc="PROCESSING")):
        image_features = image_features.to(DEVICE)
        image_features = image_features.view(-1, 2048)
        image_features_proj = image_features
        if decoder.input_projection is not None:
            image_features_proj = decoder.input_projection(image_features)

        image_features = image_features.unsqueeze(1).detach().numpy()
        image_features_proj = image_features_proj.unsqueeze(1).detach().numpy()

        classes = classes.to(DEVICE)
        indices = reindex_np(classes, idx2cls, word2idx)
        indices = indices.reshape([-1])

        for i, idx in enumerate(indices):
            # ignore <pad>
            if idx == 0:
                continue
            if idx not in proj_vectors:
                orig_vectors[idx] = [image_features[i]]
                proj_vectors[idx] = [image_features_proj[i]]
            else:
                orig_vectors[idx].append(image_features[i])
                proj_vectors[idx].append(image_features_proj[i])
                
    orig_centroids = {}
    for idx, vectors in orig_vectors.items():
        orig_vectors[idx] = np.concatenate(orig_vectors[idx], 0)
        orig_centroids[idx] = np.mean(orig_vectors[idx], axis=0)
    
    proj_centroids = {}
    for idx, vectors in proj_vectors.items():
        proj_vectors[idx] = np.concatenate(proj_vectors[idx], 0)
        proj_centroids[idx] = np.mean(proj_vectors[idx], axis=0)

    orig_centroids = np.stack([v for _, v in orig_centroids.items()], axis=0)
    proj_centroids = np.stack([v for _, v in proj_centroids.items()], axis=0)

    if args.format == "plaintext":
        with open("{}/centroids_orig.txt".format(args.output_dir), "w") as fh:
            for i in range(orig_centroids.shape[0]):
                print(
                    "{} {}".format(
                        idx2word[i],
                        " ".join(orig_centroids[i].astype(np.str))),
                    file=fh)
        with open("{}/centroids_proj.txt".format(args.output_dir), "w") as fh:
            for i in range(proj_centroids.shape[0]):
                print(
                    "{} {}".format(
                        idx2word[i],
                        " ".join(proj_centroids[i].astype(np.str))),
                    file=fh)
    elif args.format == "numpy":
        np.save("{}/centroids_orig".format(args.output_dir), orig_centroids)
        np.save("{}/centroids_proj".format(args.output_dir), proj_centroids)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing feature vectors")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="TODO")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="TODO")
    parser.add_argument(
        "--split", type=str, default="TEST",
        help="TODO")
    parser.add_argument(
        "--format", type=str, default="plaintext",
        help="TODO")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
