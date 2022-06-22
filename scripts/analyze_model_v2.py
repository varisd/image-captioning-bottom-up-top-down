#!/usr/bin/env python2.7
from __future__ import print_function

import os
import sys
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
MAX_SIZE = 25000

cudnn.benchmark = True


def main(args):
    data_dir = args.data_dir
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_dir, DATA_NAME, args.split),
        batch_size=100, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available())

    idx2word, word2idx = load_vocab(os.path.join(args.data_dir, "words_vocab.txt"))
    idx2cls, cls2idx = load_vocab(os.path.join(args.data_dir, "objects_vocab.txt"))
    idx2attr, attr2idx = load_vocab(
        os.path.join(data_dir, "attributes_vocab.txt"))

    torch.nn.Module.dump_patches = True
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    decoder = checkpoint["decoder"]
    decoder = decoder.to(DEVICE)
    decoder.eval()

    embeddings = decoder.embedding.weight.detach().numpy()

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

    centroid_labels = np.array([idx for idx, _ in orig_centroids.items()])
    orig_centroids = np.stack([v for _, v in orig_centroids.items()], axis=0)
    proj_centroids = np.stack([v for _, v in proj_centroids.items()], axis=0)

    embeddings_subset = embeddings[centroid_labels]

    # mNNO
    print("mNNO (orig): {}".format(
        mean_nearest_neighbor_overlap(
            orig_centroids, embeddings_subset, args.num_neighbors)))
    print("mNNO (proj): {}".format(
        mean_nearest_neighbor_overlap(
            proj_centroids, embeddings_subset, args.num_neighbors)))

    vector_indices = np.concatenate(
        [
            [
                idx for _ in  orig_vectors[idx]
            ] for idx in orig_vectors
        ],
        axis=0)
    orig_vectors = np.concatenate(
        [vec for _, vec in orig_vectors.items()], axis=0)
    proj_vectors = np.concatenate(
        [vec for _, vec in proj_vectors.items()], axis=0)

    res = {
        "c_intra_orig": [],
        "c_intra_proj": [],
        "c_inter_orig": [],
        "c_inter_proj": [],
        "pearson_orig": [],
        "pearson_proj": []
    }
    indices_all = np.random.permutation(vector_indices.shape[0])
    for i in range((indices_all.shape[0] / MAX_SIZE) - 1):
        indices = indices_all[(i * MAX_SIZE):((i+1) * MAX_SIZE)]
        v_indices_subset = vector_indices[indices]
        orig_subset = orig_vectors[indices]
        proj_subset = proj_vectors[indices]

        # Cluster information
        # mask_intra: mask values NOT USED for the C_intra computation
        # mask_inter: see mask_intra 
        mask_intra = np.equal(
            np.expand_dims(v_indices_subset, 0),
            np.expand_dims(v_indices_subset, 1)).astype(np.float32)
        np.fill_diagonal(mask_intra, 0)
        mask_inter = 1 - mask_intra
        np.fill_diagonal(mask_inter, 0)

        orig_sim = cosine_similarity(orig_subset)
        res["c_intra_orig"].append(masked_mean(orig_sim, mask_intra))
        res["c_inter_orig"].append(masked_mean(orig_sim, mask_inter))

        proj_sim = cosine_similarity(proj_subset)
        res["c_intra_proj"].append(masked_mean(proj_sim, mask_intra))
        res["c_inter_proj"].append(masked_mean(proj_sim, mask_inter))

        # Correlation (only between non-equal classes)
        embedded = embeddings[v_indices_subset]
        res["pearson_orig"].append(
            pearson(orig_subset, embedded, mask_inter))
        res["pearson_proj"].append(
            pearson(proj_subset, embedded, mask_inter))

    print("C intra (orig): {0:.3f}({1:.3f})".format(
        np.mean(100*res["c_intra_orig"]), 100*np.var(res["c_intra_orig"])))
    print("C inter (orig): {0:.3f}({1:.3f})".format(
        np.mean(100*res["c_inter_orig"]), 100*np.var(res["c_inter_orig"])))
    print("C intra (proj): {0:.3f}({1:.3f})".format(
        np.mean(100*res["c_intra_proj"]), 100*np.var(res["c_intra_proj"])))
    print("C inter (proj): {0:.3f}({1:.3f})".format(
        np.mean(100*res["c_inter_proj"]), 100*np.var(res["c_inter_proj"])))
    print("Pearson (orig): {0:.3f}({1:.3f})".format(
        np.mean(100*res["pearson_orig"]), 100*np.var(res["pearson_orig"])))
    print("Pearson (proj): {0:.3f}({1:.3f})".format(
        np.mean(100*res["pearson_proj"]), 100*np.var(res["pearson_proj"])))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing feature vectors")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="TODO")
    parser.add_argument(
        "--split", type=str, default="TEST",
        help="TODO")
    parser.add_argument(
        "--k_fold", type=int, default=10,
        help="How many samples to examine")
    parser.add_argument(
        "--num_neighbors", type=int, default=3,
        help="TODO")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
