#!/usr/bin/env python2.7
import os
import argparse

from tqdm import tqdm
from nlgeval import NLGEval
from nltk.translate.bleu_score import corpus_bleu

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim
import torch.utils.data

# TODO: import exactly what is needed
from datasets import *
from utils import *


# Constants
DATA_NAME = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
NLG_EVAL = NLGEval()  # loads the evaluator

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def evaluate(args):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    # DataLoader
    data_dir = args.data_dir
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_dir, DATA_NAME, 'TEST'),
        batch_size=args.batch_size, shuffle=False, num_workers=1,
        pin_memory=torch.cuda.is_available())

    # Lists to store references (true captions), and hypothesis (prediction)
    # for each image
    # If for n images, we have n hypotheses, and references a, b, c...
    # for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...],
    # hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    idx2word, word2idx = load_vocab(os.path.join(data_dir,"words_vocab.txt"))
    idx2cls, cls2idx = load_vocab(os.path.join(data_dir, "objects_vocab.txt"))
    idx2attr, attr2idx = load_vocab(
        os.path.join(data_dir, "attributes_vocab.txt"))

    vocab_size = len(word2idx)

    # Load model
    torch.nn.Module.dump_patches = True
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    decoder = checkpoint["decoder"]
    decoder = decoder.to(DEVICE)
    decoder.eval()

    # For each image
    batch_size = args.batch_size
    beam_size = args.beam_size
    for i, (image_features, caps, caplens, classes, attributes, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size

        # Move to GPU device, if available
        image_features = image_features.to(DEVICE)  # (1, 3, 256, 256)
        if decoder.input_projection is not None:
            image_features = decoder.input_projection(image_features)

        image_features_mean = image_features.mean(1)
        image_features_mean = image_features_mean.expand(k, 2048)

        # Tensor to store top k previous words at each step; now they're just <start>
        # (k, 1)
        k_prev_words = torch.LongTensor([[word2idx['<start>']]] * k).to(DEVICE)

        # Tensor to store top k sequences; now they're just <start>
        # (k, 1)
        seqs = k_prev_words

        # Tensor to store top k sequences' scores; now they're just 0
        # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(DEVICE)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        # (batch_size, decoder_dim)
        h1, c1 = decoder.init_hidden_state(k)
        h2, c2 = decoder.init_hidden_state(k)

        # s is a number less than or equal to k, because sequences are removed
        # from this process once they hit <end>
        while True:
            # (s, embed_dim)
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            # (batch_size_t, decoder_dim)
            h1,c1 = decoder.top_down_attention(
                torch.cat([h2,image_features_mean,embeddings], dim=1), (h1,c1))
            attention_weighted_encoding = decoder.attention(image_features,h1)
            h2,c2 = decoder.language_model(
                torch.cat([attention_weighted_encoding,h1], dim=1), (h2,c2))

            # (s, vocab_size)
            scores = decoder.fc(h2)
            scores = F.log_softmax(scores, dim=1)

            # Add
            # (s, vocab_size)
            scores = top_k_scores.expand_as(scores) + scores

            # For the first step, all k points will have the same scores
            # (since same k previous words, h, c)
            if step == 1:
                # (s)
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            # (s)
            prev_word_inds = top_k_words / vocab_size
            next_word_inds = top_k_words % vocab_size

            # Add new words to sequences
            # (s, step + 1)
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds) if
                next_word != word2idx['<end>']
            ]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            # reduce beam length accordingly
            k -= len(complete_inds)

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        # remove <start> and pads
        img_captions = list(
            map(lambda c: [idx2word[w] for w in c if w not in {word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']}],
                img_caps))
        img_caps = [' '.join(c) for c in img_captions]
        #print(img_caps)
        references.append(img_caps)

        # Hypotheses
        hypothesis = ([idx2word[w] for w in seq if w not in {word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']}])
        hypothesis = ' '.join(hypothesis)

        if args.print_captions:
            print(hypothesis)
        hypotheses.append(hypothesis)
        assert len(references) == len(hypotheses)

    # Calculate scores
    metrics_dict = NLG_EVAL.compute_metrics(references, hypotheses)
    return metrics_dict


def parse_args():
    parser = argparse.ArgumentParser(description="TODO")

    parser.add_argument("--print_captions", action="store_true",
                        help="Print decoded captions.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing the data.")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint file location.",)
    parser.add_argument("--batch_size", default=1,
                        help="Batch size.")
    parser.add_argument("--beam-size", default=5,
                        help="Beam size.",)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    metrics_dict = evaluate(args)
    print(metrics_dict)
