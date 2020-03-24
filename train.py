#!/usr/bin/env python2.7
import os
import argparse
import time

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from nltk.translate.bleu_score import corpus_bleu
from models import DecoderWithAttention
from losses import ClusterLoss, PerceptualLoss

# TODO: import exactly what is needed
from datasets import *
from utils import *

# Data parameters
DATA_NAME = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
EMB_DIM = 1024  # dimension of word embeddings
ATTENTION_DIM = 1024  # dimension of attention linear layers
DECODER_DIM = 1024  # dimension of decoder RNN
DROPOUT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

BATCH_SIZE = 100
EPOCHS = 50  # number of epochs to train for (if early stopping is not triggered)
WORKERS = 1  # for data-loading; right now, only 1 works with h5py
PRINT_FREQ = 100  # print training/validation stats every __ batches


def parse_args():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("--expdir", required=True,
                        help="Experiment directory.")
    parser.add_argument("--input_projection", default=None,
                        help="Apply a projection on the input.")
    parser.add_argument("--use_perceptual_loss", action="store_true")
    parser.add_argument("--use_cluster_loss", action="store_true")

    args = parser.parse_args()
    return args


def reindex(x, x2str, str2y):
    z = x2str[x]
    if z in str2y:
        return str2y[z]
    else:
        return 0


def reindex_np(x, x2str, str2y):
    y = torch.tensor([[reindex(b, x2str, str2y) for b in a] for a in x])
    return y.to(DEVICE)


def main():
    """
    Training and validation.
    """
    global best_bleu4, epochs_since_improvement, start_epoch, checkpoint
    global idx2word, word2idx, idx2cls, cls2idx, idx2attr, attr2idx

    # Training parameters
    # Starting epoch (if we continue training)
    start_epoch = 0
    # keeps track of number of epochs since the last improvement in val. BLEU
    epochs_since_improvement = 0
    # BLEU-4 score right now
    best_bleu4 = 0.
    # path to checkpoint, None if none
    checkpoint = None

    args = parse_args()
    data_dir = os.path.join(args.expdir, "data")

    idx2word, word2idx = load_vocab(
        os.path.join(data_dir,"words_vocab.txt"))
    idx2cls, cls2idx = load_vocab(
        os.path.join(data_dir, "objects_vocab.txt"))
    idx2attr, attr2idx = load_vocab(
        os.path.join(data_dir, "attributes_vocab.txt"))

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=ATTENTION_DIM,
                                       embed_dim=EMB_DIM,
                                       decoder_dim=DECODER_DIM,
                                       vocab_size=len(word2idx),
                                       input_projection=args.input_projection,
                                       dropout=DROPOUT)

        decoder_optimizer = torch.optim.Adamax(
            params=filter(lambda p: p.requires_grad, decoder.parameters()))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
       
    # Move to GPU, if available
    decoder = decoder.to(DEVICE)

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss().to(DEVICE)
    criterion_dis = nn.MultiLabelMarginLoss().to(DEVICE)
    criterion_cl = None
    if args.use_cluster_loss:
        criterion_cl = ClusterLoss(margin=0.5).to(DEVICE)
    criterion_pe = None
    if args.use_perceptual_loss:
        criterion_pe = PerceptualLoss().to(DEVICE)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_dir, DATA_NAME, 'TRAIN'),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_dir, DATA_NAME, 'VAL'),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
        pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, EPOCHS):
        # Decay learning rate if there is no improvement for 8 consecutive
        # epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
    
        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce=criterion_ce,
              criterion_dis=criterion_dis,
              criterion_cl=criterion_cl,
              criterion_pe=criterion_pe,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                decoder=decoder,
                                criterion_ce=criterion_ce,
                                criterion_dis=criterion_dis,
                                criterion_cl=criterion_cl,
                                criterion_pe=criterion_pe)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: {:d}\n".format(
                epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(
            DATA_NAME, args.expdir, epoch, epochs_since_improvement,
            decoder,decoder_optimizer, recent_bleu4, is_best)


def train(train_loader,
          decoder,
          criterion_ce,
          criterion_dis,
          criterion_cl,
          criterion_pe,
          decoder_optimizer,
          epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    # train mode (dropout and batchnorm is used)
    decoder.train()

    # forward prop. + back prop. time
    batch_time = AverageMeter()

    # data loading time
    data_time = AverageMeter()

    # loss (per word decoded)
    losses = AverageMeter()

    # top5 accuracy
    top5accs = AverageMeter()

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, classes, attributes) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(DEVICE)
        caps = caps.to(DEVICE)
        classes = classes.to(DEVICE)
        attributes = attributes.to(DEVICE)
        caplens = caplens.to(DEVICE)

        # Forward prop.
        decoder_output = decoder(imgs, caps, caplens)
        scores = decoder_output[0]
        scores_d = decoder_output[1]
        caps_sorted = decoder_output[2]
        decode_lengths = decoder_output[3]
        sort_ind = decoder_output[4]
        img_features = decoder_output[5]

        classes = classes[sort_ind]
        attributes = attributes[sort_ind]

        indices = reindex_np(classes, idx2cls, word2idx)

        # Max-pooling across predicted words across time steps
        # for discriminative supervision
        scores_d = scores_d.max(1)[0]

        # Since we decoded starting with <start>, the targets
        # are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        targets_d = torch.zeros(scores_d.size(0),scores_d.size(1)).to(DEVICE)
        targets_d.fill_(-1)

        for length in decode_lengths:
            targets_d[:,:length-1] = targets[:,:length-1]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss_d = criterion_dis(scores_d,targets_d.long())
        loss_g = criterion_ce(scores, targets)
        loss_c = 0
        if criterion_cl is not None:
            loss_c = criterion_cl(img_features, indices)
        loss_p = 0
        if criterion_pe is not None:
            loss_p = criterion_pe(
                img_features, decoder.embedding(indices), indices)
        loss = loss_g + (10 * loss_d)
       
        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()
	
        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % PRINT_FREQ == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    top5=top5accs))


def validate(val_loader,
             decoder,
             criterion_ce,
             criterion_dis,
             criterion_cl,
             criterion_pe):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad(): 
        for i, (imgs, caps, caplens, classes, attributes, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(DEVICE)
            caps = caps.to(DEVICE)
            classes = classes.to(DEVICE)
            attributes = attributes.to(DEVICE)
            caplens = caplens.to(DEVICE)

            decoder_output = decoder(imgs, caps, caplens)
            scores = decoder_output[0]
            scores_d = decoder_output[1]
            caps_sorted = decoder_output[2]
            decode_lengths = decoder_output[3]
            sort_ind = decoder_output[4]
            img_features = decoder_output[5]

            classes = classes[sort_ind]
            attributes = attributes[sort_ind]

            indices = reindex_np(classes, idx2cls, word2idx)

            # Max-pooling across predicted words across time steps
            # for discriminative supervision
            scores_d = scores_d.max(1)[0]

            # Since we decoded starting with <start>, the targets are
            # all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            targets_d = torch.zeros(
                scores_d.size(0),scores_d.size(1)).to(DEVICE)
            targets_d.fill_(-1)

            for length in decode_lengths:
                targets_d[:,:length-1] = targets[:,:length-1]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(
                scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss_d = criterion_dis(scores_d,targets_d.long())
            loss_g = criterion_ce(scores, targets)
            loss_c = 0
            if criterion_cl is not None:
                loss_c = criterion_cl(img_features, indices)
            loss_p = 0
            if criterion_pe is not None:
                loss_p = criterion_pe(
                    img_features, decoder.embedding(indices), indices)
            loss = loss_g + (10 * loss_d) + loss_c + loss_p

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % PRINT_FREQ == 0:
                print(
                    'Validation: [{0}/{1}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction)
            # for each image
            # If for n images, we have n hypotheses, and references a, b, c...
            # for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...],
            # hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word2idx['<start>'], word2idx['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                # remove pads
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    bleu4 = round(bleu4,4)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, '
        'BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
