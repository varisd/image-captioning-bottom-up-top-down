#!/bin/bash

expdir=$1
SLEEP=${2:-120}
for f in $expdir/BEST*; do
    out_eval=`echo $f | sed 's#/BEST#/eval/BEST#'`.out
    out_emb=`echo $f | sed 's#/BEST#/eval/BEST#'`.emb.txt
    out_hyps=`echo $f | sed 's#/BEST#/eval/BEST#'`.hyp.txt
    [ -e $out_eval ] || qsubmit \
        --jobname=top-down-eval \
        --logdir=logs \
        --mem=15g \
        --cores=4 \
        --gpumem=11g \
        --gpus=1 "source ~/python-virtualenv/python-2.7/bin/activate && python eval.py --data_dir data/karpathy_genome_with_labels_intersection --checkpoint $f --caption_output_file $out_hyps > $out_eval" &
    [ -e $out_emb ] || qsubmit \
        --jobname=top-down-analyze \
        --logdir=logs \
        --mem=60g \
        --cores=4 "source ~/python-virtualenv/python-2.7/bin/activate && python scripts/analyze_model.py --data_dir data/karpathy_genome_with_labels_intersection --checkpoint $f > $out_emb" &
    sleep $SLEEP
done
