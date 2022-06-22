#!/bin/bash

expdir=$1
echo Epoch CIDEr Bleu_4 Bleu_3 Bleu_2 Bleu_1 ROUGE_L METEOR
for n in `seq 0 1 50`; do
    echo $n `cat $expdir/eval/BEST_${n}[^0-9]* | tr -d "{}" | sed "s/,[^:]*://g;s/'CIDEr': //"`; done
