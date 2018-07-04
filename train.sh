#!/bin/bash
USR_DIR=poleval_problem/trainer/
PROBLEM=polish_language_problem
DATA_DIR=$HOME/workspace/polishlanguagemodel/data/
TMP_DIR=$HOME/workspace/polishlanguagemodel/tmp_dir
OUTDIR=$HOME/workspace/polishlanguagemodel/output
mkdir -p $DATA_DIR $TMP_DIR $OUTDIR

t2t-trainer \
 --data_dir=$DATA_DIR \
 --t2t_usr_dir=$USR_DIR \
 --problem=$PROBLEM \
 --model=transformer \
 --hparams_set=transformer_polish_language_poleval \
 --output_dir=$OUTDIR --job-dir=$OUTDIR --train_steps=10
