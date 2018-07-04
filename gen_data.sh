#!/bin/bash
USR_DIR=poleval_problem/trainer/
PROBLEM=polish_language_problem
DATA_DIR=$HOME/workspace/polishlanguagemodel/data/
TMP_DIR=$HOME/workspace/polishlanguagemodel/tmp_dir
mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

