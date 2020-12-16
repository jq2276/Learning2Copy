#!/bin/bash

set -e

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# Norm Generation described in [Wu et al. 2019]
TOPIC_GENERALIZATION=1

# set python path according to your actual environment
pythonpath='python'

# the prefix of the file name used by the model, must be consistent with the configuration in network.py
prefix=demo

# data path
datapath=./data

# data (DuConv or DuRecDial)
data=DuConv    # need consistent with run_test.sh

# DATA_TYPE = "train" or "dev"
datatype=(train dev)

# data preprocessing
for ((i=0; i<${#datatype[*]}; i++))
do
    corpus_file=${datapath}/resource/${data}/${datatype[$i]}.txt
    sample_file=${datapath}/resource/${data}/sample.${datatype[$i]}.txt
    text_file=${datapath}/${prefix}.${data}.${datatype[$i]}
    topic_file=${datapath}/${prefix}.${data}.${datatype[$i]}.topic

    # step 1: firstly have to convert session data to sample data
    ${pythonpath} ./util/${data}/convert_session_to_sample.py ${corpus_file} ${sample_file}

    # step 2: convert sample data to text data required by the model
    ${pythonpath} ./util/${data}/convert_conversation_corpus_to_model_text.py ${sample_file} ${text_file} ${topic_file} ${TOPIC_GENERALIZATION}
done

# step 3: in train stage, we just use train.txt and dev.txt, so we copy dev.txt to test.txt for model training
cp ${datapath}/${prefix}.${data}.dev ${datapath}/${prefix}.${data}.test

# step 4: train model, you can find the model file in ./models/ after training

## stage -- 0
${pythonpath} ./network.py --data_prefix demo.${data} --stage 0  --num_epochs 5 --gpu 0 > log.txt
## stage -- 1, note that the num_epoch of stage 1 must > the num_epoch of stage 0
${pythonpath} ./network.py --data_prefix demo.${data} --stage 1  --num_epochs 25 --ckpt ./models/best_0 --gpu 0 > log.txt
