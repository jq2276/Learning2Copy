#!/bin/bash

set -e

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# generalizes target_a/target_b of goal for all outputs, replaces them with slot mark
TOPIC_GENERALIZATION=1

# set python path according to your actual environment
pythonpath='python'

# the prefix of the file name used by the model, must be consistent with the configuration in network.py
prefix=demo

# put all data set that used and generated for testing under this folder: datapath
# for more details, please refer to the following data processing instructions
datapath=./data

# data (DuConv or DuRecDial)
data=DuConv    # need consistent with run_train.sh

# test or dev
datapart=test


corpus_file=${datapath}/resource/${data}/${datapart}.txt
sample_file=${datapath}/resource/${data}/sample.${datapart}.txt
text_file=${datapath}/${prefix}.${data}.test
topic_file=${datapath}/${prefix}.${data}.test.topic

# step 1: if eval dev.txt, firstly have to convert session data to sample data
# if eval test.txt, we can use test.txt provided by the organizer directly.
${pythonpath} ./util/${data}/convert_session_to_sample.py ${corpus_file} ${sample_file}

# step 2: convert sample data to text data required by the model
${pythonpath} ./util/${data}/convert_conversation_corpus_to_model_text.py ${sample_file} ${text_file} ${topic_file} ${TOPIC_GENERALIZATION}

# step 3: predict by model
${pythonpath} ./network.py --test --data_prefix demo.${data} --ckpt models/best_1.model --gen_file ./output/test.${data}.result --use_posterior False --gpu 0 > log.txt 2>&1

# step 4: replace slot mark generated during topic generalization with real text
${pythonpath} ./util/topic_materialization.py ./output/test.${data}.result ./output/test.${data}.result.final ${topic_file}

# step 5: if you eval dev.txt, you can run the following command to get result
# if you eval test.txt, you can upload the ./output/test.result.final to the competition website to get result
# if [ "${datapart}"x = "dev"x ]; then
${pythonpath} ./evaluation/convert_result_for_eval.py ${sample_file} ./output/test.${data}.result.final ./output/test.${data}.result.eval
${pythonpath} ./evaluation/eval.py ./output/test.${data}.result.eval
# fi

