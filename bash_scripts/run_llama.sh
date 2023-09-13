#!/bin/bash

TRAIN_FILE="data/expertqa/rand_lfqa_train.json"
VAL_FILE="data/expertqa/rand_lfqa_val.json"
TEST_FILE="data/expertqa/rand_lfqa_test.json"
OUTPUT_DIR="saved_models/rand_expertqa_llama2_13b_chat_bs4"
# meta-llama/Llama-2-7b-chat-hf
# google/flan-t5-xxl


python modeling/lfqa/run_sft_qa.py \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --train_file ${TRAIN_FILE} \
    --validation_file ${TEST_FILE} \
    --do_train False \
    --do_eval True \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir ${OUTPUT_DIR} \
    --seq_length 2048 \
    --learning_rate_scheduler constant \
    --learning_rate 2e-4
    # --load_in_4bit
 