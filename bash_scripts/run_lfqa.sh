#!/bin/bash

TRAIN_FILE="data/expertqa/domain_lfqa_train.json"
VAL_FILE="data/expertqa/domain_lfqa_val.json"
TEST_FILE="data/expertqa/domain_lfqa_test.json"

# meta-llama/Llama-2-7b-hf
# google/flan-t5-xxl
# lmsys/vicuna-7b-v1.5

# FlanT5 training
# OUTPUT_DIR="saved_models/domain_expertqa_flant5_xxl"
deepspeed --num_gpus=8 --master_port=$RANDOM modeling/lfqa/run_gen_qa.py \
--model_name_or_path google/flan-t5-xxl \
--train_file ${TRAIN_FILE} \
--validation_file ${TEST_FILE} \
--context_column context \
--question_column question \
--answer_column answer \
--do_eval \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 16 \
--learning_rate 1e-4 \
--num_train_epochs 3 \
--max_seq_length 512 \
--eval_accumulation_steps 100 \
--predict_with_generate \
--save_strategy "no" \
--save_total_limit 1 \
--gradient_checkpointing True \
--overwrite_cache True \
--report_to wandb \
--logging_steps 1 \
--output_dir ${OUTPUT_DIR} \
--deepspeed modeling/lfqa/ds_configs/ds_new_config.json

# Llama2 / Vicuna training
OUTPUT_DIR="saved_models/domain_expertqa_vicuna_7b_bs4"
python modeling/lfqa/run_sft_qa.py \
    --model_name lmsys/vicuna-7b-v1.5 \
    --train_file ${TRAIN_FILE} \
    --validation_file ${TEST_FILE} \
    --do_train True \
    --do_eval False \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir ${OUTPUT_DIR} \
    --seq_length 2048 \
    --learning_rate_scheduler constant \
    --learning_rate 2e-4 \
    --load_in_4bit
