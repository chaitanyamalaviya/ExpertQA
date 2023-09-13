#!/bin/bash

export OPENAI_API_KEY=""

# gpt4

python3 -u modeling/response_collection/fetch_openai_responses.py \
--input_file data/r1_data.jsonl \
--output_file data/r1_data_answers_gpt4.jsonl \
--model_name gpt4

# bingchat

for i in $(seq 0 5 1); 
do
echo $i;
python3 -u modeling/response_collection/fetch_bingchat_responses.py \
--input_file data/r1_data_bingchat_balanced.jsonl \
--output_file data/r1_data_answers_bingchat_balanced.jsonl \
--start_idx $i \
--limit 5
done


# rr_gs_gpt4

python3 -u modeling/retrieval/retrieve_and_read.py \
--input_file data/r1_data.jsonl \
--output_file data/r1_data_answers_rr_gs_gpt4.jsonl


# post_hoc_gs_gpt4

python3 modeling/response_collection/post_hoc_cite.py \
--input_file data/r1_data_answers_gpt4_claims.jsonl \
--output_file data/r1_data_answers_post_hoc_gs_gpt4.jsonl