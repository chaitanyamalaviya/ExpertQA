#!/bin/bash

cd /mnt/nlpgridio3/data/cmalaviya/expert-attribs

export PYTHONPATH=$(pwd):PYTHONPATH
export OPENAI_API_KEY=""

python -u modeling/retrieval/sphere_and_read.py \
--index_dir /mnt/nlpgridio3/data/cmalaviya/sphere/faiss_index/sparse \
--input_file data/r1_data.jsonl \
--output_file data/r1_data_answers_rr_sphere_gpt4_bm25.jsonl \
--gpt_model_name gpt-4