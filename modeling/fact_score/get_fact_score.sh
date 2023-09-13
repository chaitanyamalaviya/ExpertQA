#!/bin/bash

export TRANSFORMERS_CACHE=""
export HF_DATASETS_CACHE=""
export OPENAI_API_KEY=""
export CUSTOM_SEARCH_API_KEY=""
export CUSTOM_SEARCH_CX=""

python3 modeling/fact_score/break_down_to_atomic_claims.py \
    --input_file data/r2_compiled_out_corrected_revised.jsonl \
    --output_file data/r2_compiled_out_corrected_revised_atomic.jsonl

python3 modeling/fact_score/retrieve_evidence_for_claims.py \
    --start_idx 0 \
    --input_file data/r2_compiled_out_corrected_revised_atomic.jsonl \
    --output_file data/r2_compiled_out_corrected_revised_atomic_w_evidences_0.jsonl

python3 modeling/fact_score/factscore.py \
    --input_file data/r2_compiled_out_corrected_revised_atomic_w_evidences_full.jsonl \
    --output_file data/r2_compiled_out_corrected_revised_atomic_w_evidences_factscores.jsonl
