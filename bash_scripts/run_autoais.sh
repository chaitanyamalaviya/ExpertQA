#!/bin/bash

TEST_FILE=data/r2_compiled_anon.jsonl

python3 modeling/auto_attribution/autoais.py \
  --input_file ${TEST_FILE} \
  --ais_output_file modeling/auto_attribution/r2_compiled_anon_autoais.jsonl