# <img src="images/steth.png" alt="drawing" width="20"/> ExpertQA <img src="images/gavel.png" alt="drawing" width="20"/>: Expert-Curated Questions and Attributed Answers

# Paper
Find the paper at [https://arxiv.org/abs/2309.07852](https://arxiv.org/abs/2309.07852)

# Dataset

ExpertQA contains 2177 examples, which are validated on various axes of factuality and attribution. The main data can be found at 
* `data/r2_compiled_anon.jsonl`

This can be loaded simply using the data loaders at `data_utils` as:

```data = example_utils.read_examples("data/r2_compiled_anon.jsonl")```

The file contains newline-separated json dictionaries with the following fields:
* `question` - Question written by an expert.
* `annotator_id` - Anonymized annotator ID of the author of the question.
* `answers` - Dict mapping model names to an Answer object. The model names can be one of `{gpt4, bing_chat, rr_sphere_gpt4, rr_gs_gpt4, post_hoc_sphere_gpt4, post_hoc_gs_gpt4}`. 
* `metadata` - A dictionary with the following fields:
    * `question_type` - The question type(s) separated by "|".
    * `field` - The field to which the annotator belonged.
    * `specific_field` - More specific field name within the broader field.

Each Answer object contains the following fields:
* `answer_string`: The answer string.
* `attribution`: List of evidences for the answer (not linked to specific claims). Note that these are only URLs, the evidence passages are stored in the Claim object -- see below.
* `claims`: List of Claim objects for the answer.
* `revised_answer_string`: Revised answer by annotator.
* `usefulness`: Usefulness of original answer marked by annotator.
* `annotation_time`: Time taken for annotating this answer.
* `annotator_id`: Anonymized annotator ID of the person who validated this answer.

Each Claim object contains the following fields:
* `claim_string`: Original claim string.
* `evidence`: List of evidences for the claim (URL+passage or URL).
* `support`: Attribution marked by annotator.
* `reason_missing_support`: Reason for missing support specified by annotator.
* `informativeness`: Informativeness of claim for the question, marked by annotator.
* `worthiness`: Worthiness of citing claim marked by annotator.
* `correctness`: Factual correctness of claim marked by annotator.
* `reliability`: Reliability of source evidence marked by annotator.
* `revised_claim`: Revised claim by annotator.
* `revised_evidence`: Revised evidence by annotator.
* `atomic_claims`: Atomic claims for Fact score estimation.
* `atomic_evidences`: Atomic claim-evidences for Fact score estimation.
* `fact_score`: Fact score for each claim.
* `autoais_label`: Autoais label for the original claim and original evidence.

## Additional Files

* We also provide the list of questions (2507 in total) collected in stage 1 of our annotation. These can be found at `data/r1_data_anon.jsonl`.
* Answers were sampled from different systems for the purpose of annotation. Files containing all answers from a specific system can be found at `data/r1_data_answers_{MODEL_KEY}_claims_anon.jsonl`.
* In the main dataset, evidences for each claim can be URL+passages OR only URLs, depending on which system the answer was sampled from. We provide all passage evidences in the file `data/r2_compiled_all_evidences_autoais_anon.jsonl`.

## Long-form QA

The random and domain split for the long-form QA dataset can be found at `data/lfqa/`. The files for the random split are prefixed with `rand_lfqa_` and the files for the domain split are prefixed with `domain_lfqa_`.

# Modeling

## Response collection

Found at `modeling/response_collection`. The scripts for collecting responses from different systems are at:
* `bing_chat`: `fetch_bingchat_responses.py`
* `gpt4`: `fetch_openai_responses.py`
* `rr_gs_gpt4`: `retrieve_and_read.py`
* `rr_sphere_gpt4`: `sphere_and_read.py`
* `post_hoc_gs_gpt4`: `post_hoc_cite.py`
* `post_hoc_sphere_gpt4`: `post_hoc_cite_sphere.py`

## Attribution estimation

Found at `modeling/auto_attribution`.
* First, the script `convert_for_autoais.py` may be used to fetch textual evidences when URLs are returned as attributions.
* The script `autoais.py` can then be used to generate autoAIS predictions using the TRUE model.
* The evaluation scripts `compute_autoais_score.py` and `compute_human_correlation.py` compute averaged autoAIS scores, and correlations with reference judgements of attribution in our dataset.
* To finetune the TRUE model on the domain split of our dataset, use the script `finetune_autoais.py`.

## Factuality estimation

Found at `modeling/fact_score`. See sample usage at `get_fact_score.sh`.
* First, we need to break down the claims in the dataset to atomic claims. This can be done with the script `break_down_to_atomic_claims.py`.
* Next, we need to retrieve evidence for all atomic claims. This can be done using `retrieve_evidence_for_claims.py`, which retrieves the top-5 passages from the top-10 Google search results, with each atomic claim as the query.
* Finally, we compute the scores using `factscore.py`, which prompts ChatGPT for the factuality of each atomic claim.
* The claim-level factuality scores and the averaged F1 scores can then be computed using `compute_factuality_f1.py`.

## Long-form QA

Found at `modeling/lfqa`. Example usages at `bash_scripts/run_lfqa.sh`.
* The script `convert_for_lfqa.py` converts the split data into a format required for long-form QA training.
* For finetuning FlanT5-XXL, use `run_gen_qa.py`.
* For finetuning Llama-2-7B and Vicuna-7B, use `run_sft_qa.py`.

# Evaluation

Scripts and documentation for running evaluation are in the `eval/` directory.
* `nlg_eval.py` computes ROUGE and QAFactEval scores. Download the QAFactEval models from `https://github.com/salesforce/QAFactEval` and place them at `qafacteval_models`.

# License
This project is licensed under the MIT License - see the LICENSE file for details

# Citation
```
@inproceedings{malaviya23expertqa,
    title = {ExpertQA: Expert-Curated Questions and Attributed Answers},
    author = {Chaitanya Malaviya and Subin Lee and Sihao Chen and Elizabeth Sieber and Mark Yatskar and Dan Roth},
    booktitle = {arXiv},
    month = {September},
    year = {2023},
    url = "https://arxiv.org/abs/2309.07852"
}
```
