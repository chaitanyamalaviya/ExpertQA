"""Evaluation of Attributed QA systems."""
import dataclasses
import json
import os
import re
import sys

from absl import app
from absl import flags
from absl import logging
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Config
from transformers import T5Tokenizer
import torch

from nltk import sent_tokenize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils


AUTOAIS = "google/t5_xxl_true_nli_mixture"

PASSAGE_FORMAT = re.compile("« ([^»]*) » « ([^»]*) » (.*)")

# File locations.
flags.DEFINE_string("input_file", "", "Path to system predictions.")
flags.DEFINE_string("ais_output_file", "", "Path for file to write AIS outputs.")
flags.DEFINE_string("t5_ckpt_path", None, "Path to finetuned T5-xxl TRUE checkpoint.")
FLAGS = flags.FLAGS


def format_passage_for_autoais(passage):
    """Produce the NLI format for a passage.

    Args:
        passage: A passage from the Wikipedia scrape.

    Returns:
        a formatted string, e.g.

        Luke Cage (season 2), Release. The second season of Luke Cage was released
        on June 22, 2018, on the streaming service Netflix worldwide, in Ultra HD
        4K and high dynamic range.
    """
    m = PASSAGE_FORMAT.match(passage)
    if not m:
        return passage

    headings = m.group(2)
    passage = m.group(3)
    return f"{headings}. {passage}"


def format_example_for_autoais(evidence, claim):
    return "premise: {} hypothesis: {}".format(evidence, claim)


def get_entail_label_ids(tokenizer):
    pos_id = tokenizer.convert_tokens_to_ids("1")
    neg_id = tokenizer.convert_tokens_to_ids("0")
    return pos_id, neg_id


def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def _autoais_predict(evidence,
                     claim,
                     model,
                     tokenizer,
                     entail_token_id,
                     non_entail_token_id,
                     cuda: bool = True):
    input_text = format_example_for_autoais(evidence, claim)
    input_ids = tokenizer(input_text,
                          return_tensors="pt",
                          padding="max_length",
                          max_length=512,
                          truncation=True).input_ids

    if cuda:
        input_ids = input_ids.cuda()

    outputs = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
    result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    inference = "Y" if result == "1" else "N"

    # Get softmax score between "1" and "0"
    tok_scores = outputs.scores[0][0].detach().cpu()
    pos_score = tok_scores[entail_token_id].numpy()
    neg_score = tok_scores[non_entail_token_id].numpy()
    label_scores = _softmax([pos_score, neg_score])

    return inference, label_scores


def stretch_nli_rerank(example,
                       tokenizer,
                       model,
                       entail_token_id: int,
                       non_entail_token_id: int,
                       k: int = 2
                       ):
    """Slice evidence into smaller slices and re-rank the slices based on model confidence.

    See https://arxiv.org/abs/2204.07447 for details.
    Since the AutoAIS model doesn't distinguish between contradiction and netural,

    """

    # Slice evidence into smaller chunks
    claim_text = example["claim"]
    evidence_text = example["evidence"]
    sents = sent_tokenize(evidence_text)

    per_sent_score = []
    for sent in sents:
        input_text = format_example_for_autoais(sent, claim_text)
        label, score = _autoais_predict(evidence=input_text,
                                        claim=claim_text,
                                        model=model,
                                        tokenizer=tokenizer,
                                        entail_token_id=entail_token_id,
                                        non_entail_token_id=non_entail_token_id)
        per_sent_score.append(score[1]) # order by neutral prob, so that argsort by ascending order will give us
                                        # evidence sentences from most-to-least entailed


    # Take top k entailment sentences + top k neutral sentences, run through the autoais model again
    high_to_low = np.argsort(per_sent_score)
    reranked_indices = np.concatenate((high_to_low[:k], high_to_low[-k:]), axis=None)
    reranked_input_sents = [sents[idx] for idx in reranked_indices]

    reranked_input = " ".join(reranked_input_sents)
    label, score = _autoais_predict(evidence=reranked_input,
                                    claim=claim_text,
                                    model=model,
                                    tokenizer=tokenizer,
                                    entail_token_id=entail_token_id,
                                    non_entail_token_id=non_entail_token_id)

    return label


def infer_autoais(example,
                  tokenizer,
                  model,
                  max_len:int = 512):
    """Runs inference for assessing AIS between a premise and hypothesis.

    Args:
        example: Dict with the example data.
        tokenizer: A huggingface tokenizer object.
        model: A huggingface model object.

    Returns:
        A string representing the model prediction.
    """
    input_text = format_example_for_autoais(example["evidence"], example["claim"])
    pos_id, neg_id = get_entail_label_ids(tokenizer)

    # Try tokenize the input text, see if it exceeds the token limit
    tokens = tokenizer.encode(input_text)
    if len(tokens) > max_len:
        inference = stretch_nli_rerank(
            example=example,
            tokenizer=tokenizer,
            model=model,
            entail_token_id=pos_id,
            non_entail_token_id=neg_id,
            k=2
        )
    else:
        inference, _ = _autoais_predict(
            evidence=example["evidence"],
            claim=example["claim"],
            model=model,
            tokenizer=tokenizer,
            entail_token_id=pos_id,
            non_entail_token_id=neg_id
        )

    return inference


def remove_urls(input_string):
    # Regular expression to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Replace URLs with an empty string
    cleaned_string = re.sub(url_pattern, '', input_string)
    return cleaned_string


def cleanup_claim_and_evidence(raw_text):
    """
    Remove square brackets and urls in claim and evidence
    """
    # remove square brackets in the string
    cleaned_str = re.sub(r'\[\d\]\s?', '', raw_text)

    # remove urls in evidence
    parts = cleaned_str.split("\n\n")
    if parts[0].startswith("http"):
        cleaned_str = " ".join(parts[1:])
    else:
        cleaned_str = " ".join(parts)

    return cleaned_str


def score_predictions(input_data,
                      output_path,
                      t5_ckpt_path: str = None):
    """Scores model predictions using AutoAIS.

    Args:
        input_data: list of input example dicts with claim and evidence keys

    Returns:
        input data examples along with autoais_prediction for each example
    """
    hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS, cache_dir="/nlp/data/huggingface_cache/")

    if t5_ckpt_path is None:
        hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS, device_map='auto',
                                                              cache_dir="/nlp/data/huggingface_cache/")
    else:
        cfg = T5Config.from_pretrained(AUTOAIS)
        with init_empty_weights():
            hf_model = T5ForConditionalGeneration._from_config(cfg)

        hf_model = load_checkpoint_and_dispatch(hf_model, t5_ckpt_path, device_map='auto')

    with open(output_path, "w") as fout:
        for i, ex in enumerate(tqdm(input_data)):
            for system, answer in ex.answers.items():
                for c in answer["claims"]:
                    claim_string = c["claim_string"]
                    claim_string = cleanup_claim_and_evidence(claim_string)
                    if not c["evidence"] or len(c["evidence"]) == 0:
                        print(f"empty evidence for {claim_string}!")
                        c["autoais_label"] = "N"
                        continue
                    evidence = [cleanup_claim_and_evidence(e) for e in c["evidence"]]
                    evidence_joined = " ".join(evidence)

                    input_ex = {
                        "claim": claim_string,
                        "evidence": evidence_joined
                    }
                    autoais_label = infer_autoais(input_ex, hf_tokenizer, hf_model)
                    c["autoais_label"] = autoais_label

            ex_json = dataclasses.asdict(ex)
            ex_str = json.dumps(ex_json)
            fout.write(ex_str)
            fout.write("\n")


def main(unused_argv):
    input_data = example_utils.read_examples(FLAGS.input_file)

    logging.info("Scoring and writing predictions...")
    score_predictions(input_data, FLAGS.ais_output_file, FLAGS.t5_ckpt_path)


if __name__ == "__main__":
  app.run(main)