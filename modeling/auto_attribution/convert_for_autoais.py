'''Script to convert examples for evaluating with AutoAIS'''

from absl import app
from absl import flags
from absl import logging
import os
import re
import sys
from newspaper import Article
from newspaper.article import ArticleException
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import jsonl_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'data/r1_data_answers_bingchat_balanced_final_claims.jsonl', 'Input filepath.')
flags.DEFINE_string('output_file', 'data/r1_data_answers_bingchat_balanced_final_claims_autoais.json', 'Output filepath.')


def scrape_and_parse(url: str):
    """Scrape a webpage and parse it into a Document object"""
    a = Article(url)
    try:
        a.download()
        a.parse()
    except ArticleException:
        return None

    return {
        "url": url,
        "text": a.text,
    }


def main(unused_argv):
    input_data = jsonl_utils.read_jsonl(FLAGS.input_file)
    corrected_data = []
    evidence_tok_len = {}
    processed_claim_counts = 0
    for i, ex in enumerate(tqdm(input_data)):

        for answer_model_name, answer in ex["answers"].items():
            # Check if the system is bingchat or gpt4
            # Only scrape evidence for the two systems
            if answer_model_name not in ["bing_chat", "gpt4"]:
                continue

            if answer_model_name not in evidence_tok_len:
                evidence_tok_len[answer_model_name] = []

            for claim in answer["claims"]:
                if "evidence" not in claim or len(claim["evidence"]) == 0:
                    continue
                urls = [re.search("(?P<url>https?://[^\s]+)", ev).group("url") for ev in claim["evidence"]]
                evidences = [scrape_and_parse(url) for url in urls]
                if evidences is None or len(evidences) == 0:
                    continue

                evidence_text = [ev["text"] for ev in evidences if ev is not None and "text" in ev and ev["text"]]
                claim["evidence"] = evidence_text
                evidence_tok_len[answer_model_name].append(sum([len(ev.split()) for ev in evidence_text]))
                processed_claim_counts += 1

        corrected_data.append(ex)

    logging.info("Crawled evidence for %d claims...", processed_claim_counts)
    jsonl_utils.write_jsonl(FLAGS.output_file, corrected_data)

    # log the evidence token counts for each system
    for answer_model_name, counts in evidence_tok_len.items():
        logging.info("Avg. evidence token counts for %s: %f", answer_model_name, float(np.mean(counts)))


if __name__ == "__main__":
    app.run(main)