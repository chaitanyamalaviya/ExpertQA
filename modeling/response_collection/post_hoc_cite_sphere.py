import dataclasses
import json
import os
import re
import sys
from typing import List, Dict, Union

from pyserini.search import LuceneSearcher, FaissSearcher

import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="input filepath", type=str)
parser.add_argument("--output_file", help="output filepath", type=str)
parser.add_argument("--ret_type", help="retriever type for wiki, dense or bm25",
                    type=str,
                    default="bm25",
                    choices=["dense", "bm25"])
parser.add_argument("--index_dir",
                    help="directory to processed wikipedia index",
                    default="/mnt/nlpgridio3/data/cmalaviya/sphere/faiss_index/sparse",
                    type=str)
parser.add_argument("--dense_index_model",
                    help="model used for indexing, only applies when using dense index.",
                    default="facebook/dpr-question_encoder-single-nq-base",
                    type=str)
parser.add_argument("--start_idx", help="start index", type=int, default=0)
parser.add_argument("--num_examples", help="Number of examples to process", type=int, default=200)
parser.add_argument("--topk", help="the value of k for the topk passages to use for QA", type=int, default=1)

args = parser.parse_args()


def sphere_search(query: str,
                  searcher: Union[LuceneSearcher, FaissSearcher],
                  top_k: int = 1):
    hits = searcher.search(query, k=top_k)
    retrieved = []
    for hit in hits:
        para_text = hit.raw
        retrieved.append({
            "url": json.loads(hit.docid)["url"],
            "text": para_text
        })

    return retrieved


def init_searcher(index_dir: str,
                  is_dense: bool = False,
                  dense_index_model: str = None):
    """
    Initialize the searcher
    """
    if is_dense:
        assert dense_index_model is not None, "with dense model, --dense_index_model needs to be set to the model that were used to create the index"
        print(f"Initializing the dense searcher from {index_dir} (model {dense_index_model})...")
        searcher = FaissSearcher(index_dir, dense_index_model)
    else:
        print(f"Initializing the BM25 searcher from {index_dir}...")
        searcher = LuceneSearcher(index_dir)

    return searcher


def remove_brackets(input_string):
    # Pattern to match substrings in square brackets
    pattern = r'\[.*?\]'
    # Substitute matching substrings with empty string
    output_string = re.sub(pattern, '', input_string)
    return output_string


if __name__ == '__main__':

    input_data = example_utils.read_examples(args.input_file)
    f = open(args.output_file, "w")

    if args.ret_type == "dense":
        searcher = init_searcher(index_dir=args.index_dir,
                                 is_dense=True,
                                 dense_index_model=args.dense_index_model)
    elif args.ret_type == "bm25":
        searcher = init_searcher(index_dir=args.index_dir,
                                 is_dense=False)
    else:
        raise ValueError(f"Retrieval type {args.wiki_ret_type} not supported.")

    for example in input_data[args.start_idx: args.start_idx + args.num_examples]:
        query_text = example.question
        post_hoc_claims = []
        for claim in example.answers["gpt4"]["claims"]:
            posthoc_attributions = sphere_search(
                query=claim["claim_string"],
                searcher=searcher,
                top_k=args.topk
            )

            post_hoc_claims.append(
                example_utils.Claim(claim_string=claim["claim_string"],
                                    evidence=posthoc_attributions)
            )

        example.answers[f"gpt4_post_hoc_sphere_{args.ret_type}"] = example.answers["gpt4"]
        example.answers[f"gpt4_post_hoc_sphere_{args.ret_type}"]["claims"] = post_hoc_claims
        example.answers[f"gpt4_post_hoc_sphere_{args.ret_type}"]["answer_string"] = remove_brackets(example.answers["gpt4"]["answer_string"])
        example.answers[f"gpt4_post_hoc_sphere_{args.ret_type}"]["attribution"] = []
        ex_json = json.dumps(dataclasses.asdict(example))
        f.write(ex_json)
        f.write("\n")
