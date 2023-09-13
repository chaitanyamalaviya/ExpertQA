import dataclasses
import json
import os
import sys
from typing import List, Dict, Union
from tqdm import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from pyserini.search import LuceneSearcher, FaissSearcher
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from data_utils import example_utils


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def sphere_search(query: str,
                  searcher: Union[LuceneSearcher, FaissSearcher],
                  top_k: int = 5):
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


def retrieval_gpt_generate(query: str,
                           paragraphs: List[str],
                           prompt_template: str,
                           gpt_model_name: str = "gpt-4"):

    topk_relevant_passages_content = ["Passage ID " + str(i+1) + ": " + doc.replace("\n","") for i, doc in enumerate(paragraphs)]
    cur_prompt = prompt_template.format(context="Context: \n" + "\n".join(topk_relevant_passages_content),
                                        question=query)

    resp = chat_completion_with_backoff(
        model=gpt_model_name,
        messages=[{"role": "user", "content": cur_prompt}],
        max_tokens=2048,
    )

    answer = resp["choices"][0]["message"]["content"]
    return answer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        help="input data to process",
                        type=str)
    parser.add_argument("--index_dir",
                        help="directory to processed sphere index",
                        default="/mnt/nlpgridio3/data/cmalaviya/sphere/faiss_index/sparse",
                        type=str)
    parser.add_argument("--dense_index_model",
                        help="model used for indexing, only applies when using dense index.",
                        default="facebook/dpr-question_encoder-single-nq-base",
                        type=str)
    parser.add_argument("--output_file",
                        help="output filepath",
                        type=str)
    parser.add_argument("--dense",
                        help="enable dense retrieval",
                        action="store_true")
    parser.add_argument("--topk",
                        help="the value of k for the topk passages to use for QA",
                        type=int,
                        default=5)
    parser.add_argument("--gpt_model_name",
                        help="Name of gpt model to use",
                        type=str,
                        default="gpt-4")
    parser.add_argument("--start_idx",
                        help="Start index",
                        type=int,
                        default="0")

    args = parser.parse_args()

    # set PYTHONPATH
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    # Initialize LLM
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY is None:
        raise ValueError("Please set the OpenAI API key as environment variable 'OPENAI_API_KEY'.")

    llm = ChatOpenAI(model_name=args.gpt_model_name, openai_api_key=OPENAI_API_KEY)

    # Prepare prompt
    prompt_template = """Use the following pieces of context to answer the question completely and precisely in up to 500 words. If you don't know the answer, just say "I don't know" and explain why the context is insufficient to answer the question.
    You need to support every statement in the answer with in-line citations to passages given in the the context. The citations should appear as numbers such as [1], [2] that refer to the Passage IDs of the given passages. A statement may need to be supported by multiple references and should then be cited as [1] [2]. (for example, "Paris is the capital of France [1] [2]." where "1" and "2" are the Passage IDs of the first and second passage).

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Initialize retriever + docid to text mapping
    searcher = init_searcher(index_dir=args.index_dir,
                             is_dense=args.dense,
                             dense_index_model=args.dense_index_model)

    # Process the data
    input_data = example_utils.read_examples(args.input_file)
    with open(args.output_file, 'a') as fout:
        print(f"Using {args.gpt_model_name} to generate answers...")
        for example in tqdm(input_data[args.start_idx: args.start_idx+1000]):
            query_text = example.question
            relevant_paras = sphere_search(query_text,
                                           searcher=searcher,
                                           top_k=args.topk)
            relevant_paras_text = [p["text"] for p in relevant_paras]

            answer = retrieval_gpt_generate(query=query_text,
                                            paragraphs=relevant_paras_text,
                                            prompt_template=prompt_template,
                                            gpt_model_name=args.gpt_model_name)

            attributions = []
            for p in relevant_paras:
                attributions.append({
                    'url': p["url"],
                    'text': p["text"]
                })

            example.answers[f"rr_sphere_gpt4"] = example_utils.Answer(answer_string=answer,
                                                                      attribution=attributions)
            example_str = json.dumps(dataclasses.asdict(example))
            fout.write(example_str)
            fout.write("\n")
