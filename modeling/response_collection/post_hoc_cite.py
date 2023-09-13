import dataclasses
import json
import numpy as np
import os
import re
import requests
import sys
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

from newspaper import Article
from newspaper.article import ArticleException

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from bs4 import BeautifulSoup # Required to parse HTML
from urllib.parse import unquote # Required to unquote URLs
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="input filepath", type=str)
parser.add_argument("--output_file", help="output filepath", type=str)
parser.add_argument("--start_idx", help="start index", type=int, default=0)
parser.add_argument("--topk", help="the value of k for the topk passages to use for QA", type=int, default=1)

args = parser.parse_args()


text_splitter = CharacterTextSplitter(separator=' ', chunk_size=1000, chunk_overlap=200)

#####
# OPENAI
#####
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError("Please set the OpenAI API key as environment variable 'OPENAI_API_KEY'.")

openai.organization = ""
openai.api_key = ""

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def google_search(
        query: str,
):
    """Get top 10 webpages from Google search.

    Args:
        query: search query
    Returns:
        top-10 search results in json format
    """
    response = requests.get(f"https://www.google.com/search?q={query}") # Make the request
    soup = BeautifulSoup(response.text, "html.parser") # Parse the HTML
    links = soup.find_all("a") # Find all the links in the HTML

    urls = []
    for l in [link for link in links if link["href"].startswith("/url?q=")]:
        # get the url
        url = l["href"]
        # remove the "/url?q=" part
        url = url.replace("/url?q=", "")
        # remove the part after the "&sa=..."
        url = unquote(url.split("&sa=")[0])
        # special case for google scholar
        if url.startswith("https://scholar.google.com/scholar_url?url=http"):
            url = url.replace("https://scholar.google.com/scholar_url?url=", "").split("&")[0]
        elif 'google.com/' in url: # skip google links
            continue
        if url.endswith('.pdf'): # skip pdf links
            continue
        if '#' in url: # remove anchors (e.g. wikipedia.com/bob#history and wikipedia.com/bob#genetics are the same page)
            url = url.split('#')[0]
        # print the url
        urls.append(url)

    # Use numpy to dedupe the list of urls after removing anchors
    urls = list(np.unique(urls))
    return urls


def scrape_and_parse(url: str):
    """Scrape a webpage and parse it into a Document object"""
    a = Article(url)
    try:
        a.download()
        a.parse()
    except Exception as e:
        return None

    return {
        "url": url,
        "text": a.text,
    }


def scrape_and_filter(urls: list):
    doc_list = []
    for u in urls:
        print(f"Processing: {u}")
        doc = scrape_and_parse(u)
        if doc is None:
            continue
        elif "Access" in doc["text"] and "Denied" in doc["text"]:
            continue
        else:
            doc_list.append(doc)

    return doc_list


def rerank(claim: str,
           retrieved_documents: List[Document],
           topk: int):

    texts = text_splitter.split_documents(retrieved_documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, max_retries=1000)
    while True:
        try:
            docsearch = Chroma.from_documents(texts, embeddings)
        except Exception as e:
            continue
        break
    doc_retriever = docsearch.as_retriever(search_kwargs={"k": topk})
    topk_relevant_passages = doc_retriever.get_relevant_documents(claim)
    ret_passages = [{"text": d.page_content, "url": d.metadata["source"]} for d in topk_relevant_passages]
    return ret_passages


def remove_brackets(input_string):
    # Pattern to match substrings in square brackets
    pattern = r'\[.*?\]'
    # Substitute matching substrings with empty string
    output_string = re.sub(pattern, '', input_string)
    return output_string


if __name__ == '__main__':

    input_data = example_utils.read_examples(args.input_file)
    f = open(args.output_file, "a")
    for example in input_data[args.start_idx: args.start_idx + 200]:
        query_text = example.question
        post_hoc_claims = []
        for claim in example.answers["gpt4"]["claims"]:
            search_results = google_search(claim["claim_string"])

            if len(search_results) > 0:
                all_docs = scrape_and_filter(search_results)
            else:
                all_docs = []
                post_hoc_claims.append(example_utils.Claim(claim_string=claim["claim_string"], evidence=[]))
                continue                

            all_docs = [Document(page_content=d['text'], metadata={'source': d['url']}) for d in all_docs]
            all_docs_content_lens = [len(doc.page_content.strip()) for doc in all_docs]
            if not all_docs or not sum(all_docs_content_lens):
                post_hoc_claims.append(example_utils.Claim(claim_string=claim["claim_string"], evidence=[]))
                continue

            post_hoc_attributions = rerank(claim["claim_string"], all_docs, args.topk)
            post_hoc_claims.append(example_utils.Claim(claim_string=claim["claim_string"], evidence=post_hoc_attributions))
        example.answers["post_hoc_gs_gpt4"] = example.answers["gpt4"]
        example.answers["post_hoc_gs_gpt4"]["claims"] = post_hoc_claims
        example.answers["post_hoc_gs_gpt4"]["answer_string"] = remove_brackets(example.answers["gpt4"]["answer_string"])
        example.answers["post_hoc_gs_gpt4"]["attribution"] = []
        json.dump(dataclasses.asdict(example), f)
        f.write("\n")
