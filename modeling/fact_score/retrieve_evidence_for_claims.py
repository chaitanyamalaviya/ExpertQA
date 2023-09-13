import numpy as np
import os
import requests
import sys
from tqdm import tqdm
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain import VectorDBQA
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from newspaper import Article
from newspaper.article import ArticleException

import openai

from bs4 import BeautifulSoup # Required to parse HTML
from urllib.parse import unquote # Required to unquote URLs


import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="input filepath", type=str)
parser.add_argument("--output_file", help="output filepath", type=str)
parser.add_argument("--topk", help="the value of k for the topk passages to use for QA", type=int, default=5)
parser.add_argument("--start_idx", type=int, default=0)

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
#####
# Google Search
#####

def google_search(
        query: str,
        api_key: str = None,
        cx: str = None
):
    """Get top 10 webpages from Google search.

    Args:
        query: search query
        api_key: custom search engine api key
        cx: programmable search engine id
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


def retrieve_topk_passages(query: str,
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
    topk_relevant_passages = doc_retriever.get_relevant_documents(query)
    topk_relevant_passages_content = ["Passage ID " + str(i+1) + ": " + doc.page_content.replace("\n","") for i, doc in enumerate(topk_relevant_passages)]

    return "\n\n".join(topk_relevant_passages_content)


if __name__ == '__main__':

    input_data = example_utils.read_examples(args.input_file)
    start_idx = args.start_idx

    for example in tqdm(input_data[start_idx: start_idx+500]):
        claims = example.answers[list(example.answers.keys())[0]]["claims"]       
        for claim in claims:
            atomic_evidences = []
            for i, a_claim in enumerate(claim["atomic_claims"]):
                # Google search for each atomic claim 
                search_results = google_search(a_claim)
                # search_urls = [r['link'] for r in search_results]
                if len(search_results) > 0:
                    all_docs = scrape_and_filter(search_results)
                else:
                    atomic_evidences.append('')
                    continue
                all_docs = [Document(page_content=d['text'], metadata={'source': d['url']}) for d in all_docs]
                all_docs_content_lens = [len(doc.page_content.strip()) for doc in all_docs]
                if not all_docs or not sum(all_docs_content_lens):
                    atomic_evidences.append('')
                    continue
                relevant_passages = retrieve_topk_passages(a_claim, all_docs, args.topk)
                atomic_evidences.append(relevant_passages)
            claim["atomic_evidences"] = atomic_evidences

        example_utils.write_examples(args.output_file, [example], True)
