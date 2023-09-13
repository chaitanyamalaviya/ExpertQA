import dataclasses
import json
import numpy as np
import os
import requests
import sys
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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from bs4 import BeautifulSoup # Required to parse HTML
from urllib.parse import unquote # Required to unquote URLs


import argparse

prompt_template = """Use the following pieces of context to answer the question completely and precisely in up to 500 words. If you don't know the answer, just say "I don't know" and explain why the context is insufficient to answer the question.
You need to support every statement in the answer with in-line citations to passages given in the the context. The citations should appear as numbers such as [1], [2] that refer to the Passage IDs of the given passages. A statement may need to be supported by multiple references and should then be cited as [1] [2]. (for example, "Paris is the capital of France [1] [2]." where "1" and "2" are the Passage IDs of the first and second passage).

{context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="input filepath", type=str)
parser.add_argument("--output_file", help="output filepath", type=str)
parser.add_argument("--topk", help="the value of k for the topk passages to use for QA", type=int, default=5)

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

llm = ChatOpenAI(model_name='gpt-4', openai_api_key=OPENAI_API_KEY)

#####
# Google Search
#####
CSE_URL = "https://www.googleapis.com/customsearch/v1"
CSE_URL = "https://cse.google.com/cse?cx=74509b47ac2e54393"
gs_api_key = os.getenv('CUSTOM_SEARCH_API_KEY')
pse_cx = os.getenv('CUSTOM_SEARCH_CX')
if gs_api_key is None:
    raise ValueError("Please set the Custom search API key as environment variable 'CUSTOM_SEARCH_API_KEY'.")

if pse_cx is None:
    raise ValueError("Please set the Programmable search engine ID as environment variable 'CUSTOM_SEARCH_CX'.")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


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


    # if api_key is None:
    #     api_key = gs_api_key

    # if cx is None:
    #     cx = pse_cx

    # res = requests.get(
    #     url=CSE_URL,
    #     params={
    #         "q": query,
    #         "key": api_key,
    #         "cx": cx,
    #     },
    # )
    # if res.status_code != 200:
    #     print(f"Google search error: {res.status_code}")
    #     return []

    # res = res.json()

    # if 'items' in res:
    #     return res['items']
    # else:
    #     return []


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


def retrieval_gpt_generate(query: str,
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
    ret_passages = [{"text": p, "url": d.metadata["source"]} for p, d in zip(topk_relevant_passages_content, topk_relevant_passages)]
    cur_prompt = prompt_template.format(context="Context: \n" + "\n\n".join(topk_relevant_passages_content), question=query)
    resp = chat_completion_with_backoff(
        model="gpt-4",
        messages=[{"role": "user", "content": cur_prompt}],
        max_tokens=2048,
    )
    answer = resp["choices"][0]["message"]["content"]
    # chain_type_kwargs = {"prompt": PROMPT}
    # qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": topk}), return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    # result = qa({"query": query})

    return answer, ret_passages


if __name__ == '__main__':

    input_data = example_utils.read_examples(args.input_file)

    f = open(args.output_file, "a")
    for example in input_data:
        query_text = example.question

        # Google search first
        search_results = google_search(query_text)
        # search_urls = [r['link'] for r in search_results]
        if len(search_results) > 0:
            all_docs = scrape_and_filter(search_results)
        else:
            all_docs = []
            example.answers["ret_read_gpt4"] = example_utils.Answer(answer_string="I don't know.",
                                                                    attribution=[])
            json.dump(dataclasses.asdict(example), f)
            f.write("\n")
            continue

        all_docs = [Document(page_content=d['text'], metadata={'source': d['url']}) for d in all_docs]
        all_docs_content_lens = [len(doc.page_content.strip()) for doc in all_docs]
        if not all_docs or not sum(all_docs_content_lens):
            example.answers["ret_read_gpt4"] = example_utils.Answer(answer_string="I don't know.",
                                                                    attribution=[])
            json.dump(dataclasses.asdict(example), f)
            f.write("\n")
            continue
        gpt_query_text = "I am an expert in the field of " + example.metadata.field + ". Please answer my question: " + query_text
        answer, attributions = retrieval_gpt_generate(gpt_query_text, all_docs, args.topk)
        example.answers["rr_gs_gpt4"] = example_utils.Answer(answer_string=answer,
                                                                attribution=attributions)
        json.dump(dataclasses.asdict(example), f)
        f.write("\n")
