from absl import app
from absl import flags
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils


def find_urls(citations):
    citations = citations.split()
    urls = []
    MAX_URLS = 100
    for i in range(1, MAX_URLS):
        if f"[{i}]:" in citations:
            idx = citations.index(f"[{i}]:")
            urls.append(f"[{i}] " + citations[idx+1])
        else:
            break
    return urls


def main(unused_argv):
    input_data = example_utils.read_examples("data/r1_data.jsonl")

    with open("data/r1_data_answers_bingchat_balanced_compiled.jsonl") as f:
        bing_responses = [json.loads(line) for line in f.readlines()]
    
    cc = 0
    for ex, resp in zip(input_data, bing_responses):
        # ans_text = ans['item']['messages'][-1]['text']
        # source_urls = [url["seeMoreUrl"] for url in ans['item']['messages'][-1]['sourceAttributions']]
        j = 0
        if "text" not in resp['item']['messages'][-1]['adaptiveCards'][0]['body'][0]:
            j += 1
        if len(resp['item']['messages'][-1]['adaptiveCards'][0]['body']) <= j:
             continue
        vcard = resp['item']['messages'][-1]['adaptiveCards'][0]['body'][j]['text']
        if "\n\n" not in vcard:
            citations = []
        else:
            citations, ans_text = vcard.split("\n\n", 1)
            citations = citations.replace("\n", " ")
            source_urls = find_urls(citations)
        ex.answers["bing_chat"] = example_utils.Answer(answer_string=ans_text,
                                                       attribution=source_urls)
        cc += 1

    example_utils.write_examples("data/r1_data_answers_bingchat_balanced_final.jsonl", input_data)
    print(cc)


if __name__ == "__main__":
    app.run(main)