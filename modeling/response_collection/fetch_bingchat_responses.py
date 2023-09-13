'''Script to get retrieve BingChat responses for questions, that uses https://github.com/acheong08/EdgeGPT. 
You would need to request access to BingChat, then follow directions in the above repo to get `cookies.json`.
'''

import asyncio
import json
from EdgeGPT import Chatbot, ConversationStyle
from absl import app
from absl import flags
from tqdm import tqdm
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'data/r1_data.jsonl', 'Input filepath.')
flags.DEFINE_string('output_file', 'data/r1_data_answers_bingchat_balanced_part2.jsonl', 'Output filepath.')
flags.DEFINE_integer('limit', None, 'Limit to the number of examples processed.')
flags.DEFINE_integer('start_idx', 0, 'Start index.')


async def get_response(question):
    # cookies = json.loads(open("data/cookies.json", encoding="utf-8").read())
    # bot = await Chatbot.create(cookies=cookies)
    bot = await Chatbot.create()
    resp = await bot.ask(prompt=question, conversation_style=ConversationStyle.balanced)
    await bot.close()
    return resp


def find_urls(citations):
    citations = citations.split()
    urls = []
    MAX_URLS = 100
    for i in range(1, MAX_URLS):
        if f"[{i}]:" in citations:
            idx = citations.index(f"[{i}]:")
            urls.append(citations[idx+1])
        else:
            break
    return urls


def main(unused_argv):
    input_data = example_utils.read_examples(FLAGS.input_file)
    sampled_data = input_data[FLAGS.start_idx:]
    fw = open(FLAGS.output_file, "a")
    for i, ex in enumerate(tqdm(sampled_data)):
        if i==FLAGS.limit:
            break
        query = f"I am an expert from the field of {ex.metadata.field}. Please answer my question: " + ex.question
        ans = asyncio.run(get_response(query))
        # if "messages" not in ans['item']:
        #     continue
        # if "text" not in ans['item']['messages'][-1]['adaptiveCards'][0]['body'][0]:
        #     continue
        # ans_text = ans['item']['messages'][-1]['text']
        # source_urls = [url["seeMoreUrl"] for url in ans['item']['messages'][-1]['sourceAttributions']]
       
        # vcard = ans['item']['messages'][-1]['adaptiveCards'][0]['body'][0]['text']
        # citations, ans_text = vcard.split("\n\n", 1)
        # citations = citations.replace("\n", " ")
        # source_urls = find_urls(citations)
        # ex.answers["bing_chat"] = example_utils.Answer(answer_string=ans_text,
        #                                                attribution=source_urls)
        json.dump(ans, fw)
        fw.write("\n")
        time.sleep(3)

    # example_utils.write_examples(FLAGS.output_file, sampled_data)


if __name__ == "__main__":
    app.run(main)