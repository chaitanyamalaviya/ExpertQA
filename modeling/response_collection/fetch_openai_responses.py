"""Given an answer, split it into claims using GPT3"""

import openai
from tqdm import tqdm
from absl import app
from absl import flags
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import dataclasses
import json
import os
import sys

# import config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils

openai.organization = ""
openai.api_key = ""

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'data/r1_data.jsonl', 'Input filepath.')
flags.DEFINE_string('output_file', 'data/r1_data_answers_gpt4.jsonl', 'Output filepath.')
flags.DEFINE_string('prompt_config', 'prompts/qa_prompt.txt', 'Prompt path.')
flags.DEFINE_string('model_name', 'gpt4', 'Prompt path.')
flags.DEFINE_integer('limit', None, 'Limit to the number of examples processed')


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def main(unused_argv):

    with open(FLAGS.prompt_config) as f:
        prompt = f.read()

    input_data = example_utils.read_examples(FLAGS.input_file)

    f = open(FLAGS.output_file, "w")

    for i, ex in enumerate(tqdm(input_data)):
        if i==FLAGS.limit:
            break
        cur_prompt = prompt[:]
        cur_prompt = cur_prompt.replace("[FIELD]", ex.metadata.field)
        cur_prompt = cur_prompt.replace("[QUESTION]", ex.question)
        # resp = completion_with_backoff(
        #     model=FLAGS.model_name,
        #     prompt=cur_prompt,
        #     max_tokens=2048,
        #     )

        # chatgpt
        resp = chat_completion_with_backoff(
            model="gpt-4",
            messages=[{"role": "user", "content": cur_prompt}],
            max_tokens=2048,
        )
        output = resp["choices"][0]["message"]["content"]

        # output = resp["choices"][0]["text"]
        attribution_begin_idx = output.find("\n[1]")
        if attribution_begin_idx == -1:
            ex.answers[FLAGS.model_name] = example_utils.Answer(answer_string=output.strip())
        else:
            ex.answers[FLAGS.model_name] = example_utils.Answer(answer_string=output[:attribution_begin_idx].strip(),
                                                                  attribution=output[attribution_begin_idx+1:].split("\n"))
        json.dump(dataclasses.asdict(ex), f)
        f.write("\n")
    
    # example_utils.write_examples(FLAGS.output_file, input_data)


if __name__ == "__main__":
    app.run(main)
