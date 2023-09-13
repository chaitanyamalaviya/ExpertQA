'''Script to convert examples for LFQA'''

from absl import app
from absl import flags
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils, jsonl_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_prefix', 'data/expertqa/rand_', 'Input filepath.')
flags.DEFINE_string('output_prefix', 'data/expertqa/rand_lfqa_', 'Output filepath.')


def main(unused_argv):
    train_data = example_utils.read_examples(FLAGS.input_prefix + "train.jsonl")
    val_data = example_utils.read_examples(FLAGS.input_prefix + "val.jsonl")
    test_data = example_utils.read_examples(FLAGS.input_prefix + "test.jsonl")
    train_wdata = []
    for i, ex in enumerate(train_data):
        for answer_model_name, answer in ex.answers.items():
            answer_string = re.sub(r'\[\d+\]', '', answer["revised_answer_string"]).replace("\n"," ")
            if not answer_string.strip() or answer_string == None:
                continue
            train_wdata.append({"example_id": len(train_wdata), "context": "", "question": ex.question, "answer": answer_string})
    
    val_wdata = []
    for i, ex in enumerate(val_data):
        for answer_model_name, answer in ex.answers.items():
            answer_string = re.sub(r'\[\d+\]', '', answer["revised_answer_string"]).replace("\n"," ")
            if not answer_string.strip() or answer_string == None:
                continue
            val_wdata.append({"example_id": len(val_wdata), "context": "", "question": ex.question, "answer": answer_string})

    test_wdata = []
    for i, ex in enumerate(test_data):
        for answer_model_name, answer in ex.answers.items():
            # if answer["usefulness"] == "Not useful at all":
            #     continue
            answer_string = re.sub(r'\[\d+\]', '', answer["revised_answer_string"]).replace("\n"," ")
            if not answer_string.strip() or answer_string == None:
                continue
            test_wdata.append({"example_id": len(test_wdata), "context": "", "question": ex.question, "answer": answer_string})
    
    jsonl_utils.write_jsonl(FLAGS.output_prefix + "train.json", train_wdata)
    jsonl_utils.write_jsonl(FLAGS.output_prefix + "val.json", val_wdata)
    jsonl_utils.write_jsonl(FLAGS.output_prefix + "test.json", test_wdata)


if __name__ == "__main__":
    app.run(main)