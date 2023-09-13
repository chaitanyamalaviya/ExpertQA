# based off of https://github.com/shmsw25/FActScore/blob/main/factscore/factscorer.py

import argparse
import string
import os
import sys
import numpy as np
from tqdm import tqdm

from load_open_ai_model import OpenAIModel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="input filepath", type=str)
parser.add_argument("--output_file", help="output filepath", type=str)
args = parser.parse_args()


def get_score(topic, atomic_claim, atomic_evidence, lm):
        atomic_claim = atomic_claim.strip()
        # if lm:
        definition = "Answer the question about {} based on the given context.\n\n".format(topic)
        definition += atomic_evidence.strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atomic_claim.strip())

        output = lm.generate(prompt)

        if type(output[1])==np.ndarray:
            # when logits are available
            logits = np.array(output[1])
            assert logits.shape[0] in [32000, 32001]
            true_score = logits[5852]
            false_score = logits[7700]
            is_supported = true_score > false_score
        else:
            # when logits are unavailable
            generated_answer = output[0].lower()
            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index("true") > generated_answer.index("false")
            else:
                is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

        # else:
        #     is_supported = True
        
        return is_supported


if __name__ == '__main__':

    lm = OpenAIModel("ChatGPT", cache_file='.cache/factscore')

    input_data = example_utils.read_examples(args.input_file)

    for example in tqdm(input_data):
        claims = example.answers[list(example.answers.keys())[0]]["claims"]
        for claim in claims:
            score_total = 0
            for i in range(len(claim["atomic_claims"])):
                a_claim = claim["atomic_claims"][i]
                a_evidence = claim["atomic_evidences"][i]
                if "Passage ID 4" in a_evidence:
                    a_evidence = a_evidence.split("Passage ID 4")[0]
                score = get_score(example.metadata.field, a_claim, a_evidence, lm)
                score_total += score
            if claim["atomic_claims"]:
                claim["fact_score"] = float(score_total / len(claim["atomic_claims"]))
            else:
                claim["fact_score"] = None
        example_utils.write_examples(args.output_file, [example], True)
