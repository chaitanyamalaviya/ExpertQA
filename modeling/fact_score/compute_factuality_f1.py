"""
Compute FActScore F1 scores with respect to human factuality judgements.
"""
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_utils.jsonl_utils import read_jsonl


def did_abstain(answer):
    if "The context does not" in answer or \
    "I don't know" in answer or \
    "The context provided does not" in answer or \
    "The context provided doesn't" in answer or \
    "The context doesn't" in answer or \
    "not sure if I understand" in answer or \
    "I'm sorry, but I can't" in answer or \
    "The given context does not" in answer or \
    "I couldn't find any relevant information" in answer:
        return True
    else:
        return False


def is_void_claim(claim):
    # Filter out any claims that contain the following substrings
    filter_out_claims = set(["Hello",
                         "Hello, this is Bing.", 
                         "this is Bing",
                         "Thank you for your question"
                         "glad you're",
                         "glad you",
                         "glad to help you",
                         "glad to hear",
                         "Bing",
                         "Hope this helps",
                         "Hope this is helpful",
                         "I can provide more information based on your answer", 
                         "are you interested in?", 
                         "Are you looking for", 
                         "what aspect are you interested", 
                         "I found some information that might help you", 
                         "I found some information", 
                         "happy to help you", 
                         "I hope this information",
                         "I hope this helps",
                         "I hope this answer helps you",
                         "I can provide more specific information",
                         "I can provide more details",
                         "I searched for your question and found some results.", 
                         "I can help you",
                         "Thank you for your question",
                         "That's an interesting question",
                         "not sure if I understand your question",
                         "I hope this answers your question",
                         "I can provide you with more specific resources",
                         "I can try to answer your question",
                         "I can provide more information",
                         "I can search for more",
                         "I can answer your question",
                         "I can give you more",
                         "I can try to summarise",
                         "I searched for your question and found some information",
                         "I can answer your question",
                         "I can provide you",
                         "I can try to find",
                         "I cannot give you a definitive answer",
                         "I might be able to provide",
                         "please let me know"])
    for f_claim in filter_out_claims:
        if f_claim in claim:
            return True
    return False




def human_corr(input_file):
    label_to_score = {"Definitely correct": 1.0, "Probably correct": 0.5, "Unsure": 0.0,
                      "Likely incorrect": -0.5, "Definitely incorrect": -1.0}
    examples = read_jsonl(input_file)
    threshold = 0.5

    preds = []
    golds = []
    ipreds = []
    igolds = []
    per_sys_preds = {}
    per_sys_golds = {}
    for ex in examples:
        for system, answer in ex["answers"].items():
            if did_abstain(answer["answer_string"]):
                continue

            if system not in per_sys_preds:
                per_sys_preds[system] = []
                per_sys_golds[system] = []

            for c in answer["claims"]:

                if c["correctness"] == None or c["fact_score"] == None or is_void_claim(c["claim_string"]):
                    continue
                fact_score = c["fact_score"]
                factuality_label = label_to_score[c["correctness"]]

                preds.append(fact_score)
                golds.append(factuality_label)
                ipreds.append(1 if fact_score >= threshold else 0)
                igolds.append(1 if " correct" in c["correctness"] else 0)
                per_sys_preds[system].append(fact_score)
                per_sys_golds[system].append(factuality_label)

    print(f"Accuracy: {accuracy_score(ipreds, igolds)}")
    print(f"F1 score averaged: {f1_score(igolds, ipreds, average='micro')}")
    print(f"F1 score breakdown (P/R/F1/support): {precision_recall_fscore_support(igolds, ipreds, average=None)}")
    print(f"Total number of claims: {len(golds)}\n")

    for system, sys_golds in per_sys_golds.items():
        sys_preds = [1 if fact_score >= threshold else 0 for fact_score in per_sys_preds[system]]
        sys_golds_binary = [1 if gold >= 0.5 else 0 for gold in sys_golds]
        print(f"{system}")
        print(f"Accuracy: {accuracy_score(sys_golds_binary, sys_preds)}")
        print(f"F1 score averaged: {f1_score(sys_golds_binary, sys_preds, average='micro')}")
        print(f"F1 score breakdown (P/R/F1/support): {precision_recall_fscore_support(sys_golds_binary, sys_preds, average=None)}")
        print(f"\t\tNumber of claims: {len(sys_golds_binary)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        default="data/r2_compiled_out_corrected_revised_atomic_w_evidences_factscores.jsonl",
                        help="example file with per-claim fact scores",
                        type=str)
    args = parser.parse_args()

    human_corr(args.input_file)
