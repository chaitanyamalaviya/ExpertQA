'''Given a file where autoais score has already been added to each claim, compute dataset-wise autoais score. '''
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils


def did_abstain(answer):
    if "The context does not" in answer["answer_string"] or \
        "I don't know" in answer["answer_string"] or \
        "The context provided does not" in answer["answer_string"] or \
        "The context provided doesn't" in answer["answer_string"] or \
        "The context doesn't" in answer["answer_string"] or \
        "not sure if I understand" in answer["answer_string"] or \
        "I'm sorry, but I can't" in answer["answer_string"] or \
        "The given context does not" in answer["answer_string"] or \
        "I couldn't find any relevant information" in answer["answer_string"]:
        return True
    else:
        return False

def print_autoais_for_system(predictions):
    # Compute macro- (question-level) and micro- (claim-level) averaged autoais score over all predictions for a system
    all_autoais_scores = {}
    all_autoais_scores_without_abstain = {}
    all_revised_autoais_scores = {}
    abstain_questions_count = {}
    empty_evidence_count = {}
    total_claim_count = {}
    valid_claim_count = {}
    for ex in predictions:
        for system, answer in ex.answers.items():
            if system not in all_autoais_scores:
                all_autoais_scores[system] = []
                all_revised_autoais_scores[system] = []
                empty_evidence_count[system] = 0
                total_claim_count[system] = 0
                valid_claim_count[system] = 0
                all_autoais_scores_without_abstain[system] = []
                abstain_questions_count[system] = 0

            per_answer_autoais = []
            is_abstain = did_abstain(answer)
            for c in answer["claims"]:
                total_claim_count[system] += 1

                # if c["support"] != "Complete":
                #     continue
                # Skip the claim if there's no evidence, or if the evidence is too short
                # TODO(sihaoc): Change this if this is not what we want
                if not c["evidence"] or len(c["evidence"]) == 0:
                    empty_evidence_count[system] += 1
                    continue

                evidence_joined = "|".join(c["evidence"])
                if "http" in evidence_joined and system == "gpt4":  # for GPT4, we discount evidence without valid links
                    continue

                if c["support"] != "Complete" or c["worthiness"] != "Yes":
                    continue

                valid_claim_count[system] += 1

                if c["autoais_label"] == "Y":
                    per_answer_autoais.append(1)
                else:
                    per_answer_autoais.append(0)

            all_autoais_scores[system].append(per_answer_autoais)
            if not is_abstain:
                all_autoais_scores_without_abstain[system].append(per_answer_autoais)
            else:
                abstain_questions_count[system] += 1


    for system, scores in all_autoais_scores.items():
        # macro avg (avg autoais across question/answer)
        all_answer_autoais = [np.mean(ex) for ex in scores if not np.isnan(np.mean(ex))]
        # all_answer_autoais = np.nan_to_num(all_answer_autoais)  # if the answer doesn't have a claim
        #                                                         # Assign avg ais score of 0 to the answer
        macro_autoais = np.mean(all_answer_autoais)

        scores_wo_abstain = all_autoais_scores_without_abstain[system]
        all_answer_autoais_wo_abstain = [np.mean(ex) for ex in scores_wo_abstain if not np.isnan(np.mean(ex))]
        # all_answer_autoais_wo_abstain = np.nan_to_num(all_answer_autoais_wo_abstain)
        macro_autoais_wo_abstain = np.mean(all_answer_autoais_wo_abstain)

        # micro avg (avg autoais across claims)
        all_autoais_scores_flat = []
        all_autoais_scores_flat_wo_abstain = []
        for ex in scores:
            all_autoais_scores_flat += ex
        for ex in scores_wo_abstain:
            all_autoais_scores_flat_wo_abstain += ex
        micro_autoais = np.mean(all_autoais_scores_flat)
        micro_autoais_wo_abstain = np.mean(all_autoais_scores_flat_wo_abstain)

        print(f"System: {system}")
        print(f"\tMacro Avg. (avg autoais across questions): {macro_autoais:.3f}")
        print(f"\tMicro Avg. (avg autoais across claims): {micro_autoais:.3f}")
        print(f"\tNumber of claims without evidence: {empty_evidence_count[system]} / {total_claim_count[system]}")
        print(f"\tAbstain question count: {abstain_questions_count[system]} / {len(all_autoais_scores[system])}")
        print(f"\tClaims worthy + has complete support: {valid_claim_count[system]}")
        print("\tPerformance without abstained answers (i.e. excluding all 'I don't know' answers)")
        print(f"\t\tMacro Avg. (avg autoais across questions): {macro_autoais_wo_abstain:.3f}")
        print(f"\t\tMicro Avg. (avg autoais across claims): {micro_autoais_wo_abstain:.3f}")


def main(args):
    print("Reading system predictions with autoais score per claim...")
    for fin in args.prediction_files:
        input_data = example_utils.read_examples(fin)
        print_autoais_for_system(input_data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_files", nargs='+', help="bar", type=str)
    args = parser.parse_args()

    main(args)