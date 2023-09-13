"""
Given the full r2 data, and a subset (e.g. domain val/test), extract the subset from the full r2 data.

This is due to the fact that gpt4 and bing-chat in the official train/test/val split only contains evidence urls, not
the full text. We want to extract the processed evidence text + autoais labels from the full r2 data.
"""
from data_utils.jsonl_utils import read_jsonl, write_jsonl


def extract_subset(args):
    r2_data = read_jsonl(args.r2_compiled)
    subset_data = read_jsonl(args.subset)

    r2_data_map = {ex["question"]: ex for ex in r2_data}

    all_examples = []
    for ex in subset_data:
        if ex["question"] not in r2_data_map:
            print(f"{ex['question']} not found in r2 data; This shouldn't happen!!")
            continue

        new_ex = r2_data_map[ex["question"]]
        all_examples.append(new_ex)

    write_jsonl(args.output, all_examples)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2_compiled",
                        default="modeling/auto_attribution/r2_compiled_out_corrected_autoais.jsonl",
                        help="full r2 compiled data",
                        type=str)
    parser.add_argument("--subset",
                        default="data/expertqa/domain_test.jsonl",
                        help="subset of data",
                        type=str)
    parser.add_argument("--output",
                        default="modeling/auto_attribution/domain_test_autoais_0shot.jsonl",
                        help="path to output",
                        type=str)
    args = parser.parse_args()

    extract_subset(args)

