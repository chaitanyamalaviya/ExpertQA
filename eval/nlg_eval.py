from absl import app
from absl import flags
from qafacteval import QAFactEval
import os
import sys
from evaluate import load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import example_utils, jsonl_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('prediction_file', 'saved_models/domain_expertqa_flant5_xxl/test_eval_outputs.json', 'Prediction file.')
flags.DEFINE_string('reference_file', 'data/expertqa/domain_lfqa_test.json', 'Reference file.')


def calculate_rouge(predictions, references):
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    print(results)

def calculate_qafacteval_score(predictions, references):
    
    kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
            "verbose": True, "generation_batch_size": 32, \
            "answering_batch_size": 32, "lerc_batch_size": 8}

    model_folder = "eval/qafacteval_models" # path to models downloaded with download_models.sh
    metric = QAFactEval(
        lerc_quip_path=f"{model_folder}/quip-512-mocha",
        generation_model_path=f"{model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{model_folder}/answering",
        lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
        **kwargs
    )

    results = metric.score_batch_qafacteval(references, [predictions], return_qa_pairs=True)
    score = results[0][0]['qa-eval']['lerc_quip']
    print(score)


def main(unused_argv):
    # Example usage
    predictions = [ex["output"][0] for ex in jsonl_utils.read_jsonl(FLAGS.prediction_file)]
    references = [ex["answer"] for ex in jsonl_utils.read_jsonl(FLAGS.reference_file)]
    
    calculate_rouge(predictions, references)
    calculate_qafacteval_score(predictions, references)


if __name__ == "__main__":
    app.run(main)