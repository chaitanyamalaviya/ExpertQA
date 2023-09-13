"""Finetune AutoAIS model on human judgements in ExpertQA. """

import json
import os
import argparse
import numpy as np
from functools import partial
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import Dataset
import torch
import evaluate
import nltk
import numpy as np

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from data_utils.jsonl_utils import read_jsonl
from modeling.auto_attribution.autoais import cleanup_claim_and_evidence, format_example_for_autoais

nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")
# evaluation generation args
gen_kwargs = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_new_tokens": 50,
    "min_length": 30,
    "no_repeat_ngram_size": 3,
    "num_beams": 4,
}


def load_expertqa_for_autoais(data_path):
    ds = read_jsonl(data_path)

    autoais_examples = []
    for ex in ds:
        for system, res in ex["answers"].items():
            for c in res["claims"]:
                if "evidence" not in c or len(c["evidence"]) == 0:
                    continue

                label = "1" if c["support"] == "Complete" else "0"
                claim_str = cleanup_claim_and_evidence(c["claim_string"])
                evidence_str = [cleanup_claim_and_evidence(e) for e in c["evidence"]]
                evidence_str = " ".join(evidence_str)

                input_str = format_example_for_autoais(evidence=evidence_str, claim=claim_str)
                autoais_examples.append({
                    "inputs": input_str,
                    "targets": label
                })

    return autoais_examples


def preprocess_function(sample,
                        tokenizer,
                        padding: str = "max_length",
                        max_seq_len: int = 512,
                        input_column: str = "inputs",
                        target_column: str = "targets"
                        ):
    # created prompted input
    inputs = sample[input_column]
    targets = sample[target_column]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_seq_len, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_seq_len, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def training_function(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    train_dataset = load_expertqa_for_autoais(args.train_data)
    eval_dataset = load_expertqa_for_autoais(args.val_data)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = Dataset.from_list(eval_dataset)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    preproc_fn = partial(preprocess_function, tokenizer=tokenizer)
    train_dataset = train_dataset.map(preproc_fn, batched=True)
    eval_dataset = eval_dataset.map(preproc_fn, batched=True)


    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]
    model_name = args.model_id.split("/")[-1]
    if args.output_dir is None:
        output_dir = "out"
    else:
        output_dir = args.output_dir

    model_output_dir = os.path.join(output_dir, model_name)
    log_output_dir = os.path.join(output_dir, "logs")

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        bf16_full_eval=args.bf16,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=log_output_dir,
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(output_dir)
    trainer.create_model_card()
    # Push the results to the hub
    if args.repository_id:
        trainer.push_to_hub()

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--train_data", type=str, default="data/expertqa/domain_train.jsonl", help="Path to expertqa training split.")
    parser.add_argument("--val_data", type=str, default="data/expertqa/domain_val.jsonl", help="Path to expertqa validation split.")

    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=10, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=3, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Path to deepspeed config file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to model + log output dir")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    return args

def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()