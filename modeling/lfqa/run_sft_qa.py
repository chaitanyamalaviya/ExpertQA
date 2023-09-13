# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer
from accelerate import Accelerator
from peft import PeftModel    

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils, jsonl_utils

tqdm.pandas()


llama2_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n[QUESTION] [/INST]"
alpaca_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following question completely and accurately.\n\n### Input:\n[QUESTION]\n\n### Response:"
vicuna_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: [QUESTION]\nASSISTANT:"


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    train_file: Optional[str] = field(
        default="data/expertqa/rand_lfqa_train.json", metadata={"help": "train file"}
    )
    validation_file: Optional[str] = field(
        default="data/expertqa/rand_lfqa_val.json", metadata={"help": "validation file"}
    )
    test_file: Optional[str] = field(
        default="data/expertqa/rand_lfqa_test.json", metadata={"help": "test file"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    learning_rate_scheduler: Optional[str] = field(default="linear", metadata={"help": "the learning rate scheduler"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    do_train: Optional[bool] = field(default=True, metadata={"help": "train the model on train_file"})
    do_eval: Optional[bool] = field(default=True, metadata={"help": "evaluate on validation_file"})

    # parser.add_argument("--eos_token_id", type=int, default=49152)

    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    # parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    # parser.add_argument("--num_warmup_steps", type=int, default=100)
    # parser.add_argument("--weight_decay", type=float, default=0.05)

    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--no_fp16", action="store_false")
    # parser.add_argument("--bf16", action="store_true", default=False)
    # parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--num_workers", type=int, default=None)
    # parser.add_argument("--output_dir", type=str, default="./checkpoints")
    # parser.add_argument("--log_freq", default=1, type=int)
    # parser.add_argument("--eval_freq", default=1000, type=int)
    # parser.add_argument("--save_freq", default=1000, type=int)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"### Question: {example['question']}\n### Answer: {example['answer']}"
    text = llama2_prompt.replace("[QUESTION]", example["question"]) + f"\n{example['answer']}</s>"
    return text


def create_datasets(tokenizer, script_args):
    data_files = {}
    if script_args.train_file is not None:
        data_files["train"] = script_args.train_file
        extension = script_args.train_file.split(".")[-1]
    if script_args.validation_file is not None:
        data_files["validation"] = script_args.validation_file
        val_filename = script_args.validation_file
        extension = script_args.validation_file.split(".")[-1]
    if script_args.test_file is not None:
        data_files["test"] = script_args.test_file
        extension = script_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir="/mnt/nlpgridio3/data/cmalaviya/huggingface_data/")

    print(f"Size of the train set: {len(raw_datasets['train'])}. Size of the validation set: {len(raw_datasets['validation'])}")

    chars_per_token = chars_token_ratio(raw_datasets['train'], tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # TODO: Do we really need ConstantLengthDataset?
    
    train_dataset = ConstantLengthDataset(
        tokenizer,
        raw_datasets['train'],
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        raw_datasets['validation'],
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )
    test_dataset = ConstantLengthDataset(
        tokenizer,
        raw_datasets['test'],
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset, test_dataset


def main():

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # This means: fit the entire model on the GPU:0
        device_map = {"": 0}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    # config = AutoConfig.from_pretrained(script_args.model_name)
    # architecture = config.architectures[0]
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_auth_token=True)
    print("Setting EOS, BOS, and UNK tokens")
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "bos_token": "</s>",
            "unk_token": "</s>",
        }
    )
    print("Loading the model")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )

    # Step 2: Load the dataset
    # dataset = load_dataset(script_args.dataset_name, split="train")
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, script_args)

    if script_args.do_train:
        # Step 3: Define the training arguments
        training_args = TrainingArguments(
            output_dir=script_args.output_dir,
            per_device_train_batch_size=script_args.batch_size,
            per_device_eval_batch_size=script_args.batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            learning_rate=script_args.learning_rate,
            logging_steps=script_args.logging_steps,
            num_train_epochs=script_args.num_train_epochs,
            max_steps=script_args.max_steps,
        )
        # training_args = TrainingArguments(
        #     output_dir=args.output_dir,
        #     dataloader_drop_last=True,
        #     evaluation_strategy="steps",
        #     max_steps=args.max_steps,
        #     eval_steps=args.eval_freq,
        #     save_steps=args.save_freq,
        #     logging_steps=args.log_freq,
        #     per_device_train_batch_size=args.batch_size,
        #     per_device_eval_batch_size=args.batch_size,
        #     learning_rate=args.learning_rate,
        #     lr_scheduler_type=args.lr_scheduler_type,
        #     warmup_steps=args.num_warmup_steps,
        #     gradient_accumulation_steps=args.gradient_accumulation_steps,
        #     gradient_checkpointing=not args.no_gradient_checkpointing,
        #     fp16=not args.no_fp16,
        #     bf16=args.bf16,
        #     weight_decay=args.weight_decay,
        #     run_name="llama-7b-finetuned",
        #     report_to="wandb",
        #     ddp_find_unused_parameters=False,
        # )

        # Step 4: Define the LoraConfig
        if script_args.use_peft:
            peft_config = LoraConfig(
                r=script_args.peft_lora_r,
                lora_alpha=script_args.peft_lora_alpha,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            peft_config = None

        # Step 5: Define the Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # dataset_text_field=script_args.dataset_text_field,
            peft_config=peft_config,
            packing=True
        )
        print("Training...")
        trainer.train()

        # Step 6: Save the model
        print("Saving the model")
        trainer.save_model(script_args.output_dir)
        # trainer.model.save_pretrained(os.path.join(script_args.output_dir, "final_checkpoint/"))

    if script_args.do_eval:
        adapters_name = script_args.output_dir
        model = PeftModel.from_pretrained(model, adapters_name)
        model = model.merge_and_unload()
        model = model.cuda()
        validation_examples = jsonl_utils.read_jsonl(script_args.validation_file)
        for ex in tqdm(validation_examples):
            cur_prompt = llama2_prompt.replace("[QUESTION]", ex["question"])
            input_ids = tokenizer(cur_prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.cuda()
            outputs = model.generate(input_ids, max_new_tokens=4096, do_sample=True, top_p=0.9, temperature=0.9)
            predicted_ans = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(cur_prompt):]            
            ex["output"] = [predicted_ans]
        
        jsonl_utils.write_jsonl(os.path.join(script_args.output_dir, "test_predictions.jsonl"), validation_examples)

        # predictions = trainer.predict(test_dataset=val_dataset, metric_key_prefix="val")


if __name__ == "__main__":
    main()