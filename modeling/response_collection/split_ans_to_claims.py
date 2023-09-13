"""Given an answer, split it into claims either using sentence boundaries or using GPT."""

import openai
import nltk
from tqdm import tqdm
from absl import app
from absl import flags
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import os
import re
import sys
import spacy
from spacy.lang.en import English

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
# nlp.add_pipe("sentencizer")

# import config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_utils import example_utils

openai.organization = ""
openai.api_key = ""

MIN_CLAIM_LENGTH=5

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', 'data/r1_data_answers.jsonl', 'Input filepath.')
flags.DEFINE_string('output_file', 'data/r1_data_answers_claims.jsonl', 'Output filepath.')
flags.DEFINE_string('granularity', 'sentence', 'Boundary for determining claims.')
flags.DEFINE_string('split_model_name', 'gpt4', 'Model name (required for sub-sentence claims).')
flags.DEFINE_string('prompt_config', 'prompts/claims_prompt.txt', 'Prompt path (required for sub-sentence claims).')
flags.DEFINE_integer('limit', None, 'Limit to the number of examples processed')

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
cite_pattern = r'\[\^.*?\^\]'


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def fix_sentence_boundaries(claim_sents):
    modified_sentences = []
    for i in range(len(claim_sents)):
        if i == len(claim_sents) - 1:
            modified_sentences.append(claim_sents[i])
            break
        
        next_sentence = claim_sents[i+1]
        bracketed_strings = ''
        while next_sentence.startswith('[') and ']' in next_sentence:
            end_index = next_sentence.index(']') + 1
            bracketed_strings += next_sentence[:end_index]
            next_sentence = next_sentence[end_index:].lstrip()  # remove leading whitespaces

        modified_sentences.append(claim_sents[i] + ' ' + bracketed_strings)
        claim_sents[i+1] = next_sentence
    return modified_sentences


def remove_passage_numbers_and_ids(string):
    pattern = r"Passage (Number|ID) \d+(?:[:])?"
    return re.sub(pattern, "", string)


def main(unused_argv):

    input_data = example_utils.read_examples(FLAGS.input_file)

    with open(FLAGS.prompt_config) as f:
        prompt = f.read()

    for i, ex in enumerate(tqdm(input_data)):
        if i==FLAGS.limit:
            break
        if "post_hoc_gs_gpt4" in FLAGS.input_file:
            del ex.answers["gpt4"]
            ex.answers["post_hoc_gs_gpt4"] = ex.answers.pop("gpt4_post_hoc_gs")
        elif "post_hoc_sphere_gpt4" in FLAGS.input_file:
            del ex.answers["gpt4"]
            ex.answers["post_hoc_sphere_gpt4"] = ex.answers.pop("gpt4_post_hoc_sphere_bm25")
        elif "rr_gs_gpt4" in FLAGS.input_file:
            ex.answers["rr_gs_gpt4"] = ex.answers.pop("ret_read_gpt4")
        for answer_model_name, answer in ex.answers.items():        
            if not answer["answer_string"].strip():
                continue
            cleaned_answer = re.sub(cite_pattern, '', answer["answer_string"]).strip()
            if "bingchat" in FLAGS.input_file:
                cleaned_answer = cleaned_answer.replace("]\n\nAre", "]. \n\nAre").replace("]\n\nWhat", "]. \n\nWhat").replace("]\n\nWhich", "]. \n\nWhich").replace("]\n\nHow", "]. \n\nHow").replace("]\n\nDoes", "]. \n\nDoes")
                cleaned_answer = cleaned_answer.replace("😊", "")
            ex.answers[answer_model_name]["answer_string"] = cleaned_answer
            # claim_sents = nltk.sent_tokenize(cleaned_answer.replace("\n", " "))
            doc = nlp(cleaned_answer.replace("\n", " "))
            claim_sents = [sent.text for sent in list(doc.sents)]

            # Citations in text-davinci-003 appear after the sentence boundary
            if answer_model_name == "text-davinci-003":
                claim_sents = fix_sentence_boundaries(claim_sents)

            if FLAGS.granularity == "sentence":
                if "post_hoc" not in answer_model_name:
                    ex.answers[answer_model_name]["claims"] = []
                    for claim in claim_sents:
                        # Check if claim needs to be filtered out
                        filt = False
                        for f_claim in filter_out_claims:
                            if f_claim in claim:
                                filt = True
                                break
                        # Exclude very short claims as they are likely a result of incorrect sentence tokenization
                        if len(claim) <= MIN_CLAIM_LENGTH:
                            filt = True
                        # Exclude questions from claims
                        if claim[-1].strip() == "?":
                            filt = True
                        
                        # If claim is being filtered, also remove it from the answer string
                        if filt:
                            ex.answers[answer_model_name]["answer_string"] = ex.answers[answer_model_name]["answer_string"].replace(claim.strip(), "")
                            continue
                        # Remove any substring enclosed by '[^' and '^]' as these indicate index of citation for BingChat
                        cleaned_claim = re.sub(cite_pattern, '', claim)
                        # Find all citation indexes
                        citation_matches = [i for i in re.findall(r'\[(.*?)\]', cleaned_claim) if i.isdigit()]
                        # Check if any attribution exists for the answer
                        if answer["attribution"] is not None and len(answer["attribution"]):
                            # Check if each attribution is a dict (true for `rr` systems with "text" and "url" keys)
                            if isinstance(answer["attribution"][0], dict):
                                evidence = []
                                for c_idx in citation_matches:
                                    # Invalid citation index
                                    if int(c_idx)-1 >= len(answer["attribution"]):
                                        continue
                                    evidence.append("[" + c_idx + "] " + answer["attribution"][int(c_idx)-1]["url"] + "\n\n" + remove_passage_numbers_and_ids(answer["attribution"][int(c_idx)-1]["text"]))
                                ex.answers[answer_model_name]["claims"].append(example_utils.Claim(cleaned_claim,
                                                                                                   evidence=evidence))
                            # true for bing_chat and gpt4
                            else:
                                ex.answers[answer_model_name]["claims"].append(example_utils.Claim(cleaned_claim,
                                                                                                   evidence=[answer["attribution"][int(c_idx)-1] for c_idx in citation_matches]))
                        else:
                            ex.answers[answer_model_name]["claims"].append(example_utils.Claim(cleaned_claim,
                                                                                               evidence=[]))
                    # Update attribution list for `rr` systems (initially a list of dicts to a list of strings)
                    if answer["attribution"] is not None and len(answer["attribution"]) and isinstance(answer["attribution"][0], dict):
                        ex.answers[answer_model_name]["attribution"] = ["[" + str(i+1) + "] " + attr["url"] for i, attr in enumerate(answer["attribution"])]
                else:
                    # post-hoc citation processing (already split into claims)
                    evidence = []
                    ex.answers[answer_model_name]["attribution"] = []
                    claim_idx = 1
                    claims = ex.answers[answer_model_name]["claims"][:]
                    ex.answers[answer_model_name]["claims"] = []                    
                    for claim in claims:
                        cleaned_claim = re.sub(r'\[\d+\]', '', claim["claim_string"])
                        if not cleaned_claim.strip() or len(cleaned_claim) <= MIN_CLAIM_LENGTH:
                            continue
                        claim["claim_string"] = cleaned_claim
                        evidence = []
                        if claim["evidence"]:
                            evidence = [f"[{claim_idx}] " + claim["evidence"][0]["url"] + "\n\n" + claim["evidence"][0]["text"]]
                            claim["claim_string"] = claim["claim_string"][:-1] + f"[{claim_idx}]" + claim["claim_string"][-1]
                            ex.answers[answer_model_name]["attribution"].append(f"[{claim_idx}] " + claim["evidence"][0]["url"])
                            claim_idx += 1
                        claim["evidence"] = evidence
                        ex.answers[answer_model_name]["claims"].append(claim)
                    ex.answers[answer_model_name]["answer_string"] = " ".join([claim["claim_string"] for claim in ex.answers[answer_model_name]["claims"]])

            else:
                for sent in claim_sents:
                    cur_prompt = prompt[:]
                    cur_prompt = cur_prompt.replace("[SENTENCE]", sent)
                    resp = completion_with_backoff(
                        model=FLAGS.split_model_name,
                        prompt=cur_prompt,
                        max_tokens=200,
                    )
                    output = resp["choices"][0]["text"]
                    claim_strings = output.split("- ")
                    ex.answers[answer_model_name]["claims"] += [example_utils.Claim(claim, 
                                                                                    evidence=[answer["attribution"][int(c_idx)-1] for c_idx in citation_matches]) for claim in claim_strings if claim.strip()]  
    
    example_utils.write_examples(FLAGS.output_file, input_data)


if __name__ == "__main__":
    app.run(main)
