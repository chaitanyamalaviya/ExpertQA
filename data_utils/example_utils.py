"""Utilities for reading and writing examples."""

import dataclasses
import typing
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from data_utils import jsonl_utils


@dataclasses.dataclass(frozen=False)
class Claim:
    """Represents a claim."""
    # Original claim
    claim_string: str
    # Evidence for claim
    evidence: typing.Optional[typing.Sequence[str]] = None
    # Supported (by annotator)
    support: typing.Optional[int] = None
    # Reason for missing support (by annotator)
    reason_missing_support: typing.Optional[str] = None
    # Informativeness of claim for the question (by annotator)
    informativeness: typing.Optional[int] = None
    # Worthiness of citing claim (by annotator)
    worthiness: typing.Optional[int] = None
    # Factual correctness of claim (by annotator)
    correctness: typing.Optional[int] = None
    # Reliability of source (by annotator)
    reliability: typing.Optional[int] = None
    # Revised claim (by annotator)
    revised_claim: typing.Optional[str] = None
    # Revised evidence (by annotator)
    revised_evidence: typing.Optional[typing.Sequence[str]] = None
    # Atomic claims for Fact score estimation
    atomic_claims: typing.Optional[typing.Sequence[str]] = None
    # Atomic claim-evidences for Fact score estimation
    atomic_evidences: typing.Optional[typing.Sequence[str]] = None
    # Fact score for each claim
    fact_score: typing.Optional[float] = None
    # Autoais label for the original claim and original evidence
    autoais_label: typing.Optional[str] = None


@dataclasses.dataclass(frozen=False)
class Answer:
    """Represents an answer."""
    # Original answer returned by system
    answer_string: str
    # Attribution (not linked to specific claims)
    attribution: typing.Optional[typing.Sequence[str]] = None
    # List of claims for the answer
    claims: typing.Optional[typing.Sequence[Claim]] = None
    # Revised answer (by annotator)
    revised_answer_string: typing.Optional[str] = None
    # Usefulness of original answer (by annotator)
    usefulness: typing.Optional[int] = None
    # Time taken for annotating this answer (from annotation)
    annotation_time: typing.Optional[float] = None
    # Prolific ID (TODO: anonymize before release) or email ID of annotator who annotated answer
    annotator_id: typing.Optional[str] = None


@dataclasses.dataclass(frozen=False)
class ExampleMetadata:
    """Optional metadata."""
    # The type of the question.
    question_type: str = None
    # The field of the expert who wrote the the question.
    field: str = None
    # The specific area in the field (if specified)
    specific_field: typing.Optional[str] = None


@dataclasses.dataclass(frozen=False)
class Example:
    """Represents a question."""
    question: str
    # Prolific ID (TODO: anonymize before release) or email ID of annotator who wrote the question
    annotator_id: str
    # Dict of model name to answer string
    answers: typing.Optional[typing.DefaultDict[str, Answer]] = None
    # Optional metadata.
    metadata: typing.Optional[ExampleMetadata] = None


def read_examples(filepath):
    examples_json = jsonl_utils.read_jsonl(filepath)
    examples = [Example(**example) for example in examples_json]
    for example in examples:
        example.metadata = ExampleMetadata(**example.metadata)
    return examples


def write_examples(filepath, examples, append=False):
    examples_json = [dataclasses.asdict(example) for example in examples]
    jsonl_utils.write_jsonl(filepath, examples_json, append)
