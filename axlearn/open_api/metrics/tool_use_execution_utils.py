# Copyright © 2024 Apple Inc.
"""Utilities for the detailed too use metrics."""

import re
import string
from typing import Dict, List, Union

from typing_extensions import TypeAlias

Value = Union[str, int, bool, float]
ValueOrListOf: TypeAlias = Union[Value, List[Value]]

STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "couldn",
    "couldn't",
    "d",
    "did",
    "didn",
    "didn't",
    "do",
    "does",
    "doesn",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "hadn't",
    "has",
    "hasn",
    "hasn't",
    "have",
    "haven",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightn't",
    "more",
    "most",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "needn",
    "needn't",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "y",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def string_lenient_transform(str_value: str) -> str:
    normalized_str_value = str_value.lower()
    normalized_str_value = normalized_str_value.translate(str.maketrans("", "", string.punctuation))
    words = normalized_str_value.split()
    while len(words) > 1 and words[0] in STOP_WORDS:
        words = words[1:]
    while len(words) > 1 and words[-1] in STOP_WORDS:
        words = words[:-1]
    normalized_str_value = " ".join(words)
    return normalized_str_value


def word_set(input_str: str) -> set[str]:
    """
    Returns a set of words splitting by whitespace, newline and tab,
    and skipping empty strings from the set.
    """
    return {s for s in re.split(r"\s+", input_str) if s}


def match_strings_bag_of_words(pred_str: str, target_str: str, threshold: float = 1.0) -> bool:
    """
    Match strings using a bag of words approach.

    Args:
    ----
        pred_str: .
        target_str: .
        threshold: applied to (# unique common words) / (# unique pred_str words)
    """
    if pred_str == target_str:
        return True

    assert threshold > 0, "bag of words string matching threshold must be above 0"
    pred_word_set = word_set(pred_str)
    target_word_set = word_set(target_str)
    common_words = target_word_set.intersection(pred_word_set)
    if len(pred_word_set) == 0:
        return False
    ratio = len(common_words) / len(target_word_set)
    return ratio >= threshold


def is_arg_value_equal(
    pred_arg: ValueOrListOf,
    target_arg: ValueOrListOf,
    check_lenient: bool,
    bag_of_words: bool,
) -> bool:
    if check_lenient:
        if (
            isinstance(pred_arg, list)
            and isinstance(target_arg, list)
            and len(pred_arg) == len(target_arg)
        ):
            return all(
                is_arg_value_equal(
                    pred_arg=el_pred,
                    target_arg=el_target,
                    check_lenient=check_lenient,
                    bag_of_words=bag_of_words,
                )
                for el_pred, el_target in zip(pred_arg, target_arg)
            )
        # Only handling string payloads for lenient evaluation
        if isinstance(pred_arg, str) and isinstance(target_arg, str):
            pred_lenient = string_lenient_transform(pred_arg)
            target_lenient = string_lenient_transform(target_arg)
            if bag_of_words:
                return match_strings_bag_of_words(
                    pred_str=pred_lenient, target_str=target_lenient, threshold=1.0
                )
            return pred_lenient == target_lenient
    return pred_arg == target_arg


def get_kwargs_alignment(
    pred_args: Dict[str, ValueOrListOf],
    target_args: Dict[str, ValueOrListOf],
    check_lenient: bool = False,
    bag_of_words: bool = False,
) -> bool:
    # Check names are not duplicated
    target_args_copy = dict(target_args.items())

    # Find different label/value pairs
    for pred_arg_name in pred_args:
        if pred_arg_name in target_args_copy and is_arg_value_equal(
            pred_args[pred_arg_name],
            target_args[pred_arg_name],
            check_lenient,
            bag_of_words,
        ):
            target_args_copy.pop(pred_arg_name)
        else:
            return False

    # If there are still element in to_kwargs, to_kwargs contains more entries than from_kwargs
    # and miss.
    return len(target_args_copy) == 0
