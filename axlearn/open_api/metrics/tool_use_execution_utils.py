# Copyright © 2024 Apple Inc.
"""Utilities for the detailed tool use metrics."""

import re
import string
from enum import Enum
from typing import Union

from typing_extensions import TypeAlias

Value = Union[str, int, bool, float]
ValueOrListOf: TypeAlias = Union[Value, list[Value]]

_STOP_WORDS = {
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


class ArgumentMatchType(Enum):
    """The argument matching types."""

    STRICT = 1
    LENIENT = 2
    LENIENT_BAG_OF_WORD = 3


def _string_lenient_transform(str_value: str) -> str:
    """Performs a lenient string transformation."""
    normalized_str_value = str_value.lower()
    normalized_str_value = normalized_str_value.translate(str.maketrans("", "", string.punctuation))
    words = normalized_str_value.split()
    start_index = next(
        (i for i in range(len(words)) if words[i] not in _STOP_WORDS),
        len(words) - 1 if len(words) > 0 and words[-1] not in _STOP_WORDS else len(words),
    )

    end_index = next(
        (
            i + 1
            for i in reversed(range(start_index + 1, len(words)))
            if words[i] not in _STOP_WORDS
        ),
        start_index + 1,
    )

    words = words[start_index:end_index]
    normalized_str_value = " ".join(words)
    return normalized_str_value


def _word_set(input_str: str) -> set[str]:
    """Returns a set of words splitting by whitespace, newline and tab,
    and skipping empty strings from the set."""
    return {s for s in re.split(r"\s+", input_str) if s}


def _match_strings_bag_of_words(*, pred_str: str, target_str: str, threshold: float = 1.0) -> bool:
    """Match strings using a bag of words approach.

    Args:
        pred_str: The predicted argument string.
        target_str: The target argument string.
        threshold:  Thresold to be compared with the ratio (# unique common words) / (# unique
            pred_str words). The predicted string is considered to match the target if the ratio
            is higher or equal to this threshold.

    Returns:
        True if recall is higher or equal to threshold; otherwise return False.
    """
    if pred_str == target_str:
        return True

    if threshold <= 0:
        raise ValueError(
            f"Bag of words string matching threshold must be above 0, but is {threshold}."
        )
    pred_word_set = _word_set(pred_str)
    target_word_set = _word_set(target_str)

    if len(pred_word_set) == 0 and len(target_word_set) == 0:
        return True
    if len(target_word_set) == 0:
        return False

    common_words = target_word_set.intersection(pred_word_set)
    ratio = len(common_words) / len(target_word_set)
    return ratio >= threshold


def _is_arg_value_equal(
    *,
    pred_arg: ValueOrListOf,
    target_arg: ValueOrListOf,
    match_type: ArgumentMatchType,
) -> bool:
    """Checks if the predicted and target arguments are equal under different checks."""
    if match_type == ArgumentMatchType.STRICT:
        return pred_arg == target_arg

    if (
        isinstance(pred_arg, list)
        and isinstance(target_arg, list)
        and len(pred_arg) == len(target_arg)
    ):
        return all(
            _is_arg_value_equal(
                pred_arg=el_pred,
                target_arg=el_target,
                match_type=match_type,
            )
            for el_pred, el_target in zip(pred_arg, target_arg)
        )
    # Only handling string payloads for lenient evaluation.
    if isinstance(pred_arg, str) and isinstance(target_arg, str):
        pred_lenient = _string_lenient_transform(pred_arg)
        target_lenient = _string_lenient_transform(target_arg)
        if match_type == ArgumentMatchType.LENIENT_BAG_OF_WORD:
            return _match_strings_bag_of_words(
                pred_str=pred_lenient, target_str=target_lenient, threshold=1.0
            )
        return pred_lenient == target_lenient

    return False


def check_arguments(
    *,
    pred_args: dict[str, ValueOrListOf],
    target_args: dict[str, ValueOrListOf],
    match_type: ArgumentMatchType,
) -> bool:
    """Checks if the predicted and targets arguments are matching.

    Args:
        pred_args: The predicted arguments.
        target_args: The target (GT) arguments.
        match_type: The match type.

    Returns:
        True if the predicted and targets arguments are matching according to the flags.
    """
    # Check names are not duplicated.
    target_args_copy = dict(target_args.items())

    # Find different label/value pairs.
    for pred_arg_name in pred_args:
        if pred_arg_name in target_args_copy and _is_arg_value_equal(
            pred_arg=pred_args[pred_arg_name],
            target_arg=target_args[pred_arg_name],
            match_type=match_type,
        ):
            target_args_copy.pop(pred_arg_name)
        else:
            return False

    # If there are still elements in to_kwargs, to_kwargs contains more entries than from_kwargs
    # and the arguments are not matching.
    return len(target_args_copy) == 0
