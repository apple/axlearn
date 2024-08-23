# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tylin/coco-caption:
# Copyright (c) 2015, Xinlei Chen, Hao Fang, Tsung-Yi Lin, and Ramakrishna Vedantam.
# All rights reserved.
# Licensed under the BSD 2-Clause License.
#
# GT-Vision-Lab/VQA:
# Copyright (c) 2014, Aishwarya Agrawal. All rights reserved.
# Licensed under the BSD 2-Clause License.

"""Visual question answering accuracy between a prediction and ground truth answers.

Original code VQA implementation:
- Tsung-Yi Lin for MSCOCO Python API:
https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py
- Aishwarya Agrawal for VQA API:
https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L112

This evaluation metrics compares a predicted answer to a set of 10 ground truth answers.
All questions are annotated with 10 concise, open-ended answers each.
Annotations on the training and validation sets are publicly available.

To be consistent with different human phrasing (e.g "blue", "blue one", "it's blue", etc.),
a few preprocessings are done, such as converting number, removing articles, etc.

Official result format expected by the VQA API evaluation metrics is:
    results = [result]

    result{
        "answer": str
        "question_id": int,
    }

The overall organisation follows a separation in 3 sets:
    - Questions:
        - e.g. v2_OpenEnded_mscoco_val2014_questions.json
    - Annotations:
        - e.g. v2_mscoco_val2014_annotations.json
    - Results:
        - model predictions

┌─────────────────────┐     ┌──────────────────────────────────────┐     ┌──────────────────────┐
│                     │     │                                      │     │                      │
│  QUESTIONS (List)   │     │  ANNOTATIONS (List)                  │     │  RESULTS (List)      │
│                     │     │                                      │     │                      │
│ * image_id: int     │     │ * answer_type: str                   │     │ * answer: str        │
│                     │     │                                      │     │                      │
│ * question: str     │     │ * multiple_choice_answer: str        │  ┌──┤ * question_id: int   │
│                     │     │                                      │  │  │                      │
│ * question_id: int  ├──┐  │ * answers: List[                     │  │  │                      │
│                     │  │  │                                      │  │  └──────────────────────┘
│                     │  │  │     Dict[ * answer: str              │  │
└─────────────────────┘  │  │                                      │  │
                         │  │           * answer_confidence: str   │  │
                         │  │                                      │  │
                         │  │           * answer_id: int ]         │  │
                         │  │     ]                                │  │
                         │  │ * image_id: int                      │  │
                         │  │                                      │  │
                         │  │ * question_type: str                 │  │
                         │  │                                      │  │
                         └─►│ * question_id: int                   │◄─┘
                            │                                      │
                            │                                      │
                            └──────────────────────────────────────┘

Statistics from VQA validation:
    - 3 answer_type ("number", "yes/no", "other")
    - 65 question_type, determined by the first few tokens of the question (e.g. "is that", etc.)
    - 14k multiple_choice_answer, most frequent ground-truth answer.

Evaluation can be done online, i.e there is no need to process the whole dataset at once.

For details see: https://visualqa.org/evaluation.html
For public leaderboard see: https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278

Original VQA code from:

Reference:
    - Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.
    L., & Parikh, D. (2015). Vqa: Visual question answering. In Proceedings
    of the IEEE international conference on computer vision (pp.
    2425-2433).

"""
import re
from typing import Callable

# fmt: off
EN_VQA_CONTRACTIONS = {
   "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", \
   "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've", \
   "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
   "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",\
   "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",\
   "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive":\
   "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", \
   "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", \
   "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
   "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",\
   "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",\
   "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",\
   "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", \
   "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",\
   "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", \
   "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", \
   "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll", \
   "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
   "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", \
   "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", \
   "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",\
   "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",\
   "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",\
   "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",\
   "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",\
   "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",\
   "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",\
   "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",\
   "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",\
   "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", \
   "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",\
   "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",\
   "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}

EN_VQA_DIGITS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

EN_VQA_ARTICLES = {"a", "an", "the"}
# pylint: disable=anomalous-backslash-in-string
EN_VQA_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
EN_VQA_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
# pylint: enable=anomalous-backslash-in-string

EN_VQA_PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}",
    "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]
# fmt: on


def _en_normalizer(answer: str) -> str:
    def _process_punctuation(answer: str) -> str:
        # https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L130
        normalized_answer = answer
        for p in EN_VQA_PUNCT:
            if (p + " " in answer or " " + p in answer) or (
                re.search(EN_VQA_COMMA_STRIP, answer) is not None
            ):
                normalized_answer = normalized_answer.replace(p, "")
            else:
                normalized_answer = normalized_answer.replace(p, " ")
        normalized_answer = EN_VQA_PERIOD_STRIP.sub("", normalized_answer, re.UNICODE)
        return normalized_answer

    def _process_digit_article(answer: str) -> str:
        # https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L142
        normalized_answer = []
        temp_text = answer.lower().split()
        for word in temp_text:
            word = EN_VQA_DIGITS.setdefault(word, word)

            if word not in EN_VQA_ARTICLES:
                normalized_answer.append(word)
            else:
                pass
        for word_id, word in enumerate(normalized_answer):
            if word in EN_VQA_CONTRACTIONS:
                normalized_answer[word_id] = EN_VQA_CONTRACTIONS[word]
        normalized_answer = " ".join(normalized_answer)
        return normalized_answer

    answer = _process_punctuation(answer)
    answer = _process_digit_article(answer)

    return answer


def _get_preprocessor() -> Callable[[str], str]:
    # https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L94
    def _preprocessor(answer: str):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()

        return answer

    return _preprocessor


def _get_normalizer(lang: str) -> Callable[[str], str]:
    if lang == "en":
        return _en_normalizer
    else:
        raise NotImplementedError(f"Normalizer for {lang} is not implemented.")


def vqa_accuracy_score(*, answer: str, gt_answers: list[str], lang: str = "en") -> float:
    """Computes visual question answering accuracy between a prediction and ground truth answers.

    This evaluation metrics compares a predicted answer to a set of 10 ground truth answers.
        Acc(answer) = min { (# GT answers == answer) / 3, 1 }

    Args:
        answer: The predicted answer string.
        gt_answers: An array of ground truth answers.
        lang: The language of both prediction and ground_truth.

    Returns:
        The accuracy match between the predicted answer and ground_truth answers. Is 1.0 if there
        is an exact match between the two, and 0.0 otherwise.

    Raises:
        ValueError: The number of ground truth answers is zero.
    """
    if len(gt_answers) == 0:
        raise ValueError("The number of ground truth answers must be larger than 0")

    # Preprocessing.
    preprocessor = _get_preprocessor()
    answer = preprocessor(answer)
    gt_answers = [preprocessor(gt_answer) for gt_answer in gt_answers]

    # Normalization only if there is a variation in the GT answers.
    # If all human annotators agree and produce an identical answer, expectation is exact match.
    # References:
    # https://github.com/GT-Vision-Lab/VQA/issues/14
    # https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L98
    normalizer = _get_normalizer(lang=lang)
    if len(set(gt_answers)) > 1:
        answer = normalizer(answer)
        gt_answers = [normalizer(gt_answer) for gt_answer in gt_answers]

    # Computing accuracy.
    all_answer_accuracies = []
    for gt_answer_index, gt_answer in enumerate(gt_answers):
        other_gt_answers = [
            item for index, item in enumerate(gt_answers) if index != gt_answer_index
        ]
        matching_answers = [item for item in other_gt_answers if item == answer]
        per_answer_accuracy = min(1, float(len(matching_answers)) / 3)
        all_answer_accuracies.append(per_answer_accuracy)

    return float(sum(all_answer_accuracies)) / len(all_answer_accuracies)
