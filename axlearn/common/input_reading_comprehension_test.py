# Copyright © 2023 Apple Inc.

"""Tests reading comprehension inputs."""

import json
import os
from typing import Callable, Union

import pytest
from absl.testing import parameterized
from transformers import RobertaTokenizer

from axlearn.common.config import config_for_class, config_for_function
from axlearn.common.input_reading_comprehension import (
    HFTokenizerForReadingComprehension,
    TokenizerForReadingComprehension,
    build_example_dataset_fn,
    convert_example_to_features,
    examples_to_jsonl_dataset,
    parse_examples_jsonl,
)
from axlearn.common.input_test_utils import BPE_DIR as _BPE_DIR
from axlearn.common.input_test_utils import roberta_base_merges_file as _ROBERTA_BASE_MERGES_FILE
from axlearn.common.input_test_utils import roberta_base_vocab_file as _ROBERTA_BASE_VOCAB_FILE


def build_hf_roberta_tokenizer() -> RobertaTokenizer:
    # pylint: disable=duplicate-code
    # TODO(@zhucheng_tu): Consider switching to from_pretrained().
    tokenizer = config_for_class(RobertaTokenizer).set(
        vocab_file=_ROBERTA_BASE_VOCAB_FILE,
        merges_file=_ROBERTA_BASE_MERGES_FILE,
        # We set add_prefix_space=True for compatibility with production CGO/Rust tokenizer.
        add_prefix_space=True,
        kwargs={},
    )
    # pylint: enable=duplicate-code

    tokenizer = tokenizer.instantiate()
    # When creating HF PreTrainedTokenizer using its constructor instead of the
    # from_pretrained() factory, special tokens are not registered with the tokenizer.
    # We need to call sanitize_special_tokens() explicitly to register the special tokens,
    # just like what happens in from_pretrained().
    # https://github.com/huggingface/transformers/blob/ca485e562b675341409e3e27724072fb11e10af7/src/transformers/tokenization_utils_base.py#L2003
    tokenizer.sanitize_special_tokens()

    return tokenizer


def dummy_reading_comprehension_inputs_simple() -> (
    list[dict[str, Union[str, list[dict[str, Union[str, int]]]]]]
):
    reading_comprehension_inputs = [
        dict(
            context="Dogs are small",
            question="What are small?",
            answers=[
                dict(
                    text="Dogs",
                    answer_start=0,
                ),
            ],
        ),
    ]
    return reading_comprehension_inputs


def dummy_reading_comprehension_inputs_simple_multiple_word_answer() -> (
    list[dict[str, Union[str, list[dict[str, Union[str, int]]]]]]
):
    reading_comprehension_inputs = [
        dict(
            context="All Siberian Huskies are energetic",
            question="What are energetic?",
            answers=[
                dict(
                    text="Siberian Huskies",
                    answer_start=4,
                ),
            ],
        ),
    ]
    return reading_comprehension_inputs


def dummy_reading_comprehension_inputs_simple_answer_end() -> (
    list[dict[str, Union[str, list[dict[str, Union[str, int]]]]]]
):
    reading_comprehension_inputs = [
        dict(
            context="Dogs are small",
            question="What are dogs?",
            answers=[
                dict(
                    text="small",
                    answer_start=9,
                ),
            ],
        ),
    ]
    return reading_comprehension_inputs


def dummy_reading_comprehension_inputs_multiple_answers() -> (
    list[dict[str, Union[str, list[dict[str, Union[str, int]]]]]]
):
    reading_comprehension_inputs = [
        dict(
            context="Dogs are small. Cats are small too.",
            question="What are small?",
            answers=[
                dict(
                    text="Dogs",
                    answer_start=0,
                ),
                dict(
                    text="Cats",
                    answer_start=16,
                ),
            ],
        ),
    ]
    return reading_comprehension_inputs


def dummy_reading_comprehension_inputs_no_answer() -> (
    list[dict[str, Union[str, list[dict[str, Union[str, int]]]]]]
):
    reading_comprehension_inputs = [
        dict(
            context="Dogs are small.",
            question="What are large?",
            answers=[],
        ),
    ]
    return reading_comprehension_inputs


def dummy_reading_comprehension_inputs_multiple_squad() -> (
    list[dict[str, Union[str, list[dict[str, Union[str, int]]]]]]
):
    reading_comprehension_inputs = [
        # Example taken from training set of SQuAD2.0 (row 2018).
        # https://rajpurkar.github.io/SQuAD-explorer/
        dict(
            context="iPods have also gained popularity for use in education. "
            "Apple offers more information on educational uses for iPods on their website, "
            "including a collection of lesson plans. There has also been academic research "
            "done in this area in nursing education and more general K-16 education. "
            "Duke University provided iPods to all incoming freshmen in the fall of 2004, "
            "and the iPod program continues today with modifications. Entertainment Weekly "
            'put it on its end-of-the-decade, "best-of" list, saying, '
            '"Yes, children, there really was a time when we roamed the earth without '
            'thousands of our favorite jams tucked comfortably into our hip pockets. Weird."',
            question="Which major university began issuing iPods to all incoming freshmen "
            "starting in 2004?",
            answers=[
                dict(
                    text="Duke",
                    answer_start=284,
                ),
            ],
        ),
        dict(
            context="The iPod has also been credited with accelerating shifts within the "
            "music industry. The iPod's popularization of digital music storage allows users "
            "to abandon listening to entire albums and instead be able to choose specific "
            "singles which hastened the end of the Album Era in popular music.",
            question="What period of music did the iPod help bring to a close?",
            answers=[
                dict(
                    text="the Album Era",
                    answer_start=259,
                ),
            ],
        ),
    ]
    return reading_comprehension_inputs


def dummy_reading_comprehension_inputs_long() -> (
    list[dict[str, Union[str, list[dict[str, Union[str, int]]]]]]
):
    reading_comprehension_inputs = [
        # Example taken from Wikipedia.
        # https://en.wikipedia.org/wiki/Apple_Inc.
        dict(
            context="Apple Inc. is an American multinational technology company that specializes "
            "in consumer electronics, software and online services headquartered in Cupertino, "
            "California, United States. Apple is the largest technology company by revenue "
            "(totaling US$365.8 billion in 2021) and, as of June 2022, is the world's "
            "biggest company by market capitalization, the fourth-largest personal computer "
            "vendor by unit sales and second-largest mobile phone manufacturer. It is one of "
            "the Big Five American information technology companies, alongside Alphabet, "
            "Amazon, Meta, and Microsoft. Apple was founded as Apple Computer Company on "
            "April 1, 1976, by Steve Jobs, Steve Wozniak and Ronald Wayne to develop and "
            "sell Wozniak's Apple I personal computer. It was incorporated by Jobs and "
            "Wozniak as Apple Computer, Inc. in 1977 and the company's next computer, the "
            "Apple II, became a best seller and one of the first mass-produced "
            "microcomputers. Apple went public in 1980 to instant financial success. The "
            "company developed computers featuring innovative graphical user interfaces, "
            "including the 1984 original Macintosh, announced that year in a critically "
            "acclaimed advertisement. By 1985, the high cost of its products and power "
            "struggles between executives caused problems. Wozniak stepped back from Apple "
            "amicably and pursued other ventures, while Jobs resigned bitterly and founded "
            "NeXT, taking some Apple employees with him. As the market for personal "
            "computers expanded and evolved throughout the 1990s, Apple lost considerable "
            "market share to the lower-priced duopoly of the Microsoft Windows operating "
            "system on Intel-powered PC clones (also known as 'Wintel'). In 1997, weeks away "
            "from bankruptcy, the company bought NeXT to resolve Apple's unsuccessful "
            "operating system strategy and entice Jobs back to the company. Over the next "
            "decade, Jobs guided Apple back to profitability through a number of tactics "
            "including introducing the iMac, iPod, iPhone and iPad to critical acclaim, "
            "launching 'Think different' and other memorable advertising campaigns, opening "
            "the Apple Store retail chain, and acquiring numerous companies to broaden the "
            "company's product portfolio. When Jobs resigned in 2011 for health reasons, "
            "and died two months later, he was succeeded as CEO by Tim Cook. Apple became "
            "the first publicly traded U.S. company to be valued at over $1 trillion in "
            "August 2018, then $2 trillion in August 2020, and most recently $3 trillion in "
            "January 2022. The company receives criticism regarding the labor practices of "
            "its contractors, its environmental practices, and its business ethics, "
            "including anti-competitive practices and materials sourcing. Nevertheless, the "
            "company has a large following and enjoys a high level of brand loyalty. It is "
            "ranked as one of the world's most valuable brands.",
            question="Where is Apple headquarters?",
            answers=[
                dict(
                    text="Cupertino, California, United States",
                    answer_start=147,
                ),
            ],
        ),
    ]
    return reading_comprehension_inputs


def find_start_of_document_index(
    tokenizer: TokenizerForReadingComprehension, input_ids: list[int]
) -> int:
    # Find the first sep token.
    index = 0
    while input_ids[index] != tokenizer.sep_token_id:
        index += 1
    # Move the index past the sep tokens.
    return index + tokenizer.num_sep_between_pair


class InputReadingComprehensionTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            "dataset_function": dummy_reading_comprehension_inputs_simple,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_multiple_answers,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_multiple_squad,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_long,
        },
    )
    def test_build_example_dataset_fn(self, dataset_function: Callable):
        dataset = dataset_function()
        ds = build_example_dataset_fn(examples=dataset)()
        ds = list(ds)

        self.assertEqual(len(ds), len(dataset))

        for ds_row, dataset_item in zip(ds, dataset):
            self.assertEqual(
                ds_row["context"].numpy().item().decode("utf-8"), dataset_item["context"]
            )
            self.assertEqual(
                ds_row["question"].numpy().item().decode("utf-8"), dataset_item["question"]
            )
            for i in range(len(dataset_item["answers"])):
                self.assertEqual(
                    ds_row["answer_texts"].numpy().tolist()[i].decode("utf-8"),
                    dataset_item["answers"][i]["text"],
                )
                self.assertEqual(
                    ds_row["answer_start_positions"].numpy().tolist()[i],
                    dataset_item["answers"][i]["answer_start"],
                )

    @parameterized.parameters(
        {
            "dataset_function": dummy_reading_comprehension_inputs_simple,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_multiple_answers,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_no_answer,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_multiple_squad,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_long,
        },
    )
    def test_parse_examples_jsonl(self, dataset_function: Callable):
        dataset = dataset_function()
        dataset_jsonl = examples_to_jsonl_dataset(examples=dataset)()

        ds = parse_examples_jsonl()(dataset_jsonl)
        ds = list(ds)

        self.assertEqual(len(ds), len(dataset))

        for ds_row, dataset_item in zip(ds, dataset):
            self.assertEqual(
                ds_row["context"].numpy().item().decode("utf-8"), dataset_item["context"]
            )
            self.assertEqual(
                ds_row["question"].numpy().item().decode("utf-8"), dataset_item["question"]
            )
            for i in range(len(dataset_item["answers"])):
                self.assertEqual(
                    ds_row["answer_texts"].numpy().tolist()[i].decode("utf-8"),
                    dataset_item["answers"][i]["text"],
                )
                self.assertEqual(
                    ds_row["answer_start_positions"].numpy().tolist()[i],
                    dataset_item["answers"][i]["answer_start"],
                )

    @parameterized.parameters(
        {
            "dataset_function": dummy_reading_comprehension_inputs_simple,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_multiple_answers,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_multiple_squad,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_long,
        },
    )
    def test_examples_to_jsonl_dataset(self, dataset_function: Callable):
        dataset = dataset_function()
        dataset_jsonl = examples_to_jsonl_dataset(examples=dataset)()

        ds = list(dataset_jsonl)

        self.assertEqual(len(ds), len(dataset))

        for ds_row, dataset_item in zip(ds, dataset):
            self.assertEqual(ds_row, json.dumps(dataset_item))

    @parameterized.parameters(
        {
            "dataset_function": dummy_reading_comprehension_inputs_simple,
            "expected_tokens": [
                [
                    "<s>",
                    "ĠWhat",
                    "Ġare",
                    "Ġsmall",
                    "?",
                    "</s>",
                    "</s>",
                    "ĠDogs",
                    "Ġare",
                    "Ġsmall",
                    "</s>",
                ]
            ],
            "expected_token_type_ids": [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                ]
            ],
            "max_length": 512,
            "doc_stride": 128,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_simple_multiple_word_answer,
            "expected_tokens": [
                [
                    "<s>",
                    "ĠWhat",
                    "Ġare",
                    "Ġenergetic",
                    "?",
                    "</s>",
                    "</s>",
                    "ĠAll",
                    "ĠSiberian",
                    "ĠHus",
                    "kies",
                    "Ġare",
                    "Ġenergetic",
                    "</s>",
                ]
            ],
            "expected_token_type_ids": [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]
            ],
            "max_length": 512,
            "doc_stride": 128,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_simple_answer_end,
            "expected_tokens": [
                [
                    "<s>",
                    "ĠWhat",
                    "Ġare",
                    "Ġdogs",
                    "?",
                    "</s>",
                    "</s>",
                    "ĠDogs",
                    "Ġare",
                    "Ġsmall",
                    "</s>",
                ]
            ],
            "expected_token_type_ids": [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                ]
            ],
            "max_length": 512,
            "doc_stride": 128,
        },
    )
    @pytest.mark.skipif(not os.path.exists(_BPE_DIR), reason="Missing testdata.")
    def test_convert_example_to_features_outputs(
        self,
        dataset_function: Callable,
        expected_tokens: list[list[str]],
        expected_token_type_ids: list[list[int]],
        max_length: int,
        doc_stride: int,
    ):
        tokenizer = (
            HFTokenizerForReadingComprehension.default_config()
            .set(tokenizer=config_for_function(build_hf_roberta_tokenizer))
            .instantiate()
        )

        dataset = dataset_function()

        ds = build_example_dataset_fn(examples=dataset)()
        ds = convert_example_to_features(
            tokenizer,
            max_length=max_length,
            doc_stride=doc_stride,
            is_training=True,
        )(ds)

        rows = list(ds)

        self.assertEqual(len(rows), len(expected_tokens))

        for i, (row, dataset_row) in enumerate(zip(rows, dataset)):
            self.assertCountEqual(
                list(row.keys()),
                ["input_ids", "token_type_ids", "start_positions", "end_positions"],
            )

            padding_length = max_length - len(expected_tokens[i])
            # pylint: disable-next=protected-access
            padded_expected_ids = tokenizer._tokenizer.convert_tokens_to_ids(
                expected_tokens[i][:max_length]
            ) + ([tokenizer.pad_token_id] * padding_length)
            padded_token_type_ids = expected_token_type_ids[i][:max_length] + ([0] * padding_length)

            input_ids = row["input_ids"].numpy().tolist()
            self.assertSequenceEqual(input_ids, padded_expected_ids)
            self.assertSequenceEqual(row["token_type_ids"].numpy().tolist(), padded_token_type_ids)
            self.assertEqual(
                input_ids[row["start_positions"] : row["end_positions"] + 1],
                tokenizer.encode(tokenizer.subtokenize(" " + dataset_row["answers"][0]["text"])),
            )

    @parameterized.parameters(
        {
            "dataset_function": dummy_reading_comprehension_inputs_simple,
            "expected_length": 1,
            "max_length": 512,
            "doc_stride": 128,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_multiple_squad,
            "expected_length": 2,
            "max_length": 512,
            "doc_stride": 128,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_long,
            "expected_length": 2,
            "max_length": 512,
            "doc_stride": 128,
        },
    )
    @pytest.mark.skipif(not os.path.exists(_BPE_DIR), reason="Missing testdata.")
    def test_convert_example_to_features_count(
        self,
        dataset_function: Callable,
        expected_length: int,
        max_length: int,
        doc_stride: int,
    ):
        ds = build_example_dataset_fn(examples=dataset_function())()
        ds = convert_example_to_features(
            HFTokenizerForReadingComprehension.default_config()
            .set(tokenizer=config_for_function(build_hf_roberta_tokenizer))
            .instantiate(),
            max_length=max_length,
            doc_stride=doc_stride,
            is_training=True,
        )(ds)

        self.assertEqual(len(list(ds)), expected_length)

    @parameterized.parameters(
        {
            "dataset_function": dummy_reading_comprehension_inputs_long,
            "max_length": 100,
            "doc_stride": 10,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_long,
            "max_length": 512,
            "doc_stride": 128,
        },
        {
            "dataset_function": dummy_reading_comprehension_inputs_long,
            "max_length": 256,
            "doc_stride": 128,
        },
    )
    @pytest.mark.skipif(not os.path.exists(_BPE_DIR), reason="Missing testdata.")
    def test_convert_example_to_features_doc_stride_overlap(
        self,
        dataset_function: Callable,
        max_length: int,
        doc_stride: int,
    ):
        # Only tests the first example of the dataset function (to ensure all rows have the same
        # question).
        tokenizer = (
            HFTokenizerForReadingComprehension.default_config()
            .set(tokenizer=config_for_function(build_hf_roberta_tokenizer))
            .instantiate()
        )

        ds = build_example_dataset_fn(examples=dataset_function()[0:1])()
        ds = convert_example_to_features(
            tokenizer,
            max_length=max_length,
            doc_stride=doc_stride,
            is_training=True,
        )(ds)

        rows = list(ds)

        doc_start = find_start_of_document_index(tokenizer, rows[0]["input_ids"])

        for i in range(len(rows) - 1):
            # Make sure the part of the context after the doc_stride in the current context appears
            # at the beginning of the next context. The 1 offset is to exclude the EOS token.
            self.assertListEqual(
                rows[i]["input_ids"].numpy().tolist()[doc_start + doc_stride : -1],
                rows[i + 1]["input_ids"].numpy().tolist()[doc_start : -(doc_stride + 1)],
            )

            if tokenizer.cls_token_id not in (
                rows[i]["start_positions"],
                rows[i + 1]["start_positions"],
            ):
                # If consecutive chunks have the answer, make sure that the offset between the
                # answer in both chunks is doc_stride.
                self.assertEqual(
                    rows[i]["start_positions"] + doc_stride, rows[i + 1]["start_positions"]
                )
                self.assertEqual(
                    rows[i]["end_positions"] + doc_stride, rows[i + 1]["end_positions"]
                )

    @pytest.mark.skipif(not os.path.exists(_BPE_DIR), reason="Missing testdata.")
    def test_hf_tokenizer_for_rc(self):
        hf_tokenizer = build_hf_roberta_tokenizer()
        wrapper = (
            HFTokenizerForReadingComprehension.default_config()
            .set(tokenizer=config_for_function(build_hf_roberta_tokenizer))
            .instantiate()
        )

        self.assertEqual(wrapper.sep_token_id, hf_tokenizer.sep_token_id)
        self.assertEqual(wrapper.pad_token_id, hf_tokenizer.pad_token_id)
        self.assertEqual(wrapper.cls_token_id, hf_tokenizer.cls_token_id)
        self.assertEqual(wrapper.eos_token_id, hf_tokenizer.eos_token_id)
        self.assertEqual(
            wrapper.num_sep_between_pair,
            hf_tokenizer.num_special_tokens_to_add(pair=True)
            - hf_tokenizer.num_special_tokens_to_add(),
        )
        self.assertListEqual(
            wrapper.subtokenize("Hello world!"), hf_tokenizer.tokenize("Hello world!")
        )
        self.assertListEqual(
            wrapper.encode(wrapper.subtokenize("Hello world!")),
            hf_tokenizer.encode(hf_tokenizer.tokenize("Hello world!"), add_special_tokens=False),
        )
        self.assertListEqual(
            wrapper("Hello world!"),
            hf_tokenizer.encode(hf_tokenizer.tokenize("Hello world!"), add_special_tokens=False),
        )
