"""Tests loss metrics."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.config import REQUIRED
from axlearn.common.embedding import ModalityVocabInfo
from axlearn.common.loss_metrics import BaseLossMetrics, ModalityLossMetrics, filter_module_outputs
from axlearn.common.metrics import WeightedSummary
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase


class DummyLossMetrics(BaseLossMetrics):
    """Dummy metrics that returns input batch and adds summaries."""

    def forward(self, input_batch, **kwargs):
        del kwargs
        self.add_summary(f"{self.name}_metric", 0)
        return WeightedSummary(0, 1), input_batch


class ModalityLossMetricsTest(TestCase):
    """Tests ModalityLossMetrics."""

    def test_forward(self):
        batch_size, seq_len, vocab_size = 2, 20, 10
        live_targets_weight = 0.5

        cfg: ModalityLossMetrics.Config = ModalityLossMetrics.default_config().set(
            name="test",
            inner=DummyLossMetrics.default_config(),
        )
        layer: ModalityLossMetrics = cfg.instantiate(parent=None)

        assert seq_len % 2 == 0
        tgt_key, live_key, img_key, tgt_pad_key, forward_key = jax.random.split(
            jax.random.PRNGKey(123), num=5
        )
        target_labels = jax.random.randint(
            tgt_key, shape=[batch_size, seq_len], minval=0, maxval=vocab_size
        )
        live_targets = (
            jax.random.randint(live_key, shape=[batch_size, seq_len], minval=0, maxval=2)
            * live_targets_weight
        )

        # Sets a number of target_labels to 10, corresponing to image.
        emb_target_labels = jnp.where(
            jax.random.randint(img_key, shape=target_labels.shape, minval=0, maxval=2),
            10,
            target_labels,
        )
        # Sets some number of target_labels to -1, corresponing to padding.
        input_target_labels = jnp.where(
            jax.random.randint(tgt_pad_key, shape=target_labels.shape, minval=0, maxval=2),
            -1,
            target_labels,
        )

        def forward(layer, input_batch):
            inputs = dict(
                input_batch=input_batch,
                # target_labels comes from embedding layer and may not have paddings masked.
                # target_labels can also be nested arbitrarily.
                module_outputs=dict(nested=dict(target_labels=emb_target_labels)),
                predict_outputs={},  # Unused.
            )
            (_, out), _ = F(layer, forward_key, state={}, inputs=inputs, is_training=True)
            assert "test" not in out  # Metrics should not be nested.
            return out

        # Tests that layer expects target_labels.
        with self.assertRaisesRegex(ValueError, "target_labels"):
            forward(layer, dict(live_targets=live_targets))

        input_batch = dict(target_labels=input_target_labels)
        expect_batch = dict(target_labels=emb_target_labels)

        # Tests without live targets.
        self.assertNestedEqual(
            {**expect_batch, "live_targets": input_target_labels >= 0}, forward(layer, input_batch)
        )

        # Update input batch with live targets.
        input_batch["live_targets"] = live_targets
        expect_batch["live_targets"] = live_targets

        # Tests with live targets. It should be combined with out-of-range target_labels mask if no
        # modality_vocab_info is configured.
        self.assertNestedEqual(
            {
                **expect_batch,
                "live_targets": live_targets * (input_target_labels >= 0),
            },
            forward(layer, input_batch),
        )

        # Tests that live_targets is combined with modality mask if modality_vocab_info
        # is configured.
        def check_input_batch(start, end, expected):
            modality_vocab_info = ModalityVocabInfo(  # NOTE: Only placeholders are used.
                vocab_start=-1,
                vocab_end=-1,
                placeholder_start=start,
                placeholder_end=end,
                modality_name="image",
            )
            layer = cfg.set(modality_vocab_info=modality_vocab_info).instantiate(parent=None)
            self.assertNestedEqual(expected, forward(layer, input_batch))

        # 1. Everything is masked since modality_id is outside of target_labels.
        check_input_batch(
            start=vocab_size,
            end=vocab_size + 1,
            expected={
                **expect_batch,
                "live_targets": jnp.zeros([batch_size, seq_len], dtype=jnp.float32),
            },
        )

        # 2. live_targets is merged with modality mask and out-of-range mask.
        check_input_batch(
            start=0,
            end=vocab_size // 2,
            expected={
                **expect_batch,
                "live_targets": live_targets
                * jnp.logical_and(
                    0 <= input_target_labels, input_target_labels < vocab_size // 2
                ).astype(jnp.float32),
            },
        )

    def test_summaries(self):
        cfg = ModalityLossMetrics.default_config().set(
            name="test",
            inner=DummyLossMetrics.default_config(),
        )
        layer: ModalityLossMetrics = cfg.instantiate(parent=None)

        input_batch = dict(target_labels=jnp.array([1]), live_targets=jnp.array([1]))
        module_outputs = dict(target_labels=jnp.array([2]))
        inputs = dict(input_batch=input_batch, predict_outputs={}, module_outputs=module_outputs)
        (_, outputs), oc = F(layer, None, state={}, inputs=inputs, is_training=True)
        self.assertNestedEqual(oc.summaries, {"inner": {"inner_metric": 0}})
        self.assertNotIn("test", outputs)


class UtilsTest(TestCase):
    @parameterized.parameters(
        # Tests basic key lookup.
        dict(
            expected={"target_labels": 123},
            module_outputs={"nested": {"target_labels": 123}},
            path_regex="(?:.*/?)(target_labels)",
        ),
        # Tests basic nested dict lookup.
        dict(
            expected={"output1": 123, "output2": jnp.array([1, 2, 3])},
            module_outputs={
                "inner": {"tokenizer_output": {"output1": 123, "output2": jnp.array([1, 2, 3])}}
            },
            path_regex="(?:.*/?)tokenizer_output/(.*)",
        ),
        # Tests selecting subkeys of nested dict.
        dict(
            expected={"tokenizer_output": {"output1": 1, "output3": 3}},
            module_outputs={
                "inner": {"tokenizer_output": {"output1": 1, "output2": 2, "output3": 3}}
            },
            path_regex="(?:.*/?)(tokenizer_output/(?:output1|output3))",
        ),
        # Tests default value.
        dict(
            expected={"target_labels": 0},
            module_outputs={},
            path_regex="(?:.*/?)(target_labels)",
            default={"target_labels": 0},
        ),
        # Tests raises when missing without default.
        dict(
            expected=ValueError("No paths matched"),
            module_outputs={},
            path_regex="(?:.*/?)(target_labels)",
        ),
        # Tests multiple capture groups.
        dict(
            expected=ValueError("matching group"),
            module_outputs={"inner": {"target_labels": 123}},
            path_regex="(.*/)(target_labels)",
        ),
        # Tests missing capture group.
        dict(
            expected=ValueError("matching group"),
            module_outputs={"inner": {"target_labels": 123}},
            path_regex="(?:.*/?)target_labels",
        ),
        # Tests key conflict.
        dict(
            expected=ValueError("Multiple paths"),
            module_outputs={"inner": {"target_labels": 123}, "other": {"target_labels": 234}},
            path_regex="(?:.*/?)(target_labels)",
        ),
    )
    def test_filter_module_outputs(self, expected, module_outputs, path_regex, default=REQUIRED):
        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                filter_module_outputs(module_outputs, path_regex=path_regex, default=default)
        else:
            self.assertNestedEqual(
                expected,
                filter_module_outputs(module_outputs, path_regex=path_regex, default=default),
            )


if __name__ == "__main__":
    absltest.main()
