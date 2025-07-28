import os

os.environ["JAX_PLATFORMS"] = "cpu"
import time

import coloredlogs
import jax
import numpy as np
from absl import flags, logging

from axlearn.common.utils import set_data_dir

# coloredlogs.install("INFO", fmt="%(asctime)s %(name)s:%(lineno)s[%(process)d] %(levelname)s %(message)s")
formatter = coloredlogs.ColoredFormatter(
    fmt="%(asctime)s %(filename)s:%(lineno)s[%(process)d] %(levelname)s %(message)s"
)

# logging.get_absl_handler().setFormatter(None)

FLAGS = flags.FLAGS


def _print_stats(res, idx):
    res = np.array(res)
    res = np.diff(res)
    logging.warning(
        f"{idx} batches, per-batch time {np.mean(res) * 1e-6:.2f} ms, std {np.std(res) * 1e-6:.2f} ms"
    )
    res = res[len(res) // 2 :]
    logging.warning(
        f"{idx} batches, per-batch time {np.mean(res) * 1e-6:.2f} ms, std {np.std(res) * 1e-6:.2f} ms"
    )


def benchmark(ds=None, ds_iter=None, max_iters=None):
    if ds_iter is None:
        assert ds is not None
        ds_iter = iter(ds)
    if max_iters is None:
        max_iters = 2**30

    idx = 1
    res = [time.time_ns()]
    while idx <= max_iters:
        next(ds_iter)
        idx += 1
        res.append(time.time_ns())
        if idx % 20 == 0:
            _print_stats(res, idx)
        if max_iters is not None and idx == max_iters:
            break

    _print_stats(res, idx)
    res = res[: len(res) // 2]
    return np.mean(res), np.std(res)


def timed(fn, msg):
    begin = time.time_ns()
    ret = fn()
    end = time.time_ns()
    logging.info(f"{msg}, {(end - begin) * 1e-6:.2f} ms")
    return ret


from ajax.experiments.speech.pretrain.online_pretrain_utils import audio_to_modality_config

if __name__ == "__main__":
    # def main(argv):
    logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().setFormatter(formatter)

    with set_data_dir("/tmp/gcsfuse/tensorflow_datasets"):
        with jax.make_mesh((1, 1, 1, 1), ("data", "expert", "fsdp", "seq")):
            from axlearn.experiments.text.gpt.c4_trainer import named_trainer_configs

            cfg = named_trainer_configs()["fuji-7B-v2-grain"]()

            from ajax.experiments import general_lm
            from jax.sharding import PartitionSpec

            cfg.input.partition_spec = PartitionSpec(
                general_lm.batch_axis_names_from(general_lm.MESH_AXIS_NAMES)
            )
            ds = timed(
                lambda: cfg.input.set(name="input").instantiate(parent=None),
                "initialize",
            )
            # ds_iter = timed(lambda: iter(ds.dataset().parents[0]), "iter")
            ds_iter = timed(lambda: iter(ds), "iter")

        x = next(ds_iter)
        while True:
            benchmark(ds_iter=ds_iter, max_iters=5000)
