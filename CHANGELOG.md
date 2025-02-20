# Change Log

## 0.1.6

* Changes
  * Upgrade Jax from 0.4.37 to 0.4.38

## 0.1.5

* Changes
    * Upgrade Jax from 0.4.33 to 0.4.37.

## 0.1.4

* Changes
    * Upgrade Jax from 0.4.33 to 0.4.34.
    * Updates the `input_base.Input` API to support configuring input partitioning behavior.
    * The config fields `batch_axis_names` and `seq_axis_names` in `causal_lm.Model` are now deprecated. Please use `input_base.Input.input_partitioner` instead.
    * Updates the `causal_lm.Model` API to support configuring metrics without subclassing. This requires a golden config change.

## 0.1.3

* Changes
    * Upgrade Jax from 0.4.30 to 0.4.33.

## 0.1.2

* Changes
    * Upgrade Python to 3.10
    * Fall back to triton backend for qkv in fp32 or with bias on gpu flash attention.

## 0.1.1

* Changes
    * Upgrade Jax from 0.4.28 to 0.4.30.

## 0.1.0 (Aug 22, 2024)

* Changes
    * Add changelog.
