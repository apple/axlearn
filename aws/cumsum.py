import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

@partial(jax.jit, static_argnums=(1, 2))
def cumsum_4d_matmul(x: jnp.ndarray, axis: int = -1, tril_size: int = 2048):
    """Efficient cumsum implementation using triangular matrix multiplication.
    
    Args:
        x: Input array of shape (d0, d1, d2, d3)
        axis: Axis along which to compute cumsum (0-3)
        tril_size: Size of triangular matrix for tiling
    Returns:
        Array of same shape with cumulative sum along specified axis
    """
    assert x.dtype == jnp.int32, f"cumsum_4d_matmul expected int32, got {x.dtype}"

    if axis < 0:
        axis = x.ndim + axis

    # Create triangular matrix once
    tril = jnp.tril(jnp.ones((tril_size, tril_size), dtype=jnp.int32))

    # Move the target axis to first position
    if axis != 0:
        perm = list(range(4))
        perm[0], perm[axis] = perm[axis], perm[0]
        x = jnp.transpose(x, perm)
    # Get shapes
    axis_size = x.shape[0]
    batch_size = np.prod(x.shape[1:])

    # Reshape to 2D for efficient matmul
    x_2d = x.reshape(axis_size, batch_size)

    # Process data based on size
    if axis_size <= tril_size:
        # Single matmul for small sizes
        tril_slice = lax.dynamic_slice(
            tril, 
            (0, 0), 
            (axis_size, axis_size)
        )
        result = jnp.matmul(tril_slice, x_2d, precision=lax.Precision.HIGHEST)
        
    else:
        # Process in tiles
        num_full_tiles = axis_size // tril_size
        remainder_size = axis_size % tril_size

        # Process full tiles with rolling sum
        full_tiles = x_2d[:num_full_tiles * tril_size].reshape(
            num_full_tiles, tril_size, batch_size
        )

        def process_tile(rolling_sum, x_tile):
            output = rolling_sum + jnp.matmul(tril, x_tile, precision=lax.Precision.HIGHEST)
            # Last row becomes new rolling sum
            new_rolling_sum = output[-1:, :]
            return new_rolling_sum, output

        rolling_sum = jnp.zeros((1, batch_size), dtype=full_tiles.dtype)
        last_rolling_sum, results = jax.lax.scan(process_tile, rolling_sum, full_tiles)
        result = results.reshape(-1, batch_size)

        # Process remainder if any
        if remainder_size > 0:
            x_remainder = x_2d[num_full_tiles * tril_size :]
            tril_remainder = tril[:remainder_size, :remainder_size]
            remainder_result = last_rolling_sum + jnp.matmul(
                tril_remainder, x_remainder, precision=lax.Precision.HIGHEST
            )

            # Concatenate remainder
            result = jnp.concatenate([result, remainder_result], axis=0)

    # Reshape back to 4D
    result = result.reshape(x.shape)

    # Transpose back if necessary
    if axis != 0:
        inv_perm = [0] * 4
        for i, p in enumerate(perm):
            inv_perm[p] = i
        result = jnp.transpose(result, inv_perm)

    return result

# Add debug and profiling capabilities
def debug_cumsum_4d(x, axis=-1, tril_size=2048):
    """Debug version with shape and performance logging."""
    import time
    
    print(f"Input shape: {x.shape}")
    print(f"Axis: {axis}")
    print(f"Tril size: {tril_size}")
    
    # Compile first
    cumsum_fn = cumsum_4d_matmul.lower(x, axis, tril_size).compile()
    
    # Time execution
    start = time.time()
    result = cumsum_fn(x)
    result.block_until_ready()
    duration = time.time() - start
    
    print(f"Execution time: {duration*1000:.2f}ms")
    print(f"Output shape: {result.shape}")
    
    # Verify result
    expected = jnp.cumsum(x, axis=axis)
    max_diff = jnp.max(jnp.abs(result - expected))
    print(f"Max difference from jnp.cumsum: {max_diff}")
    
    return result

# More detailed version with additional features
def detailed_array_comparison(a: jnp.ndarray,
                            b: jnp.ndarray,
                            rtol: float = 1e-5,
                            atol: float = 1e-8,
                            name_a: str = "a",
                            name_b: str = "b",
                            max_print: int = 10) -> None:
    """Detailed comparison of two arrays with additional statistics.
    
    Args:
        a, b: Arrays to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        name_a, name_b: Names for the arrays in output
        max_print: Maximum number of differences to print
    """
    a_np = np.array(a)
    b_np = np.array(b)
    
    print(f"\nComparing {name_a} and {name_b}:")
    print(f"Shapes: {a.shape} vs {b.shape}")
    
    if a.shape != b.shape:
        print("Shape mismatch!")
        return
    
    # Basic statistics
    print("\nBasic statistics:")
    print(f"{name_a} - min: {np.min(a_np)}, max: {np.max(a_np)}, "
          f"mean: {np.mean(a_np)}, std: {np.std(a_np)}")
    print(f"{name_b} - min: {np.min(b_np)}, max: {np.max(b_np)}, "
          f"mean: {np.mean(b_np)}, std: {np.std(b_np)}")
    
    # Differences
    abs_diff = np.abs(a_np - b_np)
    rel_diff = np.abs((a_np - b_np) / np.where(b_np != 0, b_np, 1))
    diff_mask = (abs_diff > atol) & (rel_diff > rtol)
    
    print("\nDifference statistics:")
    print(f"Max absolute difference: {np.max(abs_diff)}")
    print(f"Mean absolute difference: {np.mean(abs_diff)}")
    print(f"Max relative difference: {np.max(rel_diff)}")
    print(f"Mean relative difference: {np.mean(rel_diff)}")
    
    # Histogram of differences
    if np.any(diff_mask):
        print("\nHistogram of absolute differences:")
        hist, bins = np.histogram(abs_diff[diff_mask], bins=10)
        for i in range(len(hist)):
            print(f"{bins[i]:.2e} to {bins[i+1]:.2e}: {hist[i]} elements")
        
        # Print largest differences
        print(f"\nTop {min(max_print, np.sum(diff_mask))} largest differences:")
        flat_indices = np.argsort(abs_diff.ravel())[-max_print:]
        indices = np.unravel_index(flat_indices, a.shape)
        
        for idx in zip(*indices):
            print(f"\nIndex {idx}:")
            print(f"  {name_a}: {a_np[idx]:.8f}")
            print(f"  {name_b}: {b_np[idx]:.8f}")
            print(f"  Absolute diff: {abs_diff[idx]:.8f}")
            print(f"  Relative diff: {rel_diff[idx]:.8f}")

# Test function
def test_cumsum_4d():
    # Test cases
    shapes = [
        (2, 3, 4, 5),      # Small
        (16, 32, 6, 8),  # Medium
        (128, 256, 32, 64),  # Large
        (16, 2, 4096, 8),  # Tiled
    ]
    
    for shape in shapes:
        print(f"\nTesting shape: {shape}")
        x = jax.random.randint(jax.random.PRNGKey(0), shape=shape, minval=0, maxval=1000, dtype=jnp.int32)
        
        for axis in range(4):
            print(f"\nAxis {axis}:")
            result = cumsum_4d_matmul(x, axis)
            expected = jnp.cumsum(x, axis)
            detailed_array_comparison(result, expected, atol=1e-7, max_print=100)
            assert jnp.allclose(result, expected, atol=1e-7)

            print("Test passed!")

# Benchmark function
def benchmark_cumsum_4d(shape=(128, 256, 32, 64), num_runs=100):
    """Benchmark against native cumsum."""
    import time
    
    x = jax.random.randint(jax.random.PRNGKey(0), shape=shape, minval=0, maxval=1000, dtype=jnp.int32)
    
    # Compile both implementations
    matmul_fn = cumsum_4d_matmul.lower(x, 0).compile()
    native_fn = jax.jit(lambda x: jnp.cumsum(x, axis=0)).lower(x).compile()
    
    # Warm up
    _ = matmul_fn(x).block_until_ready()
    _ = native_fn(x).block_until_ready()
    
    # Benchmark matmul implementation
    start = time.time()
    for _ in range(num_runs):
        result = matmul_fn(x)
        result.block_until_ready()
    matmul_time = (time.time() - start) / num_runs
    
    # Benchmark native implementation
    start = time.time()
    for _ in range(num_runs):
        result = native_fn(x)
        result.block_until_ready()
    native_time = (time.time() - start) / num_runs
    
    print(f"\nShape: {shape}")
    print(f"Matmul implementation: {matmul_time*1000:.2f}ms")
    print(f"Native implementation: {native_time*1000:.2f}ms")
    print(f"Speedup: {native_time/matmul_time:.2f}x")

@jax.jit
def test_cumsum_usage():
    key = jax.random.PRNGKey(0)
    logits = jax.random.uniform(key, shape=(1, 2, 256, 8))
    raw_gates = jax.nn.softmax(logits, axis=-1)  # along E dim
    # top-1 index: OGS tensor.
    index_1 = jnp.argmax(raw_gates, axis=-1)
    # OGSE tensor.
    mask_1 = jax.nn.one_hot(index_1, raw_gates.shape[-1], dtype=jnp.int32)
    position_in_expert_1 = _cum_sum(mask_1, exclusive=True, axis=-2)
    return position_in_expert_1

if __name__ == "__main__":
    # Run tests
    test_cumsum_4d()
    
    # Run benchmarks
    shapes = [
        (16, 32, 64, 128),
        (128, 256, 32, 64),
        (512, 512, 16, 16),
        (16, 2, 4096, 8),
    ]
    
    for shape in shapes:
        benchmark_cumsum_4d(shape)