"""Run time and compile time of the transformations on a fixed set of functions.

Each time is also recorded relative to the plain function on the same inputs, in
`extra_info["ratio_to_plain"]` after `compare.py` has run. That ratio stays comparable
across machines, while the absolute times do not.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import jaxormick as jmk

BATCH_SIZE = 1024
"""Number of boxes in the batched benchmarks."""


def six_hump_camel(x: jax.Array) -> jax.Array:
    """Returns the six-hump camel function in two variables."""
    a, b = x[0], x[1]
    return (4 - 2.1 * a**2 + a**4 / 3) * a**2 + a * b + (-4 + 4 * b**2) * b**2


def exponential_sum(x: jax.Array) -> jax.Array:
    """Returns a weighted sum of 50 exponentials of products of neighboring variables."""
    weights = jnp.linspace(-1.0, 1.0, x.size)
    return jnp.sum(weights * jnp.exp(x * jnp.roll(x, 1)))


def newton_iteration(x: jax.Array) -> jax.Array:
    """Returns the square root of `x[0] + 3` after 20 Newton steps in a `fori_loop`."""
    target = x[0] + 3
    return lax.fori_loop(
        0, 20, lambda step, guess: (guess + target / guess) / 2, jnp.asarray(2.0)
    )


def quadratic_form(x: jax.Array) -> jax.Array:
    """Returns `x · (A x)` in 100 variables."""
    matrix = jnp.asarray(np.random.default_rng(0).normal(size=(x.size, x.size)))
    return x @ (matrix @ x)


def trigonometric_mix(x: jax.Array) -> jax.Array:
    """Returns a mix of `sin`, `cos` and `tanh` in two variables."""
    return jnp.sin(3 * x[0]) * jnp.tanh(x[1]) + jnp.cos(x[0] - 2 * x[1])


FUNCTIONS: dict[str, tuple[Callable[[jax.Array], jax.Array], int]] = {
    "six_hump_camel": (six_hump_camel, 2),
    "exponential_sum": (exponential_sum, 50),
    "newton_iteration": (newton_iteration, 1),
    "quadratic_form": (quadratic_form, 100),
    "trigonometric_mix": (trigonometric_mix, 2),
}
"""Benchmark functions, mapped to the function and its number of variables."""


def plain(function: Callable[[jax.Array], jax.Array]) -> Callable[..., jax.Array]:
    """Returns the function itself, with the arguments of the relaxed transformations."""
    return jax.jit(lambda reference, lower, upper: function(reference))


TRANSFORMS: dict[str, Callable[[Callable[[jax.Array], jax.Array]], Callable]] = {
    "plain": plain,
    "relax": jmk.relax,
    "interval": lambda function: (
        lambda reference, lower, upper: jmk.interval(function)(lower, upper)
    ),
    "convex_side": jmk.convex_side,
    "affine_forward": lambda function: jmk.affine_relaxation(function, "forward"),
    "affine_reverse": lambda function: jmk.affine_relaxation(function, "reverse"),
}
"""Transformations to benchmark, each taking the reference point and the box."""


def box(size: int, batch: int | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns a reference point and a box in `[−1, 1]^size`, optionally batched."""
    rng = np.random.default_rng(1)
    shape = (size,) if batch is None else (batch, size)
    corners = rng.uniform(-1, 1, size=(2, *shape))
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    reference = lower + rng.uniform(size=shape) * (upper - lower)
    reference, lower, upper = (
        jnp.asarray(array, dtype=jnp.float32) for array in (reference, lower, upper)
    )
    return reference, lower, upper


def block(output: object) -> None:
    """Waits until every array in `output` is computed."""
    jax.block_until_ready(output)


@pytest.mark.parametrize("transform", sorted(TRANSFORMS))
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_run_time(benchmark, name, transform):
    """Measures the time per call of a compiled transformation."""
    function, size = FUNCTIONS[name]
    compiled = jax.jit(TRANSFORMS[transform](function))
    arguments = box(size)
    block(compiled(*arguments))
    benchmark.extra_info.update(function=name, transform=transform, kind="run")
    benchmark(lambda: block(compiled(*arguments)))


@pytest.mark.parametrize("transform", sorted(TRANSFORMS))
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_compile_time(benchmark, name, transform):
    """Measures the time to trace and compile a transformation from scratch."""
    function, size = FUNCTIONS[name]
    arguments = box(size)

    def compile_fresh() -> None:
        jax.jit(TRANSFORMS[transform](function)).lower(*arguments).compile()

    benchmark.extra_info.update(function=name, transform=transform, kind="compile")
    benchmark.pedantic(compile_fresh, rounds=3, iterations=1, warmup_rounds=0)


@pytest.mark.parametrize("transform", ["plain", "relax", "affine_forward"])
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_batched_run_time(benchmark, name, transform):
    """Measures the time per call of a transformation batched over many boxes."""
    function, size = FUNCTIONS[name]
    compiled = jax.jit(jax.vmap(TRANSFORMS[transform](function)))
    arguments = box(size, BATCH_SIZE)
    block(compiled(*arguments))
    benchmark.extra_info.update(function=name, transform=transform, kind="batched")
    benchmark(lambda: block(compiled(*arguments)))
