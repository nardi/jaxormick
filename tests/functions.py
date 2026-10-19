"""Test functions on the box `[−2, 2]²`, each defined at every point of that box."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

DOMAIN_LOWER = -2.0
"""Lower bound of every coordinate of the test domain."""

DOMAIN_UPPER = 2.0
"""Upper bound of every coordinate of the test domain."""


def six_hump_camel(x: jax.Array) -> jax.Array:
    """Returns the six-hump camel function, whose global minimum is about −1.0316."""
    a, b = x[0], x[1]
    return (4 - 2.1 * a**2 + a**4 / 3) * a**2 + a * b + (-4 + 4 * b**2) * b**2


def newton_sqrt(x: jax.Array) -> jax.Array:
    """Returns the square root of `x[0] + 3` after five Newton steps from 2.

    The carry of the loop starts as a constant and becomes relaxed in the first step.
    """
    target = x[0] + 3
    return lax.fori_loop(
        0, 5, lambda step, guess: (guess + target / guess) / 2, jnp.asarray(2.0)
    )


def scanned_sum(x: jax.Array) -> jax.Array:
    """Returns a sum built by a reversed `scan` with relaxed stacked outputs."""

    def step(total: jax.Array, term: jax.Array) -> tuple[jax.Array, jax.Array]:
        return total + jnp.exp(term), total * term

    weights = jnp.array([0.5, -1.0, 0.25])
    total, products = lax.scan(step, x[1], weights * x[0], reverse=True)
    return total + jnp.sum(products)


def constant_branch(x: jax.Array) -> jax.Array:
    """Returns a `cond` whose branch index is known while tracing."""
    return lax.cond(True, lambda y: y[0] * y[1], lambda y: y[0] - y[1], x)


def jitted_inner(x: jax.Array) -> jax.Array:
    """Returns a product inside a nested `jax.jit` call."""
    return jax.jit(lambda y: y[0] * jnp.exp(y[1]))(x)


def checkpointed(x: jax.Array) -> jax.Array:
    """Returns a product inside a `jax.checkpoint` call."""
    return jax.checkpoint(lambda y: y[0] * jnp.exp(y[1]))(x)


def quadratic_form(x: jax.Array) -> jax.Array:
    """Returns `x · (A x)` with a dot product of two relaxed operands."""
    matrix = jnp.array([[2.0, -1.0], [0.5, 1.0]])
    return x @ (matrix @ x)


TEST_FUNCTIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "product": lambda x: x[0] * x[1],
    "exp_minus_square": lambda x: jnp.exp(x[0]) - x[1] ** 2,
    "log_times": lambda x: jnp.log(x[0] + 3) * x[1],
    "six_hump_camel": six_hump_camel,
    "cubic": lambda x: x[0] ** 3 - 2 * x[0],
    "fifth_power": lambda x: (x[0] - 0.5 * x[1]) ** 5,
    "reciprocal_and_sqrt": lambda x: 1 / (x[0] + 3) + jnp.sqrt(x[1] + 3),
    "dot_times": lambda x: jnp.dot(jnp.array([1.0, -2.0]), x) * x[0],
    "quadratic_form": quadratic_form,
    "linear_map": lambda x: jnp.sum((x @ jnp.array([[1.0, -2.0], [0.5, 3.0]])) * x),
    "sum_exp": lambda x: jnp.sum(jnp.exp(x)),
    "quotient": lambda x: x[0] / (x[1] + 4),
    "negative_power": lambda x: (x[0] + 3) ** -2,
    "square": lambda x: jnp.square(x[0] - x[1]),
    "newton_sqrt": newton_sqrt,
    "scanned_sum": scanned_sum,
    "constant_branch": constant_branch,
    "jitted_inner": jitted_inner,
    "checkpointed": checkpointed,
    "power_from_zero": lambda x: (x[0] + 2) ** 1.5 + x[1],
    "root_like_power": lambda x: (x[1] + 2.5) ** 0.3 * x[0],
    "negative_fractional_power": lambda x: (x[0] + 3) ** -1.7,
    "float_square": lambda x: (x[0] - x[1]) ** 2.0,
    "log_sum": lambda x: jnp.log(x[0] + x[1] + 4.5),
    "sqrt_from_zero": lambda x: jnp.sqrt(x[0] + 2) - x[1] ** 2,
    "log10": lambda x: jnp.log10(x[0] + 3) * x[1],
    "absolute": lambda x: jnp.abs(x[0] - 0.5) * x[1],
    "expm1_log1p": lambda x: jnp.expm1(x[0]) - jnp.log1p(x[1] + 1.5),
    "rsqrt": lambda x: lax.rsqrt(x[0] + 2.5) * x[1],
    "tanh": lambda x: jnp.tanh(2 * x[0] - x[1]),
    "sigmoid": lambda x: jax.nn.sigmoid(3 * x[0]) * x[1],
    "sin_cos": lambda x: jnp.sin(3 * x[0]) + jnp.cos(2 * x[1] + x[0]),
    "wide_cos": lambda x: jnp.cos(5 * x[0] * x[1]),
    "maximum": lambda x: jnp.maximum(x[0], x[1] ** 2 - 1),
    "minimum": lambda x: jnp.minimum(jnp.exp(x[0]), 2 - x[1]),
    "relu": lambda x: jax.nn.relu(x[0] - x[1]),
    "softplus": lambda x: jax.nn.softplus(x[0] * x[1]),
    "clip": lambda x: jnp.clip(x[0] + x[1], -1.0, 0.5),
    "stacked": lambda x: jnp.stack([x[0] ** 2, jnp.ones(()), x[1]]) @ jnp.arange(3.0),
    "atan": lambda x: jnp.arctan(2 * x[0] - x[1]),
    "asin_acos": lambda x: jnp.arcsin(x[0] / 2.5) + jnp.arccos(x[1] / 2.5),
    "sinh_cosh": lambda x: jnp.sinh(x[0]) - jnp.cosh(x[1] - 0.5),
    "asinh_acosh": lambda x: jnp.arcsinh(3 * x[0]) * jnp.arccosh(x[1] + 3.5),
    "atanh": lambda x: jnp.arctanh((x[0] + x[1]) / 4.5),
    "tan": lambda x: jnp.tan(0.35 * x[0] + 0.3 * x[1]),
    "shifted_tan": lambda x: jnp.tan(0.2 * x[0] + 3.0),
    "cbrt": lambda x: jnp.cbrt(x[0] - 0.5 * x[1]),
    "exp2_log2": lambda x: jnp.exp2(x[0]) + jnp.log2(x[1] + 3),
    "erf_erfc": lambda x: (
        jax.scipy.special.erf(2 * x[0]) - jax.scipy.special.erfc(x[1])
    ),
    "erf_inv": lambda x: jax.scipy.special.erfinv(x[0] / 2.5) * x[1],
    "reduce_max_min": lambda x: (
        jnp.max(jnp.stack([x[0] ** 2, x[1], -x[0] * x[1]]))
        - jnp.min(jnp.array([[x[0], 1.0], [x[1] ** 3, x[0] + x[1]]]), axis=1).sum()
        + jnp.max(x[None, :1], axis=(0, 1))
    ),
    "reduce_prod": lambda x: jnp.prod(jnp.stack([x[0], x[1] + 3, x[0] - x[1], x[1]])),
    "scatter_set": lambda x: (
        jnp.zeros(4)
        .at[jnp.array([0, 0, 2])]
        .set(jnp.stack([x[0] ** 2, x[1], x[0] * x[1]]))
        @ jnp.arange(1.0, 5.0)
    ),
    "scatter_add": lambda x: (
        jnp.ones(3).at[jnp.array([1, 1])].add(x**2) @ x[jnp.array([0, 1, 1])]
    ),
    "dynamic_update_slice": lambda x: jnp.exp(
        lax.dynamic_update_slice(jnp.zeros(3), x * x[::-1], (1,))
    ).sum(),
}
"""Functions that together exercise every rule."""
