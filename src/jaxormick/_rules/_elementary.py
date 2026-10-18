"""Rules for exponentials, logarithms, the absolute value and sigmoid functions."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from .._relaxation import Relaxation
from ._composition import (
    UnivariateFunction,
    as_relaxation,
    compose,
    compose_concave,
    compose_convex,
    concave_envelope_of_convex_concave,
    convex_envelope_of_convex_concave,
    interval_only,
    select,
    substitute,
)
from ._registry import Operand, Params, relaxes


@relaxes("exp")
def exp_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `exp`, which is convex and increasing."""
    operand = as_relaxation(operands[0])
    return compose_convex(operand, jnp.exp, operand.lower)


@relaxes("expm1")
def expm1_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `expm1`, which is convex and increasing."""
    operand = as_relaxation(operands[0])
    return compose_convex(operand, jnp.expm1, operand.lower)


def shifted_logarithm(
    operand: Relaxation, function: UnivariateFunction, domain_start: float
) -> Relaxation:
    """Relaxes a logarithm that is concave and increasing right of `domain_start`.

    When the interval reaches `domain_start`, the logarithm has no lower bound, so the
    convex side is `−∞` and the concave side is the constant value at the upper bound.
    """
    is_defined = operand.lower > domain_start
    safe = substitute(operand, is_defined, domain_start + 1)
    composed = compose_concave(safe, function, safe.upper)
    smallest_step = jnp.finfo(operand.dtype).tiny
    unbounded = interval_only(
        jnp.full_like(operand.lower, -jnp.inf),
        function(jnp.maximum(operand.upper, domain_start + smallest_step)),
    )
    return select(is_defined, composed, unbounded)


@relaxes("log")
def log_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `log`, which is concave and increasing on positive arguments."""
    return shifted_logarithm(as_relaxation(operands[0]), jnp.log, 0.0)


@relaxes("log1p")
def log1p_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `log1p`, which is concave and increasing for arguments above −1."""
    return shifted_logarithm(as_relaxation(operands[0]), jnp.log1p, -1.0)


def absolute(operand: Relaxation) -> Relaxation:
    """Relaxes `|operand|`, which is convex and smallest at the point closest to zero."""
    return compose_convex(operand, jnp.abs, jnp.clip(0, operand.lower, operand.upper))


@relaxes("abs")
def abs_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `abs`."""
    return absolute(as_relaxation(operands[0]))


def sigmoid(operand: Relaxation, function: UnivariateFunction) -> Relaxation:
    """Relaxes an increasing function that is convex below zero and concave above it.

    Both envelopes come from `convex_envelope_of_convex_concave`. An increasing function has
    increasing envelopes, so the convex one is smallest at the lower bound and the
    concave one is largest at the upper bound.
    """
    lower, upper = operand.lower, operand.upper
    return compose(
        operand,
        lambda point: convex_envelope_of_convex_concave(
            function, 0.0, lower, upper, point
        ),
        lower,
        lambda point: concave_envelope_of_convex_concave(
            function, 0.0, lower, upper, point
        ),
        upper,
        function(lower),
        function(upper),
    )


@relaxes("tanh")
def tanh_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `tanh`."""
    return sigmoid(as_relaxation(operands[0]), jnp.tanh)


@relaxes("logistic")
def logistic_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `logistic`, the function behind `jax.nn.sigmoid`."""
    return sigmoid(as_relaxation(operands[0]), jax.nn.sigmoid)
