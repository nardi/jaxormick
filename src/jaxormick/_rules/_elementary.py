"""Rules for exponentials, logarithms, the absolute value and sigmoid functions.

The module also covers hyperbolic functions, their inverses and error functions.
"""

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from .._relaxation import Relaxation
from ._arithmetic import negate
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


@relaxes("exp2")
def exp2_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `exp2`, which is convex and increasing."""
    operand = as_relaxation(operands[0])
    return compose_convex(operand, jnp.exp2, operand.lower)


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


@relaxes("log2")
def log2_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `log2`, which is concave and increasing on positive arguments."""
    return shifted_logarithm(as_relaxation(operands[0]), jnp.log2, 0.0)


def absolute(operand: Relaxation) -> Relaxation:
    """Relaxes `|operand|`, which is convex and smallest at the point closest to zero."""
    return compose_convex(operand, jnp.abs, jnp.clip(0, operand.lower, operand.upper))


@relaxes("abs")
def abs_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `abs`."""
    return absolute(as_relaxation(operands[0]))


def convex_concave(
    operand: Relaxation, function: UnivariateFunction, increasing: bool
) -> Relaxation:
    """Relaxes a monotone function that is convex below zero and concave above it.

    Both envelopes come from `convex_envelope_of_convex_concave`. A monotone function
    has monotone envelopes, so they reach their extremes at the bounds of the interval.
    """
    lower, upper = operand.lower, operand.upper
    smallest_at, largest_at = (lower, upper) if increasing else (upper, lower)
    return compose(
        operand,
        lambda point: convex_envelope_of_convex_concave(
            function, 0.0, lower, upper, point
        ),
        smallest_at,
        lambda point: concave_envelope_of_convex_concave(
            function, 0.0, lower, upper, point
        ),
        largest_at,
        function(smallest_at),
        function(largest_at),
    )


def concave_convex(
    operand: Relaxation, function: UnivariateFunction, increasing: bool
) -> Relaxation:
    """Relaxes a monotone function that is concave below zero and convex above it.

    Its negation is convex-concave and monotone in the other direction.
    """
    return negate(
        convex_concave(operand, lambda point: -function(point), not increasing)
    )


def increasing_on_open_domain(
    operand: Relaxation,
    relax: Callable[[Relaxation], Relaxation],
    function: UnivariateFunction,
    start: float,
    end: float,
) -> Relaxation:
    """Relaxes an increasing function that is defined between `start` and `end`.

    `relax` relaxes the function on intervals strictly inside the domain. Otherwise the
    relaxation falls back to the bounds over the part of the interval inside the
    domain, since points outside it cannot be feasible. These bounds are infinite where
    the function is.
    """
    is_inside = (operand.lower > start) & (operand.upper < end)

    # The substituted value lies inside the domain, even when it is unbounded above.
    inside = (start + end) / 2 if end < math.inf else start + 1
    safe = substitute(operand, is_inside, inside)
    fallback = interval_only(
        function(jnp.clip(operand.lower, start, end)),
        function(jnp.clip(operand.upper, start, end)),
    )
    return select(is_inside, relax(safe), fallback)


@relaxes("tanh")
def tanh_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `tanh`."""
    return convex_concave(as_relaxation(operands[0]), jnp.tanh, increasing=True)


@relaxes("logistic")
def logistic_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `logistic`, the function behind `jax.nn.sigmoid`."""
    return convex_concave(as_relaxation(operands[0]), jax.nn.sigmoid, increasing=True)


@relaxes("atanh")
def atanh_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `atanh`, which is increasing, concave below zero and convex above it."""
    return increasing_on_open_domain(
        as_relaxation(operands[0]),
        lambda operand: concave_convex(operand, jnp.atanh, increasing=True),
        jnp.atanh,
        -1.0,
        1.0,
    )


@relaxes("sinh")
def sinh_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `sinh`, which is increasing, concave below zero and convex above it."""
    return concave_convex(as_relaxation(operands[0]), jnp.sinh, increasing=True)


@relaxes("cosh")
def cosh_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `cosh`, which is convex and smallest at the point closest to zero."""
    operand = as_relaxation(operands[0])
    return compose_convex(operand, jnp.cosh, jnp.clip(0, operand.lower, operand.upper))


@relaxes("asinh")
def asinh_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `asinh`, which is increasing, convex below zero and concave above it."""
    return convex_concave(as_relaxation(operands[0]), jnp.asinh, increasing=True)


@relaxes("acosh")
def acosh_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `acosh`, which is increasing and concave on arguments above one."""
    return increasing_on_open_domain(
        as_relaxation(operands[0]),
        lambda safe: compose_concave(safe, jnp.acosh, safe.upper),
        jnp.acosh,
        1.0,
        math.inf,
    )


@relaxes("cbrt")
def cbrt_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `cbrt`, which is increasing, convex below zero and concave above it."""
    return convex_concave(as_relaxation(operands[0]), jnp.cbrt, increasing=True)


@relaxes("erf")
def erf_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `erf`, which is increasing, convex below zero and concave above it."""
    return convex_concave(
        as_relaxation(operands[0]), jax.scipy.special.erf, increasing=True
    )


@relaxes("erfc")
def erfc_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `erfc`, which is decreasing, concave below zero and convex above it."""
    return concave_convex(
        as_relaxation(operands[0]), jax.scipy.special.erfc, increasing=False
    )


@relaxes("erf_inv")
def erf_inv_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `erf_inv`, which is increasing, concave below zero and convex above."""
    return increasing_on_open_domain(
        as_relaxation(operands[0]),
        lambda safe: concave_convex(safe, jax.scipy.special.erfinv, increasing=True),
        jax.scipy.special.erfinv,
        -1.0,
        1.0,
    )
