"""Rules for powers and roots."""

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import Tracer
from jax.extend.core import Primitive

from .._errors import UnsupportedPrimitiveError
from .._relaxation import Relaxation
from ._arithmetic import reciprocal
from ._composition import (
    UnivariateFunction,
    as_relaxation,
    compose,
    compose_concave,
    compose_convex,
    convex_envelope_of_convex_concave,
    interval_only,
    is_relaxed,
    select,
    substitute,
)
from ._registry import Operand, Params, relaxes


@functools.cache
def odd_power_tangent_ratio(exponent: int) -> float:
    """Returns the ratio that locates the tangent point of the odd power `x ** exponent`.

    On an interval `[lower, upper]` around zero, the convex envelope of an odd power is
    the line from `(lower, lower ** exponent)` to a tangent point `t`, followed by the
    power itself. Liberti and Pantelides (2003) show that `t = −ratio · lower`, where
    `ratio` is the positive real root of
    `(exponent − 1) · r ** exponent + exponent · r ** (exponent − 1) − 1`. The root
    depends only on the exponent, so it is computed once while tracing.
    """
    coefficients = [exponent - 1, exponent, *([0] * (exponent - 2)), -1]
    roots = np.roots(coefficients)
    positive_real = [
        root.real for root in roots if abs(root.imag) < 1e-12 and root.real > 0
    ]
    return float(positive_real[0])


def odd_power(operand: Relaxation, exponent: int) -> Relaxation:
    """Relaxes `operand ** exponent` for an odd exponent of at least three.

    The power is concave left of zero and convex right of it. Its envelopes come from
    `convex_envelope_of_convex_concave` after mirroring, with the tangent point in closed form.
    """
    ratio = odd_power_tangent_ratio(exponent)
    lower, upper = operand.lower, operand.upper

    def power(point: jax.Array) -> jax.Array:
        return lax.integer_pow(point, exponent)

    def negated_power(point: jax.Array) -> jax.Array:
        return -lax.integer_pow(point, exponent)

    # The convex envelope of `x ** k` on `[l, u]` is the convex envelope of
    # `(−y) ** k = −y ** k` on `[−u, −l]`, evaluated at `y = −x`.
    def convex_envelope(point: jax.Array) -> jax.Array:
        return convex_envelope_of_convex_concave(
            negated_power, 0.0, -upper, -lower, -point, ratio * lower
        )

    # The concave envelope of `x ** k` is the negated convex envelope of `−x ** k`.
    def concave_envelope(point: jax.Array) -> jax.Array:
        return -convex_envelope_of_convex_concave(
            negated_power, 0.0, lower, upper, point, -ratio * upper
        )

    return compose(
        operand,
        convex_envelope,
        lower,
        concave_envelope,
        upper,
        power(lower),
        power(upper),
    )


def integer_power(operand: Relaxation, exponent: int) -> Relaxation:
    """Relaxes `operand ** exponent` for an integer exponent."""
    match exponent:
        case 0:
            return Relaxation.constant(jnp.ones_like(operand.lower))
        case 1:
            return operand
        case negative if negative < 0:
            return reciprocal(integer_power(operand, -negative))
        case even if even % 2 == 0:
            return compose_convex(
                operand,
                lambda point: lax.integer_pow(point, even),
                jnp.clip(0, operand.lower, operand.upper),
            )
        case odd:
            return odd_power(operand, odd)


def increasing_on_nonnegative(
    operand: Relaxation, function: UnivariateFunction, is_convex: bool
) -> Relaxation:
    """Relaxes a function that is increasing on `[0, ∞)` and undefined below zero.

    On intervals with a nonnegative lower bound, the function is composed with
    McCormick's rule. Otherwise the relaxation falls back to the constant bounds over
    the nonnegative part of the interval, since points with a negative argument cannot
    be feasible.
    """
    is_defined = operand.lower >= 0
    safe = substitute(operand, is_defined, 1.0)
    composed = (
        compose_convex(safe, function, safe.lower)
        if is_convex
        else compose_concave(safe, function, safe.upper)
    )
    nonnegative_lower = jnp.maximum(operand.lower, 0)
    nonnegative_upper = jnp.maximum(operand.upper, 0)
    fallback = interval_only(function(nonnegative_lower), function(nonnegative_upper))
    return select(is_defined, composed, fallback)


def decreasing_on_positive(
    operand: Relaxation, function: UnivariateFunction
) -> Relaxation:
    """Relaxes a function that is convex and decreasing on `(0, ∞)`, like `x ** −0.5`.

    When the interval reaches zero the function has no upper bound, so the concave side
    is `∞` and the convex side is the constant value at the upper bound.
    """
    is_defined = operand.lower > 0
    safe = substitute(operand, is_defined, 1.0)
    composed = compose_convex(safe, function, safe.upper)
    smallest_positive = jnp.finfo(operand.dtype).tiny
    fallback = interval_only(
        function(jnp.maximum(operand.upper, smallest_positive)),
        jnp.full_like(operand.upper, jnp.inf),
    )
    return select(is_defined, composed, fallback)


def fractional_power(operand: Relaxation, exponent: float) -> Relaxation:
    """Relaxes `operand ** exponent` for a constant exponent that is not an integer.

    Such powers are defined for nonnegative arguments only. They are convex and
    increasing for `exponent > 1`, concave and increasing for `0 < exponent < 1`, and
    convex and decreasing on positive arguments for `exponent < 0`.
    """

    def power(point: jax.Array) -> jax.Array:
        return jnp.power(point, exponent)

    if exponent > 0:
        return increasing_on_nonnegative(operand, power, is_convex=exponent > 1)
    return decreasing_on_positive(operand, power)


@relaxes("integer_pow")
def integer_pow_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `integer_pow`."""
    return integer_power(as_relaxation(operands[0]), params["y"])


@relaxes("square")
def square_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `square`."""
    return integer_power(as_relaxation(operands[0]), 2)


@relaxes("sqrt")
def sqrt_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `sqrt`, which is concave and increasing on nonnegative arguments."""
    return increasing_on_nonnegative(
        as_relaxation(operands[0]), jnp.sqrt, is_convex=False
    )


@relaxes("rsqrt")
def rsqrt_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `rsqrt`, which is convex and decreasing on positive arguments."""
    return decreasing_on_positive(as_relaxation(operands[0]), lax.rsqrt)


@relaxes("pow")
def pow_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `pow` with a constant exponent, which `x ** 1.5` and `x ** 2.0` produce.

    The exponent has to be a literal or a closed-over constant, so that its value is
    known while tracing. Integer-valued exponents use the rules of `integer_pow`.
    """
    base, exponent = operands
    if is_relaxed(exponent):
        raise UnsupportedPrimitiveError(
            "`pow` with an exponent that depends on the variables is not supported."
        )
    if isinstance(exponent, Tracer):
        raise UnsupportedPrimitiveError(
            "`pow` needs an exponent whose value is known while tracing."
        )
    value = float(np.asarray(exponent))
    base = as_relaxation(base)
    if value == round(value):
        return integer_power(base, round(value))
    return fractional_power(base, value)
