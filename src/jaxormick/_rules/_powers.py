"""Rules for powers and roots."""

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import Tracer
from jax.extend.core import Primitive
from numpy.typing import NDArray

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


def fractional_power(
    operand: Relaxation, exponent: float | NDArray[np.float64]
) -> Relaxation:
    """Relaxes `operand ** exponent` for constant exponents that are not integers.

    Such powers are defined for nonnegative arguments only. They are convex and
    increasing for `exponent > 1`, concave and increasing for `0 < exponent < 1`, and
    convex and decreasing on positive arguments for `exponent < 0`. The exponent is a
    number, or an array with the shape of the operand whose elements all lie in the
    same one of these three classes.
    """
    exponent_values = jnp.asarray(exponent, operand.dtype)

    def power(point: jax.Array) -> jax.Array:
        return jnp.power(point, exponent_values)

    if np.all(exponent > 0):
        return increasing_on_nonnegative(
            operand, power, is_convex=bool(np.all(exponent > 1))
        )
    return decreasing_on_positive(operand, power)


def power_with_constant_exponent(operand: Relaxation, exponent: float) -> Relaxation:
    """Relaxes `operand ** exponent` for one constant exponent, integer or not."""
    if exponent == round(exponent):
        return integer_power(operand, round(exponent))
    return fractional_power(operand, exponent)


def power_by_exponent_class(
    operand: Relaxation, exponents: NDArray[np.float64]
) -> Relaxation:
    """Relaxes `operand ** exponents` elementwise for exponents of different classes.

    The elements are grouped by exponent, with one group for each distinct integer and
    one for each class of `fractional_power`. A group is relaxed on its own elements
    with the rule of its class, and the results go back to their original positions.
    A group never runs on an element that belongs to another one, so a rule does not
    meet arguments outside its domain. Its derivatives then stay free of NaN values.
    The program grows with the number of groups, and not with the number of elements.
    """
    flat_exponents = exponents.reshape(-1)
    is_integer = flat_exponents == np.round(flat_exponents)
    classes = [
        flat_exponents == value for value in np.unique(flat_exponents[is_integer])
    ] + [
        ~is_integer & (flat_exponents > 1),
        ~is_integer & (flat_exponents > 0) & (flat_exponents < 1),
        ~is_integer & (flat_exponents < 0),
    ]
    groups = [np.flatnonzero(members) for members in classes if members.any()]
    flat_operand = jax.tree.map(lambda field: field.reshape(-1), operand)

    def relax_group(indices: NDArray[np.intp]) -> Relaxation:
        """Relaxes the power of the elements at `indices`, which share a class."""
        members = jax.tree.map(lambda field: field[indices], flat_operand)
        group_exponents = flat_exponents[indices]
        if is_integer[indices[0]]:
            return integer_power(members, round(group_exponents[0]))
        return fractional_power(members, group_exponents)

    # Concatenating the groups reorders the elements, and the inverse permutation of
    # that order restores them.
    inverse_order = np.argsort(np.concatenate(groups))
    return jax.tree.map(
        lambda *parts: jnp.concatenate(parts)[inverse_order].reshape(exponents.shape),
        *(relax_group(indices) for indices in groups),
    )


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


@relaxes("one_minus_square")
def one_minus_square_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `one_minus_square`, which is `1 − x²`, concave and largest at zero.

    JAX 0.11 uses it in the derivative of `tanh`.
    """
    operand = as_relaxation(operands[0])
    return compose_concave(
        operand,
        lambda point: 1 - lax.integer_pow(point, 2),
        jnp.clip(0, operand.lower, operand.upper),
    )


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
    known while tracing. It may be an array that broadcasts against the base, with a
    different exponent for each element. Integer-valued exponents use the rules of
    `integer_pow`.
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
    exponents = np.asarray(exponent, dtype=np.float64)
    base = as_relaxation(base)
    shape = np.broadcast_shapes(base.shape, exponents.shape)
    base = jax.tree.map(lambda field: jnp.broadcast_to(field, shape), base)
    match np.unique(exponents).tolist():
        case []:
            return base
        case [value]:
            return power_with_constant_exponent(base, value)
        case _:
            return power_by_exponent_class(base, np.broadcast_to(exponents, shape))
