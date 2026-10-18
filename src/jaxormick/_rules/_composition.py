"""Building blocks shared by the relaxation rules.

Most rules relax a univariate function `f(x)` of a relaxed operand `x`. McCormick's
composition rule needs a convex underestimator of `f` over the interval of `x`, the point
where that underestimator is smallest, and the same for a concave overestimator. The
helpers here compute these pieces and combine them.
"""

from collections.abc import Callable
from typing import TypeGuard

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike

from .._relaxation import Relaxation
from ._registry import Operand

UnivariateFunction = Callable[[Array], Array]
"""An elementwise function of one array."""

Envelope = Callable[[Array], Array]
"""A convex or concave bound of a univariate function over a fixed interval."""


def is_relaxed(operand: Operand) -> TypeGuard[Relaxation]:
    """Returns whether an operand depends on the variables."""
    return isinstance(operand, Relaxation)


def as_relaxation(operand: Operand) -> Relaxation:
    """Returns the operand as a `Relaxation`, treating plain arrays as exact constants."""
    return operand if isinstance(operand, Relaxation) else Relaxation.constant(operand)


def mid(convex: Array, concave: Array, point: ArrayLike) -> Array:
    """Returns McCormick's mid operator, the median of `convex`, `concave` and `point`.

    The median is written as an explicit choice of branch that relies on
    `convex ≤ concave`. A median built from `jnp.minimum` and `jnp.maximum` has the same
    value, but where `convex` and `concave` tie, `jax.grad` splits the gradient between
    them. That mixes a subgradient of the convex side with a supergradient of the concave
    side, and the resulting tangent planes can cut off feasible points. Ties are common
    on the boundary of the box, which is where the solutions of linear programs lie.
    """
    return jnp.where(point < convex, convex, jnp.where(point > concave, concave, point))


def secant(
    value_at_lower: Array,
    value_at_upper: Array,
    lower: Array,
    upper: Array,
    point: Array,
) -> Array:
    """Returns the line through `(lower, value_at_lower)` and `(upper, value_at_upper)`.

    The line is evaluated at `point`. On an interval of zero width it is the constant
    `value_at_lower`.
    """
    width = upper - lower

    # A safe width keeps the unused branch of `jnp.where` free of NaN values and NaN
    # gradients.
    safe_width = jnp.where(width > 0, width, 1)
    slope = jnp.where(width > 0, (value_at_upper - value_at_lower) / safe_width, 0)
    return value_at_lower + slope * (point - lower)


def compose(
    operand: Relaxation,
    convex_envelope: Envelope,
    argmin: Array,
    concave_envelope: Envelope,
    argmax: Array,
    lower: Array,
    upper: Array,
) -> Relaxation:
    """Relaxes `f(operand)` with McCormick's composition rule.

    `convex_envelope` and `concave_envelope` bound `f` from below and above on the
    interval of the operand. `argmin` is a point where the convex one is smallest and
    `argmax` a point where the concave one is largest. `lower` and `upper` bound `f` on
    the interval.
    """
    convex = convex_envelope(mid(operand.convex, operand.concave, argmin))
    concave = concave_envelope(mid(operand.convex, operand.concave, argmax))
    return Relaxation(
        lower=lower, upper=upper, convex=convex, concave=concave
    ).tightened()


def compose_convex(
    operand: Relaxation, function: UnivariateFunction, argmin: Array
) -> Relaxation:
    """Relaxes `function(operand)` for a function that is convex on the interval.

    `argmin` is the point of the interval where `function` is smallest. The convex side
    uses the function itself and the concave side uses its secant.
    """
    value_at_lower, value_at_upper = function(operand.lower), function(operand.upper)

    # The secant is linear, so it is largest at the endpoint with the larger value.
    argmax = jnp.where(value_at_upper >= value_at_lower, operand.upper, operand.lower)
    return compose(
        operand,
        function,
        argmin,
        lambda point: secant(
            value_at_lower, value_at_upper, operand.lower, operand.upper, point
        ),
        argmax,
        function(argmin),
        jnp.maximum(value_at_lower, value_at_upper),
    )


def compose_concave(
    operand: Relaxation, function: UnivariateFunction, argmax: Array
) -> Relaxation:
    """Relaxes `function(operand)` for a function that is concave on the interval.

    This mirrors `compose_convex`, with the secant on the convex side and the function
    itself on the concave side.
    """
    value_at_lower, value_at_upper = function(operand.lower), function(operand.upper)
    argmin = jnp.where(value_at_upper <= value_at_lower, operand.upper, operand.lower)
    return compose(
        operand,
        lambda point: secant(
            value_at_lower, value_at_upper, operand.lower, operand.upper, point
        ),
        argmin,
        function,
        argmax,
        jnp.minimum(value_at_lower, value_at_upper),
        function(argmax),
    )


def select(
    condition: Array, when_true: Relaxation, when_false: Relaxation
) -> Relaxation:
    """Returns the fields of `when_true` where `condition` holds, else of `when_false`."""
    return jax.tree.map(
        lambda true_field, false_field: jnp.where(condition, true_field, false_field),
        when_true,
        when_false,
    )


def substitute(operand: Relaxation, condition: Array, value: float) -> Relaxation:
    """Returns the operand with every field set to `value` where `condition` is false.

    A branch that `select` discards still runs, and `jax.grad` multiplies its local
    derivatives by a zero cotangent. Infinite or undefined values in that branch then
    turn the zero into NaN. Each branch therefore runs on an operand that holds a
    harmless value wherever the branch is not selected.
    """
    return jax.tree.map(lambda field: jnp.where(condition, field, value), operand)


def interval_only(lower: Array, upper: Array) -> Relaxation:
    """Returns a relaxation whose convex and concave sides are the constant bounds."""
    return Relaxation(lower=lower, upper=upper, convex=lower, concave=upper)


def derivative(function: UnivariateFunction, point: Array) -> Array:
    """Returns the derivative of an elementwise function at `point`."""
    return jax.jvp(function, (point,), (jnp.ones_like(point),))[1]


def bisection_steps(dtype: jnp.dtype) -> int:
    """Returns the number of bisection steps that resolve an interval to machine precision.

    Each step halves the bracket, so one step per bit of the significand, plus two, is
    enough to reach the spacing of floating-point numbers.
    """
    return int(jnp.finfo(dtype).nmant) + 2


def convex_envelope_of_convex_concave(
    function: UnivariateFunction,
    inflection: ArrayLike,
    lower: Array,
    upper: Array,
    point: Array,
    tangent_point: Array | None = None,
) -> Array:
    """Returns the convex envelope over `[lower, upper]` of a convex-concave function.

    `function` has to be convex left of `inflection` and concave right of it. Its convex
    envelope follows the function up to a tangent point `t` and then the line from
    `(t, function(t))` to `(upper, function(upper))`. When that line would start left of
    `lower`, the envelope is the secant through both endpoints. When the interval lies
    left of the inflection point, it is the function itself.

    The tangent point solves `function(t) + function'(t)·(upper − t) = function(upper)`.
    When `tangent_point` is not given, a fixed number of bisection steps on
    `[lower, inflection]` brackets it as `left ≤ t ≤ right`. The bracket defines two convex
    underestimators that are valid for any bracket around `t`. One follows the function
    up to `left` and then its tangent there. The other is the line through
    `(upper, function(upper))` with slope `function'(right)`, which is at least as steep
    as the envelope's line. Their maximum converges to the envelope as the bracket
    shrinks.
    """
    value_at_upper = function(upper)

    def tangent_gap(candidate: Array) -> Array:
        """Returns how far the tangent at `candidate` passes above `(upper, f(upper))`."""
        return (
            function(candidate)
            + derivative(function, candidate) * (upper - candidate)
            - value_at_upper
        )

    right_end = jnp.minimum(inflection, upper)
    if tangent_point is None:

        def bisection_step(
            step: int, bracket: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            """Halves the bracket, keeping the root of `tangent_gap` inside it."""
            left, right = bracket
            middle = (left + right) / 2
            above = tangent_gap(middle) >= 0
            return jnp.where(above, left, middle), jnp.where(above, middle, right)

        left, right = lax.fori_loop(
            0,
            bisection_steps(jnp.result_type(lower)),
            bisection_step,
            (lower, right_end),
        )
    else:
        left = right = jnp.clip(tangent_point, lower, right_end)

    follows_function = function(jnp.minimum(point, left)) + derivative(
        function, left
    ) * jnp.maximum(point - left, 0)
    steeper_line = value_at_upper + derivative(function, right) * (point - upper)
    curve = jnp.maximum(follows_function, steeper_line)

    chord = secant(function(lower), value_at_upper, lower, upper, point)
    uses_chord = (lower >= inflection) | (tangent_gap(lower) >= 0)
    return jnp.where(
        upper <= inflection, function(point), jnp.where(uses_chord, chord, curve)
    )


def concave_envelope_of_convex_concave(
    function: UnivariateFunction,
    inflection: ArrayLike,
    lower: Array,
    upper: Array,
    point: Array,
    tangent_point: Array | None = None,
) -> Array:
    """Returns the concave envelope over `[lower, upper]` of a convex-concave function.

    `function` has the shape required by `convex_envelope_of_convex_concave`. Its concave envelope
    is the negated convex envelope of `−function(−x)`, which is again convex left of
    `−inflection` and concave right of it, over the mirrored interval.
    `tangent_point` is the mirrored tangent point, if known.
    """
    return -convex_envelope_of_convex_concave(
        lambda mirrored: -function(-mirrored),
        -jnp.asarray(inflection),
        -upper,
        -lower,
        -point,
        tangent_point,
    )
