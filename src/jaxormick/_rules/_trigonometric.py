"""Rules for `sin` and `cos`.

Both rules reduce to the convex envelope of `cos`.
`sin(x) = cos(x − π/2)` and `−cos(x) = cos(x + π)`, so the concave envelope of `cos` and
both envelopes of `sin` follow by shifting the interval. The construction follows the
case analysis of Mitsos, Chachuat and Barton (2009).
"""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from .._relaxation import Relaxation
from ._composition import as_relaxation, compose, convex_envelope_of_convex_concave
from ._registry import Operand, Params, relaxes


def canonical_cos_envelope(
    lower: jax.Array, upper: jax.Array, point: jax.Array
) -> jax.Array:
    """Returns the convex envelope of `cos` on `[lower, upper]` as a convex-concave function.

    `cos` is convex on `[−π, −π/2]` and concave right of it up to `π/2`. The result is
    the convex envelope of `cos` on intervals inside `[−π, π/2]`, and a valid one on
    longer intervals under the conditions in `cos_convex_envelope`.
    """
    return convex_envelope_of_convex_concave(jnp.cos, -math.pi / 2, lower, upper, point)


def cos_convex_envelope(
    lower: jax.Array, upper: jax.Array, point: jax.Array
) -> jax.Array:
    """Returns the convex envelope of `cos` on `[lower, upper]`, evaluated at `point`.

    The interval is first shifted by a multiple of 2π so that its lower end `a` lies in
    `[−π, π)`. Let `b` be the shifted upper end.

    When the interval does not reach the minimum at π, `cos` is convex-concave on it,
    concave-convex, or convex-concave-convex. A line can touch `cos` from below at two
    points only at two minima, so the envelope touches `cos` on one side only. It is
    either the envelope that treats `cos` as convex-concave on `[a, b]`, or the same
    after the reflection `x ↦ −x`, under which `cos` is unchanged. The first is valid
    unless it is the chord and the chord is less steep than `cos` at `b`, in which case
    `cos` dips below it just left of `b`.

    When the interval contains minima, the envelope is −1 between the first one, at π,
    and the last one, `m`. Left of π it is the envelope of `cos` on `[a, π]`, reflected,
    and right of `m` the envelope on `[m, b]`, shifted to start at −π. Both are
    convex-concave envelopes whose lines are never less steep than `cos` at their far
    end, and each continues as the constant −1 on the other side.
    """
    period = 2 * math.pi
    shift = period * jnp.floor((lower + math.pi) / period)
    a, b, x = lower - shift, upper - shift, point - shift
    contains_minimum = b > math.pi
    last_minimum = math.pi + period * jnp.floor((b - math.pi) / period)
    right_shift = last_minimum + math.pi

    # Two calls cover all cases, so that only two bisections run.
    first = canonical_cos_envelope(
        jnp.where(contains_minimum, -math.pi, a),
        jnp.where(contains_minimum, -a, b),
        jnp.where(contains_minimum, -x, x),
    )
    second = canonical_cos_envelope(
        jnp.where(contains_minimum, -math.pi, -b),
        jnp.where(contains_minimum, b - right_shift, -a),
        jnp.where(contains_minimum, x - right_shift, -x),
    )

    width = b - a
    chord_slope = (jnp.cos(b) - jnp.cos(a)) / jnp.where(width > 0, width, 1)
    uses_first = (b <= math.pi / 2) | (chord_slope >= -jnp.sin(b))
    without_minimum = jnp.where(uses_first, first, second)
    left_part = jnp.where(x <= math.pi, first, -1)
    right_part = jnp.where(x >= last_minimum, second, -1)
    with_minimum = jnp.maximum(left_part, right_part)
    return jnp.where(contains_minimum, with_minimum, without_minimum)


def first_multiple_at_or_above(value: jax.Array, offset: float) -> jax.Array:
    """Returns the smallest point `offset + 2πn` with an integer `n` that is `≥ value`."""
    period = 2 * math.pi
    return offset + period * jnp.ceil((value - offset) / period)


def cos_relaxation(operand: Relaxation) -> Relaxation:
    """Relaxes `cos(operand)`."""
    lower, upper = operand.lower, operand.upper
    value_at_lower, value_at_upper = jnp.cos(lower), jnp.cos(upper)

    # The maxima of `cos` lie at multiples of 2π and the minima halfway in between.
    first_maximum = first_multiple_at_or_above(lower, 0.0)
    first_minimum = first_multiple_at_or_above(lower, math.pi)
    contains_maximum = first_maximum <= upper
    contains_minimum = first_minimum <= upper
    argmin = jnp.where(
        contains_minimum,
        first_minimum,
        jnp.where(value_at_lower <= value_at_upper, lower, upper),
    )
    argmax = jnp.where(
        contains_maximum,
        first_maximum,
        jnp.where(value_at_lower >= value_at_upper, lower, upper),
    )
    return compose(
        operand,
        lambda point: cos_convex_envelope(lower, upper, point),
        argmin,
        lambda point: (
            -cos_convex_envelope(lower + math.pi, upper + math.pi, point + math.pi)
        ),
        argmax,
        jnp.where(contains_minimum, -1, jnp.minimum(value_at_lower, value_at_upper)),
        jnp.where(contains_maximum, 1, jnp.maximum(value_at_lower, value_at_upper)),
    )


@relaxes("cos")
def cos_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `cos`."""
    return cos_relaxation(as_relaxation(operands[0]))


@relaxes("sin")
def sin_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `sin` as `cos(x − π/2)`, which is exact since the shift is constant."""
    operand = as_relaxation(operands[0])
    return cos_relaxation(jax.tree.map(lambda field: field - math.pi / 2, operand))
