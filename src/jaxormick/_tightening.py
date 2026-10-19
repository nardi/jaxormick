"""Interval bounds tightened by subgradients, after Najman and Mitsos (2019).

Every intermediate value is relaxed twice: at the reference point and at the middle of
the box. The relaxations at the middle carry their subgradients, pushed forward through
each rule with `jax.linearize`. The affine minorant and majorant that these
subgradients define bound the value over the whole box, which often narrows its
interval. The middle depends only on the box, so the tightened bounds do too, and the
relaxations at the reference point stay convex and concave.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._relaxation import Relaxation


class Box(eqx.Module):
    """The box of the variables, flattened, with the point that subgradients refer to."""

    middle: Array
    """The middle of the box, where the subgradients are taken."""
    lower: Array
    """Lower bounds of the variables."""
    upper: Array
    """Upper bounds of the variables."""


class Tracked(eqx.Module):
    """A relaxed value, together with its relaxation and subgradients at the middle.

    The tangents have the shape of the value, preceded by one axis over the flattened
    variables. A tangent is NaN where no subgradient is known, which turns off the
    tightening that depends on it.
    """

    at_reference: Relaxation
    """The relaxation at the reference point."""
    at_middle: Relaxation
    """The relaxation at the middle of the box."""
    convex_tangents: Array
    """Subgradient of the convex side at the middle, one row per variable."""
    concave_tangents: Array
    """Supergradient of the concave side at the middle, one row per variable."""


def tracked_variables(
    reference: Array, lower: Array, upper: Array
) -> tuple[Tracked, Box]:
    """Returns the tracked relaxation of the variables and the box they lie in."""
    lower = jnp.broadcast_to(lower, reference.shape)
    upper = jnp.broadcast_to(upper, reference.shape)
    middle = (lower + upper) / 2
    size = reference.size
    tangents = jnp.eye(size, dtype=reference.dtype).reshape(size, *reference.shape)
    variables = Tracked(
        at_reference=Relaxation.variable(reference, lower, upper),
        at_middle=Relaxation.variable(middle, lower, upper),
        convex_tangents=tangents,
        concave_tangents=tangents,
    )
    return variables, Box(
        middle=middle.ravel(), lower=lower.ravel(), upper=upper.ravel()
    )


def affine_range(value: Array, tangents: Array, box: Box) -> tuple[Array, Array]:
    """Returns the range over the box of the plane through `value` at the middle."""
    trailing = (1,) * value.ndim
    to_lower = (tangents * (box.lower - box.middle).reshape(-1, *trailing)).astype(
        value.dtype
    )
    to_upper = (tangents * (box.upper - box.middle).reshape(-1, *trailing)).astype(
        value.dtype
    )
    lowest = value + jnp.sum(jnp.minimum(to_lower, to_upper), axis=0)
    highest = value + jnp.sum(jnp.maximum(to_lower, to_upper), axis=0)
    return lowest, highest


def tightened_bounds(
    at_middle: Relaxation, convex_tangents: Array, concave_tangents: Array, box: Box
) -> tuple[Array, Array]:
    """Returns the interval of a value, narrowed by its planes at the middle.

    The convex side lies below the value and above its tangent plane, so the lowest
    point of that plane over the box is a lower bound. The concave side gives an upper
    bound in the same way. A plane that is not finite leaves its bound unchanged. The
    lower bound never exceeds the upper one, even after rounding.
    """
    lowest, _ = affine_range(at_middle.convex, convex_tangents, box)
    _, highest = affine_range(at_middle.concave, concave_tangents, box)
    lower = jnp.where(
        jnp.isfinite(lowest), jnp.maximum(at_middle.lower, lowest), at_middle.lower
    )
    upper = jnp.where(
        jnp.isfinite(highest), jnp.minimum(at_middle.upper, highest), at_middle.upper
    )
    lower = jnp.minimum(lower, at_middle.upper)
    return lower, jnp.maximum(upper, lower)
