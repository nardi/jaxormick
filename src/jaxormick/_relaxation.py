import functools
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree


class Relaxation(eqx.Module):
    """Interval bounds and McCormick relaxations of an array over a box.

    All four fields have the shape of the relaxed value. `convex` and `concave` are the
    values of the relaxations at the reference point, so they depend on that point,
    while `lower` and `upper` depend only on the box.
    """

    lower: Array
    """Lower bound of the value over the box."""
    upper: Array
    """Upper bound of the value over the box."""
    convex: Array
    """Convex underestimator of the value, evaluated at the reference point."""
    concave: Array
    """Concave overestimator of the value, evaluated at the reference point."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the relaxed value."""
        return jnp.shape(self.lower)

    @property
    def dtype(self) -> jnp.dtype:
        """The dtype of the relaxed value."""
        return jnp.result_type(self.lower)

    @classmethod
    def constant(cls, value: ArrayLike) -> Self:
        """Returns the exact relaxation of a value that does not depend on the variables."""
        value = jnp.asarray(value)
        return cls(lower=value, upper=value, convex=value, concave=value)

    @classmethod
    def variable(cls, reference: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> Self:
        """Returns the relaxation of the variables themselves.

        A variable is linear in itself, so both relaxations equal the reference point.
        """
        reference = jnp.asarray(reference)
        return cls(
            lower=jnp.asarray(lower),
            upper=jnp.asarray(upper),
            convex=reference,
            concave=reference,
        )

    def tightened(self) -> Self:
        """Returns the relaxation with `convex` and `concave` clipped to the bounds.

        The maximum of a convex function and a constant is still convex, so the clipped
        relaxations stay valid. Near a tie, the bound wins, even if the side is a few
        ulps tighter. Its gradient of zero gives a valid plane even where rounding has
        pushed the side past the bound.
        """
        return type(self)(
            lower=self.lower,
            upper=self.upper,
            convex=larger(self.lower, self.convex),
            concave=smaller(self.upper, self.concave),
        )


TIE_TOLERANCE = 16
"""Ulps of the arguments within which `larger` and `smaller` keep the first one."""


def tie_tolerance(first: Array, second: Array, scale: ArrayLike = 0.0) -> Array:
    """Returns a few ulps of the larger magnitude of the arguments.

    An argument that is a sum of larger terms can be near zero while its rounding error
    is a few ulps of those terms. Callers then pass the size of the terms as `scale`,
    and the tolerance covers it too. Near zero, the tolerance stays at a few times the
    smallest normal number. XLA may flush smaller numbers to zero, and a tolerance of
    zero would not stop ties flipping. The tolerance is at most the largest finite
    number, so that infinite arguments tie only with themselves: `first - tolerance`
    stays finite, and `-inf` is not within it of anything finite.
    """
    limits = jnp.finfo(jnp.result_type(first, second))
    magnitude = jnp.maximum(jnp.abs(first), jnp.abs(second))
    if not (isinstance(scale, int | float) and scale == 0):
        magnitude = jnp.maximum(magnitude, jnp.abs(scale))
    tolerance = TIE_TOLERANCE * (limits.eps * magnitude + limits.smallest_normal)
    return jnp.minimum(tolerance, limits.max)


def scale_tolerance(scale: Array) -> Array:
    """Returns the tie tolerance for values of magnitude `scale`, to share between calls."""
    limits = jnp.finfo(scale.dtype)
    return jnp.minimum(
        TIE_TOLERANCE * (limits.eps * jnp.abs(scale) + limits.smallest_normal),
        limits.max,
    )


def larger(
    first: Array,
    second: Array,
    scale: ArrayLike = 0.0,
    tolerance: Array | None = None,
) -> Array:
    """Returns the elementwise maximum, differentiated along the argument it picks.

    Where the arguments tie, `jnp.maximum` splits the gradient between them. A plain
    choice with `jnp.where` does not fix this either. XLA may evaluate the comparison
    in several fused kernels with different rounding, so that the paths of the gradient
    pick different arguments, and their sum is no subgradient. This function keeps the
    first argument unless the second is larger by more than a few ulps. Near a tie,
    rounding then cannot flip the choice. The result may fall short of the maximum by
    these few ulps, which is still valid, since each argument is. `scale` is passed to
    `tie_tolerance`, unless the caller passes the `tolerance` itself.
    """
    if tolerance is None:
        tolerance = tie_tolerance(first, second, scale)
    return jnp.where(first >= second - tolerance, first, second)


def smaller(
    first: Array,
    second: Array,
    scale: ArrayLike = 0.0,
    tolerance: Array | None = None,
) -> Array:
    """Returns the elementwise minimum, differentiated along the argument it picks.

    This mirrors `larger`.
    """
    if tolerance is None:
        tolerance = tie_tolerance(first, second, scale)
    return jnp.where(first <= second + tolerance, first, second)


def intersect_relaxations(*relaxations: PyTree[Relaxation]) -> PyTree[Relaxation]:
    """Returns the intersection of relaxations of the same function over the same box.

    The relaxations may come from different forms of the function, such as
    `x * (1 - x)` and `0.25 - (x - 0.5) ** 2`, or from different methods. Each field of
    the result is the tightest of its inputs. The maximum of convex functions is
    convex and the minimum of concave functions is concave, so the result is a valid
    relaxation and at least as tight as each input. Its derivatives follow the input
    that attains the maximum or minimum, which gives a valid subgradient.

    The arguments have the same tree structure, with a `Relaxation` at every leaf.
    """
    if not relaxations:
        raise ValueError("`intersect_relaxations` takes at least one relaxation.")

    def intersect(*leaves: Relaxation) -> Relaxation:
        """Returns the intersection of the relaxations at one leaf."""
        return Relaxation(
            lower=functools.reduce(jnp.maximum, (leaf.lower for leaf in leaves)),
            upper=functools.reduce(jnp.minimum, (leaf.upper for leaf in leaves)),
            convex=functools.reduce(larger, (leaf.convex for leaf in leaves)),
            concave=functools.reduce(smaller, (leaf.concave for leaf in leaves)),
        ).tightened()

    return jax.tree.map(
        intersect, *relaxations, is_leaf=lambda leaf: isinstance(leaf, Relaxation)
    )
