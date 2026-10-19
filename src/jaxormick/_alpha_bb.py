"""αBB relaxations, which add a quadratic to a function to make it convex or concave."""

import jax.numpy as jnp
from jaxtyping import Array


def gershgorin_shifts(
    hessian_lower: Array, hessian_upper: Array, widths: Array
) -> tuple[Array, Array]:
    """Returns the shifts of the convex and the concave side, one per variable.

    The Hessian bounds have the shape `(..., n, n)` and the widths of the box the shape
    `(n,)`. The scaled Gershgorin theorem of Adjiman et al. (1998) bounds the
    eigenvalues of every matrix in the interval Hessian. A shift of `α_i` on the
    diagonal then makes the Hessian of `f + Σ α_i (x_i − l_i)(x_i − u_i)` positive
    semidefinite. The concave shift does the same for `−f`. A variable with zero width
    gets a shift of zero, since its quadratic term vanishes. A shift is infinite or NaN
    where the Hessian bounds are not finite.
    """
    size = widths.shape[0]
    has_width = widths > 0
    is_off_diagonal = ~jnp.eye(size, dtype=bool) & has_width
    magnitude = jnp.maximum(jnp.abs(hessian_lower), jnp.abs(hessian_upper))
    scaled = jnp.where(is_off_diagonal, magnitude * widths, 0.0)
    radius = jnp.sum(scaled, axis=-1) / jnp.where(has_width, widths, 1.0)
    diagonal_lower = jnp.diagonal(hessian_lower, axis1=-2, axis2=-1)
    diagonal_upper = jnp.diagonal(hessian_upper, axis1=-2, axis2=-1)
    convex_shift = jnp.maximum(0.0, (radius - diagonal_lower) / 2)
    concave_shift = jnp.maximum(0.0, (radius + diagonal_upper) / 2)
    return (
        jnp.where(has_width, convex_shift, 0.0),
        jnp.where(has_width, concave_shift, 0.0),
    )


def shifted(value: Array, shifts: Array, quadratic: Array, fallback: float) -> Array:
    """Returns `value + shifts · quadratic`, or `fallback` where a shift is not finite.

    Infinite shifts are replaced by zero before the product, so that the gradient of
    the result stays finite.
    """
    is_finite = jnp.all(jnp.isfinite(shifts), axis=-1)
    finite_shifts = jnp.where(is_finite[..., None], shifts, 0.0)
    return jnp.where(is_finite, value + finite_shifts @ quadratic, fallback)


def alpha_bb_sides(
    value: Array,
    reference: Array,
    lower: Array,
    upper: Array,
    hessian_lower: Array,
    hessian_upper: Array,
) -> tuple[Array, Array]:
    """Returns the αBB convex and concave relaxations at the reference point.

    `value` is the function at the reference point, of any shape. The Hessian bounds
    have the shape of `value` followed by the shape of the variables twice. A side is
    infinite where the Hessian bounds are not finite.
    """
    size = reference.size
    lower = jnp.broadcast_to(lower, reference.shape).ravel()
    upper = jnp.broadcast_to(upper, reference.shape).ravel()
    matrix_shape = (*value.shape, size, size)
    convex_shift, concave_shift = gershgorin_shifts(
        hessian_lower.reshape(matrix_shape),
        hessian_upper.reshape(matrix_shape),
        upper - lower,
    )

    # The product is nonpositive on the box and zero at its corners.
    point = reference.ravel()
    quadratic = (point - lower) * (point - upper)
    return (
        shifted(value, convex_shift, quadratic, -jnp.inf),
        shifted(value, -concave_shift, quadratic, jnp.inf),
    )
