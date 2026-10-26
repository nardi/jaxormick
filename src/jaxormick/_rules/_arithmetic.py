"""Rules for sums, products, quotients and linear maps."""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jaxtyping import Array

from .._errors import UnsupportedPrimitiveError
from .._relaxation import Relaxation, larger, scale_tolerance, smaller
from ._composition import (
    as_relaxation,
    compose_concave,
    compose_convex,
    interval_only,
    select,
    substitute,
)
from ._registry import Operand, Params, relaxes


def negate(operand: Relaxation) -> Relaxation:
    """Returns the relaxation of `−operand`, which swaps the bounds and the two sides."""
    return Relaxation(
        lower=-operand.upper,
        upper=-operand.lower,
        convex=-operand.concave,
        concave=-operand.convex,
    )


def add(left: Relaxation, right: Relaxation) -> Relaxation:
    """Returns the exact relaxation of a sum, since sums of convex functions are convex."""
    return jax.tree.map(jnp.add, left, right)


def scale(operand: Relaxation, factor: Array) -> Relaxation:
    """Returns the exact relaxation of an operand multiplied by a constant factor."""
    positive = factor >= 0
    return Relaxation(
        lower=jnp.where(positive, factor * operand.lower, factor * operand.upper),
        upper=jnp.where(positive, factor * operand.upper, factor * operand.lower),
        convex=jnp.where(positive, factor * operand.convex, factor * operand.concave),
        concave=jnp.where(positive, factor * operand.concave, factor * operand.convex),
    )


def smaller_scaled(factor: Array, operand: Relaxation) -> Array:
    """Returns `min(factor · convex, factor · concave)`, chosen by the sign of `factor`."""
    return jnp.where(factor >= 0, factor * operand.convex, factor * operand.concave)


def larger_scaled(factor: Array, operand: Relaxation) -> Array:
    """Returns `max(factor · convex, factor · concave)`, chosen by the sign of `factor`."""
    return jnp.where(factor >= 0, factor * operand.concave, factor * operand.convex)


def bilinear(left: Relaxation, right: Relaxation) -> Relaxation:
    """Relaxes the product of two relaxed operands with McCormick's product rule.

    The formulas follow Mitsos, Chachuat and Barton (2009), who state the rule in terms
    of the bounds and the relaxations of both factors.
    """
    corner_products = jnp.stack(
        [
            left.lower * right.lower,
            left.lower * right.upper,
            left.upper * right.lower,
            left.upper * right.upper,
        ]
    )

    # Each term is the smaller or larger of a constant times the convex and the concave
    # side of a factor. Since `convex ≤ concave`, the sign of the constant decides which
    # side that is. Choosing by sign keeps `jax.grad` on one side when both tie, for the
    # reason given in `mid`. The terms can cancel to a value near zero, so the tie
    # tolerance scales with the corner products, which bound every term.
    tolerance = scale_tolerance(jnp.max(jnp.abs(corner_products), axis=0))
    convex = larger(
        smaller_scaled(right.lower, left)
        + smaller_scaled(left.lower, right)
        - left.lower * right.lower,
        smaller_scaled(right.upper, left)
        + smaller_scaled(left.upper, right)
        - left.upper * right.upper,
        tolerance=tolerance,
    )
    concave = smaller(
        larger_scaled(right.lower, left)
        + larger_scaled(left.upper, right)
        - left.upper * right.lower,
        larger_scaled(right.upper, left)
        + larger_scaled(left.lower, right)
        - left.lower * right.upper,
        tolerance=tolerance,
    )
    return Relaxation(
        lower=jnp.min(corner_products, axis=0),
        upper=jnp.max(corner_products, axis=0),
        convex=convex,
        concave=concave,
    ).tightened()


def multiply(left: Operand, right: Operand) -> Relaxation:
    """Relaxes a product, exactly when one factor is constant."""
    match left, right:
        case Relaxation(), Relaxation():
            return bilinear(left, right)
        case Relaxation(), _:
            return scale(left, jnp.asarray(right))
        case _, Relaxation():
            return scale(right, jnp.asarray(left))
        case _:
            raise AssertionError("A product without relaxed factors reached a rule.")


def reciprocal(operand: Relaxation) -> Relaxation:
    """Relaxes `1 / operand`.

    The reciprocal is convex and decreasing for positive arguments and concave and
    decreasing for negative ones. On an interval that contains zero it is unbounded,
    which the result reports as infinite bounds.
    """
    is_positive, is_negative = operand.lower > 0, operand.upper < 0
    positive_operand = substitute(operand, is_positive, 1.0)
    negative_operand = substitute(operand, is_negative, -1.0)
    positive = compose_convex(positive_operand, jnp.reciprocal, positive_operand.upper)
    negative = compose_concave(negative_operand, jnp.reciprocal, negative_operand.lower)
    unbounded = interval_only(
        jnp.full_like(operand.lower, -jnp.inf), jnp.full_like(operand.upper, jnp.inf)
    )
    return select(is_positive, positive, select(is_negative, negative, unbounded))


@relaxes("neg")
def neg_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `neg` exactly."""
    return negate(as_relaxation(operands[0]))


@relaxes("add", "add_any")
def add_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `add` and `add_any` exactly."""
    left, right = operands
    return add(as_relaxation(left), as_relaxation(right))


@relaxes("sub")
def sub_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `sub` as the sum of the first operand and the negated second operand."""
    left, right = operands
    return add(as_relaxation(left), negate(as_relaxation(right)))


@relaxes("mul")
def mul_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `mul`."""
    left, right = operands
    return multiply(left, right)


@relaxes("div")
def div_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `div` as the product with the reciprocal of the denominator."""
    match operands:
        case [numerator, Relaxation() as denominator]:
            return multiply(numerator, reciprocal(denominator))
        case [numerator, denominator]:
            return scale(as_relaxation(numerator), 1 / jnp.asarray(denominator))
        case _:
            raise AssertionError("`div` takes two operands.")


def dot_general_constant(
    primitive: Primitive,
    relaxed: Relaxation,
    constant: Array,
    relaxed_first: bool,
    params: Params,
) -> Relaxation:
    """Relaxes `dot_general` with one constant operand exactly.

    The constant splits into its positive and negative parts. Each output element is
    then a sum of relaxed values with nonnegative weights minus another such sum, so the
    convex side combines the convex sides of the first sum with the concave sides of the
    second.
    """

    def contract(relaxed_field: Array, weights: Array) -> Array:
        """Returns `dot_general` with the operands in their original order."""
        if relaxed_first:
            return primitive.bind(relaxed_field, weights, **params)
        return primitive.bind(weights, relaxed_field, **params)

    positive, negative = jnp.maximum(constant, 0), jnp.minimum(constant, 0)
    return Relaxation(
        lower=contract(relaxed.lower, positive) + contract(relaxed.upper, negative),
        upper=contract(relaxed.upper, positive) + contract(relaxed.lower, negative),
        convex=contract(relaxed.convex, positive) + contract(relaxed.concave, negative),
        concave=contract(relaxed.concave, positive)
        + contract(relaxed.convex, negative),
    )


def to_batch_free_contract(
    operand: Relaxation, batch_axes: Sequence[int], contract_axes: Sequence[int]
) -> tuple[Relaxation, tuple[int, ...]]:
    """Moves an operand's axes into the order batch, free, contracting and flattens each.

    Returns the rearranged operand of rank three and the shape of its free axes.
    """
    free_axes = [
        axis
        for axis in range(len(operand.shape))
        if axis not in batch_axes and axis not in contract_axes
    ]
    shape = operand.shape
    free_shape = tuple(shape[axis] for axis in free_axes)
    batch_size = math.prod(shape[axis] for axis in batch_axes)
    free_size = math.prod(free_shape)
    contract_size = math.prod(shape[axis] for axis in contract_axes)
    order = (*batch_axes, *free_axes, *contract_axes)
    rearranged = jax.tree.map(
        lambda field: jnp.transpose(field, order).reshape(
            batch_size, free_size, contract_size
        ),
        operand,
    )
    return rearranged, free_shape


def dot_general_bilinear(
    left: Relaxation, right: Relaxation, params: Params
) -> Relaxation:
    """Relaxes `dot_general` of two relaxed operands.

    Every output element is a sum of products of one element of each operand. Each
    product is relaxed with the bilinear rule, and the sum is exact.
    """
    (left_contract, right_contract), (left_batch, right_batch) = params[
        "dimension_numbers"
    ]
    left_flat, left_free_shape = to_batch_free_contract(left, left_batch, left_contract)
    right_flat, right_free_shape = to_batch_free_contract(
        right, right_batch, right_contract
    )
    products = bilinear(
        jax.tree.map(lambda field: field[:, :, None, :], left_flat),
        jax.tree.map(lambda field: field[:, None, :, :], right_flat),
    )
    batch_shape = tuple(left.shape[axis] for axis in left_batch)
    output_shape = (*batch_shape, *left_free_shape, *right_free_shape)
    return jax.tree.map(
        lambda field: jnp.sum(field, axis=-1).reshape(output_shape), products
    )


@relaxes("dot_general")
def dot_general_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `dot_general`, exactly when one operand is constant."""
    match operands:
        case [Relaxation() as left, Relaxation() as right]:
            return dot_general_bilinear(left, right, params)
        case [Relaxation() as left, right]:
            return dot_general_constant(
                primitive, left, jnp.asarray(right), True, params
            )
        case [left, Relaxation() as right]:
            return dot_general_constant(
                primitive, right, jnp.asarray(left), False, params
            )
        case _:
            raise AssertionError("`dot_general` takes two operands.")


@relaxes("eq", "ne")
def compare_identical(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Array:
    """Relaxes `eq` and `ne` of a relaxed value with itself.

    Such a comparison is how JAX tests for NaN, for example in `jax.nn.softplus`.
    Relaxations assume finite values, so the result is a constant. Any other comparison
    with a relaxed operand has no relaxation.
    """
    match operands:
        case [left, right] if left is right:
            return jnp.full(as_relaxation(left).shape, primitive.name == "eq")
        case _:
            raise UnsupportedPrimitiveError(
                f"`{primitive.name}` with an operand that depends on the variables is "
                "not supported, since the result would not be continuous."
            )
