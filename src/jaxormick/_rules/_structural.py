"""Rules for primitives that rearrange elements, select between them or take extrema."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from .._errors import UnsupportedPrimitiveError
from .._relaxation import Relaxation, larger
from ._arithmetic import add, bilinear, negate, scale
from ._composition import as_relaxation, is_relaxed
from ._elementary import absolute
from ._registry import Operand, Params, register_rule, relaxes

FIELD_NAMES = ("lower", "upper", "convex", "concave")
"""The names of the fields of `Relaxation`."""

STRUCTURAL_PRIMITIVES = (
    "reshape",
    "broadcast_in_dim",
    "transpose",
    "squeeze",
    "expand_dims",
    "slice",
    "dynamic_slice",
    "concatenate",
    "stack",
    "pad",
    "gather",
    "rev",
    "copy",
    "copy_p",
    "convert_element_type",
    "reduce_sum",
    "cumsum",
    "dynamic_update_slice",
    "scatter-add",
)
"""Primitives that rearrange elements or sum them with nonnegative weights."""


def structural_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Applies a primitive to each field separately.

    This is exact for primitives that rearrange or copy elements, such as `reshape` or
    `slice`, and for sums with nonnegative weights, such as `reduce_sum`. Plain operands
    such as indices or padding values are passed unchanged.
    """
    fields = {
        name: primitive.bind(
            *(
                getattr(operand, name) if is_relaxed(operand) else operand
                for operand in operands
            ),
            **params,
        )
        for name in FIELD_NAMES
    }
    return Relaxation(**fields)


for name in STRUCTURAL_PRIMITIVES:
    register_rule(name, structural_rule)


@relaxes("split", "unstack")
def split_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> list[Relaxation]:
    """Relaxes `split` and `unstack` exactly by splitting each field the same way.

    `jax.hessian` produces these primitives.
    """
    pieces = [
        primitive.bind(getattr(as_relaxation(operands[0]), name), **params)
        for name in FIELD_NAMES
    ]
    return [
        Relaxation(**dict(zip(FIELD_NAMES, fields, strict=True)))
        for fields in zip(*pieces, strict=True)
    ]


@relaxes("scatter")
def scatter_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `scatter`, which `x.at[i].set(y)` produces.

    With repeated indices, `scatter` keeps one of the updates and does not say which.
    Scattering each field on its own could then mix fields of different updates. The
    rule stacks the fields along a new last axis instead and scatters them together.
    That axis is a window dimension, so each update moves as a whole.
    """
    operand, indices, updates = operands
    stacked_operand, stacked_updates = (
        jnp.stack([getattr(as_relaxation(value), name) for name in FIELD_NAMES], -1)
        for value in (operand, updates)
    )
    numbers = params["dimension_numbers"]
    window_dims = (*numbers.update_window_dims, stacked_updates.ndim - 1)
    stacked = primitive.bind(
        stacked_operand,
        indices,
        stacked_updates,
        **{
            **params,
            "dimension_numbers": numbers._replace(update_window_dims=window_dims),
        },
    )
    return Relaxation(
        **{name: stacked[..., index] for index, name in enumerate(FIELD_NAMES)}
    )


@relaxes("select_n")
def select_n_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `select_n` when its predicate does not depend on the variables."""
    match operands:
        case [Relaxation(), *_]:
            raise UnsupportedPrimitiveError(
                "`select_n` with a predicate that depends on the variables is not "
                "supported."
            )
        case [predicate, *cases]:
            return structural_rule(
                primitive,
                [predicate, *(as_relaxation(case) for case in cases)],
                params,
            )
        case _:
            raise AssertionError("`select_n` takes a predicate and cases.")


def maximum(left: Relaxation, right: Relaxation) -> Relaxation:
    """Relaxes `max(left, right)`.

    The concave side comes from `max(a, b) = (a + b + |a − b|) / 2`, relaxed with the
    rules of `add` and `abs`. That formula also defines a convex side. The maximum of the
    two convex sides is another one, often tighter, so the result takes the larger.
    """
    difference = add(left, negate(right))
    formula = scale(add(add(left, right), absolute(difference)), jnp.asarray(0.5))
    return Relaxation(
        lower=jnp.maximum(left.lower, right.lower),
        upper=jnp.maximum(left.upper, right.upper),
        convex=larger(formula.convex, larger(left.convex, right.convex)),
        concave=formula.concave,
    ).tightened()


@relaxes("max")
def max_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `max`, which `jnp.maximum`, `jax.nn.relu` and `jnp.clip` produce."""
    left, right = operands
    return maximum(as_relaxation(left), as_relaxation(right))


@relaxes("min")
def min_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `min` as `−max(−left, −right)`."""
    left, right = operands
    return negate(maximum(negate(as_relaxation(left)), negate(as_relaxation(right))))


def reduce_pairwise(
    operand: Relaxation,
    axes: Sequence[int],
    combine: Callable[[Relaxation, Relaxation], Relaxation],
) -> Relaxation:
    """Reduces `operand` over `axes` by combining halves until one slice is left.

    The reduced axes are moved to the front and merged into one. Each level combines
    the first half of the slices with the second and keeps a leftover slice as it is.
    The result has `log₂ n` levels for `n` slices.
    """
    front = tuple(range(len(axes)))

    def merge_axes(field: jax.Array) -> jax.Array:
        moved = jnp.moveaxis(field, tuple(axes), front)
        return moved.reshape(-1, *moved.shape[len(axes) :])

    slices = jax.tree.map(merge_axes, operand)
    while (count := slices.lower.shape[0]) > 1:
        half = count // 2
        first = jax.tree.map(lambda field: field[:half], slices)
        second = jax.tree.map(lambda field: field[half : 2 * half], slices)
        leftover = jax.tree.map(lambda field: field[2 * half :], slices)
        slices = jax.tree.map(
            lambda combined, rest: jnp.concatenate([combined, rest]),
            combine(first, second),
            leftover,
        )
    return jax.tree.map(lambda field: field[0], slices)


@relaxes("reduce_max")
def reduce_max_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `reduce_max`, which `jnp.max` produces, with the rule of `max`."""
    return reduce_pairwise(as_relaxation(operands[0]), params["axes"], maximum)


@relaxes("reduce_min")
def reduce_min_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `reduce_min`, which `jnp.min` produces, as `−max(−x)`."""
    negated = negate(as_relaxation(operands[0]))
    return negate(reduce_pairwise(negated, params["axes"], maximum))


@relaxes("reduce_prod")
def reduce_prod_rule(
    primitive: Primitive, operands: Sequence[Operand], params: Params
) -> Relaxation:
    """Relaxes `reduce_prod`, which `jnp.prod` produces, with the product rule."""
    return reduce_pairwise(as_relaxation(operands[0]), params["axes"], bilinear)
