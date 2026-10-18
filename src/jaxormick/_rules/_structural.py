"""Rules for primitives that rearrange elements, select between them or take extrema."""

from collections.abc import Sequence

import jax.numpy as jnp
from jax.extend.core import Primitive

from .._errors import UnsupportedPrimitiveError
from .._relaxation import Relaxation
from ._arithmetic import add, negate, scale
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
    "pad",
    "gather",
    "rev",
    "copy",
    "copy_p",
    "convert_element_type",
    "reduce_sum",
    "cumsum",
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
        convex=jnp.maximum(formula.convex, jnp.maximum(left.convex, right.convex)),
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
