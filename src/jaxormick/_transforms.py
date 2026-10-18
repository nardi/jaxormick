"""The function transformations `relax`, `interval` and `affine_relaxation`.

Every transformed function takes the reference point and the box first. Further
positional arguments are passed to the original function unchanged and are treated as
constants, so a function `f(x, parameters)` shares one compiled relaxation across many
parameter values.
"""

import functools
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ._interpreter import eval_jaxpr
from ._relaxation import Relaxation
from ._rules import as_relaxation

Mode = Literal["forward", "reverse"]
"""The mode of automatic differentiation used for subgradients."""


class AffineRelaxation(eqx.Module):
    """Affine under- and overestimators of a function over a box.

    The planes `convex + convex_gradient · (x − reference)` and
    `concave + concave_gradient · (x − reference)` bound the function from below and
    above at every point `x` of the box.
    """

    convex: Array
    """Value of the convex relaxation at the reference point."""
    convex_gradient: Array
    """Subgradient of the convex relaxation at the reference point."""
    concave: Array
    """Value of the concave relaxation at the reference point."""
    concave_gradient: Array
    """Supergradient of the concave relaxation at the reference point."""


def as_float_array(value: ArrayLike) -> Array:
    """Returns the value as an array of the default floating-point dtype or wider."""
    array = jnp.asarray(value)
    return array.astype(jnp.promote_types(array.dtype, jnp.result_type(float)))


def relax(function: Callable[..., PyTree[Array]]) -> Callable[..., PyTree[Relaxation]]:
    """Returns a function that computes McCormick relaxations of `function` over a box.

    `function` takes an array of variables, then any constant arguments. The returned
    function takes a reference point, the lower and upper bounds of the box, and then
    those constant arguments. It returns, in the structure of the output of `function`,
    a `Relaxation` with the interval bounds over the box and the values of the convex
    and concave relaxations at the reference point.

    The returned function is compiled with `jax.jit` and composes with `jax.vmap`,
    `jax.grad` and further `jax.jit`.
    """

    @functools.wraps(function)
    @jax.jit
    def relaxed(
        reference: ArrayLike, lower: ArrayLike, upper: ArrayLike, *arguments: Any
    ) -> PyTree[Relaxation]:
        reference = as_float_array(reference)
        lower = as_float_array(lower).astype(reference.dtype)
        upper = as_float_array(upper).astype(reference.dtype)
        closed, output_shape = jax.make_jaxpr(function, return_shape=True)(
            reference, *arguments
        )
        variables = Relaxation.variable(reference, lower, upper)
        outputs = eval_jaxpr(closed, [variables, *jax.tree.leaves(arguments)])
        return jax.tree.unflatten(
            jax.tree.structure(output_shape), [as_relaxation(o) for o in outputs]
        )

    return relaxed


def interval(
    function: Callable[..., PyTree[Array]],
) -> Callable[..., tuple[PyTree[Array], PyTree[Array]]]:
    """Returns a function that computes interval bounds of `function` over a box.

    The returned function takes the lower and upper bounds of the box, then the
    constant arguments of `function`, and returns the lower and upper bounds of its
    output. The convex and concave sides are not needed, so XLA removes them.
    """
    relaxed = relax(function)

    @functools.wraps(function)
    @jax.jit
    def bounds(
        lower: ArrayLike, upper: ArrayLike, *arguments: Any
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        lower, upper = as_float_array(lower), as_float_array(upper)
        relaxations = relaxed((lower + upper) / 2, lower, upper, *arguments)
        is_leaf = lambda leaf: isinstance(leaf, Relaxation)  # noqa: E731
        return (
            jax.tree.map(lambda r: r.lower, relaxations, is_leaf=is_leaf),
            jax.tree.map(lambda r: r.upper, relaxations, is_leaf=is_leaf),
        )

    return bounds


def jacobian_with_value(
    function: Callable[..., Any], mode: Mode
) -> Callable[..., tuple[Any, Any]]:
    """Returns the Jacobian of `function` in its first argument, together with its value.

    Forward mode pushes unit vectors through the linearized function, batched with
    `jax.vmap`. For relaxations it produces much smaller XLA programs than reverse mode,
    because the many selections and comparisons of the rules store nothing for a
    backward pass. Reverse mode is cheaper when the variables far outnumber the outputs.
    """

    def with_value(*arguments: Any) -> tuple[Any, Any]:
        value = function(*arguments)
        return value, value

    match mode:
        case "forward":
            return jax.jacfwd(with_value, has_aux=True)
        case "reverse":
            return jax.jacrev(with_value, has_aux=True)
        case _:
            raise ValueError(f"`mode` must be 'forward' or 'reverse', not {mode!r}.")


def one_side(
    function: Callable[..., Array], side: Literal["convex", "concave"], mode: Mode
) -> Callable[..., tuple[Array, Array]]:
    """Returns a function that computes one side of the relaxation and its gradient."""
    relaxed = relax(function)

    def value(reference: Array, lower: Array, upper: Array, *arguments: Any) -> Array:
        return getattr(relaxed(reference, lower, upper, *arguments), side)

    differentiated = jacobian_with_value(value, mode)

    @functools.wraps(function)
    @jax.jit
    def value_and_gradient(
        reference: ArrayLike, lower: ArrayLike, upper: ArrayLike, *arguments: Any
    ) -> tuple[Array, Array]:
        gradient, side_value = differentiated(
            as_float_array(reference), lower, upper, *arguments
        )
        return side_value, gradient

    return value_and_gradient


def convex_side(
    function: Callable[..., Array], mode: Mode = "forward"
) -> Callable[..., tuple[Array, Array]]:
    """Returns a function that computes the convex relaxation and a subgradient.

    The returned function takes the same arguments as the one from `relax` and returns
    the value of the convex relaxation of `function` at the reference point and its
    gradient there. `function` has to return a single array. Computing one side alone
    lets XLA remove the computations that only the other side needs.
    """
    return one_side(function, "convex", mode)


def concave_side(
    function: Callable[..., Array], mode: Mode = "forward"
) -> Callable[..., tuple[Array, Array]]:
    """Returns a function that computes the concave relaxation and a supergradient.

    This mirrors `convex_side`.
    """
    return one_side(function, "concave", mode)


def affine_relaxation(
    function: Callable[..., Array], mode: Mode = "forward"
) -> Callable[..., AffineRelaxation]:
    """Returns a function that computes affine under- and overestimators of `function`.

    The returned function takes the same arguments as the one from `relax` and returns
    an `AffineRelaxation`. Its planes touch the convex and concave relaxations at the
    reference point and bound `function` on the whole box, since a tangent plane of a
    convex function stays below it. `function` has to return a single array. Both sides
    are differentiated in one pass, in forward mode by default.
    """
    relaxed = relax(function)

    def sides(
        reference: Array, lower: Array, upper: Array, *arguments: Any
    ) -> tuple[Array, Array]:
        relaxation = relaxed(reference, lower, upper, *arguments)
        return relaxation.convex, relaxation.concave

    differentiated = jacobian_with_value(sides, mode)

    @functools.wraps(function)
    @jax.jit
    def planes(
        reference: ArrayLike, lower: ArrayLike, upper: ArrayLike, *arguments: Any
    ) -> AffineRelaxation:
        (convex_gradient, concave_gradient), (convex, concave) = differentiated(
            as_float_array(reference), lower, upper, *arguments
        )
        return AffineRelaxation(
            convex=convex,
            convex_gradient=convex_gradient,
            concave=concave,
            concave_gradient=concave_gradient,
        )

    return planes
