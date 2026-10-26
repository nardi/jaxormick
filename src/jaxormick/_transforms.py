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

from ._alpha_bb import alpha_bb_sides
from ._errors import UnsupportedPrimitiveError
from ._implicit import root_tape
from ._interpreter import eval_jaxpr, eval_tracked_jaxpr
from ._relaxation import Relaxation, intersect_relaxations
from ._rules import as_relaxation
from ._tightening import Tracked, tracked_variables

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


def relax(
    function: Callable[..., PyTree[Array]],
    *,
    alpha_bb: bool = False,
    tighten_bounds: bool = False,
    enclosures: bool = False,
) -> Callable[..., Any]:
    """Returns a function that computes McCormick relaxations of `function` over a box.

    `function` takes an array of variables, then any constant arguments. The returned
    function takes a reference point, the lower and upper bounds of the box, and then
    those constant arguments. It returns, in the structure of the output of `function`,
    a `Relaxation` with the interval bounds over the box and the values of the convex
    and concave relaxations at the reference point.

    The returned function is compiled with `jax.jit` and composes with `jax.vmap`,
    `jax.grad` and further `jax.jit`.

    With `alpha_bb=True`, the result is intersected with the αBB relaxation of
    Adjiman et al. (1998), which bounds the Hessian of `function` over the box with
    `interval`. The result is often tighter for smooth functions on small boxes. It
    needs a function that returns a single array, and every primitive in its Hessian
    needs a rule.

    With `tighten_bounds=True`, the interval of every intermediate value is narrowed by
    the affine relaxations at the middle of the box, after Najman and Mitsos (2019).
    Narrower intermediate intervals also tighten the relaxations that depend on them.
    This costs about as much as one forward-mode derivative of the relaxation.

    With `enclosures=True`, the returned function takes a flat array of enclosures of
    the roots of `lax.custom_root` after the box, and returns them after the relaxations.
    A root enclosed by a box that contains this one needs no search for an enclosure, and
    the verified enclosures come back for the boxes inside this one. Use
    `root_enclosure_size` for the length of the array. Entries that are NaN stand for no
    enclosure. Only roots at the top level of `function` take part, and `tighten_bounds`
    cannot be combined with enclosures.
    """
    if enclosures and tighten_bounds:
        raise ValueError(
            "Enclosures of roots cannot be combined with `tighten_bounds`."
        )
    relax_at = relaxation_tracer(function, alpha_bb, tighten_bounds)

    if enclosures:

        @functools.wraps(function)
        @jax.jit
        def relaxed_with_enclosures(
            reference: ArrayLike,
            lower: ArrayLike,
            upper: ArrayLike,
            enclosure: ArrayLike,
            *arguments: Any,
        ) -> tuple[PyTree[Relaxation], Array]:
            enclosure = as_float_array(enclosure)
            with root_tape(enclosure) as tape:
                relaxations = relax_at(reference, lower, upper, arguments)
            return relaxations, tape.collect(enclosure.dtype)

        return relaxed_with_enclosures

    @functools.wraps(function)
    @jax.jit
    def relaxed(
        reference: ArrayLike, lower: ArrayLike, upper: ArrayLike, *arguments: Any
    ) -> PyTree[Relaxation]:
        return relax_at(reference, lower, upper, arguments)

    return relaxed


def relaxation_tracer(
    function: Callable[..., PyTree[Array]], alpha_bb: bool, tighten_bounds: bool
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, Any], PyTree[Relaxation]]:
    """Returns the function that traces the relaxation of `function` for `relax`.

    It takes the reference point, the box and the tuple of constant arguments.
    """
    hessian_bounds = interval(jax.hessian(function)) if alpha_bb else None

    def relax_at(
        reference: ArrayLike, lower: ArrayLike, upper: ArrayLike, arguments: Any
    ) -> PyTree[Relaxation]:
        reference = as_float_array(reference)
        lower = as_float_array(lower).astype(reference.dtype)
        upper = as_float_array(upper).astype(reference.dtype)
        closed, output_shape = jax.make_jaxpr(function, return_shape=True)(
            reference, *arguments
        )
        variables = Relaxation.variable(reference, lower, upper)
        constants = jax.tree.leaves(arguments)
        if tighten_bounds:
            tracked, box = tracked_variables(reference, lower, upper)
            outputs = [
                output.at_reference if isinstance(output, Tracked) else output
                for output in eval_tracked_jaxpr(closed, [tracked, *constants], box)
            ]
        else:
            outputs = eval_jaxpr(closed, [variables, *constants])
        relaxations = [as_relaxation(output) for output in outputs]
        structure = jax.tree.structure(output_shape)
        if hessian_bounds is not None:
            if not jax.tree_util.treedef_is_leaf(structure):
                raise ValueError("`alpha_bb` needs a function that returns one array.")
            relaxations = [
                with_alpha_bb(
                    function, hessian_bounds, relaxations[0], variables, *arguments
                )
            ]
        return jax.tree.unflatten(structure, relaxations)

    return relax_at


def root_enclosure_size(
    function: Callable[..., PyTree[Array]],
    reference: ArrayLike | jax.ShapeDtypeStruct,
    lower: ArrayLike | jax.ShapeDtypeStruct,
    upper: ArrayLike | jax.ShapeDtypeStruct,
    *arguments: Any,
) -> int:
    """Returns the length of the enclosures array of `relax(function, enclosures=True)`.

    Every root of `lax.custom_root` at the top level of `function` with `k` entries takes
    `2k` numbers, the lower and then the upper bounds of its enclosure. The arguments can
    be arrays or `jax.ShapeDtypeStruct`s.
    """
    trace = relaxation_tracer(function, False, False)

    def measure(*values: Any) -> PyTree[Relaxation]:
        with root_tape(None) as tape:
            result = trace(values[0], values[1], values[2], values[3:])
        sizes.append(tape.total())
        return result

    sizes: list[int] = []
    jax.eval_shape(measure, reference, lower, upper, *arguments)
    return sizes[0]


def with_alpha_bb(
    function: Callable[..., Array],
    hessian_bounds: Callable[..., tuple[Array, Array]],
    relaxation: Relaxation,
    variables: Relaxation,
    *arguments: Any,
) -> Relaxation:
    """Returns the intersection of a relaxation with the αBB relaxation of `function`.

    `variables` is the relaxation of the variables, which holds the reference point and
    the box.
    """
    try:
        hessian_lower, hessian_upper = hessian_bounds(
            variables.lower, variables.upper, *arguments
        )
    except UnsupportedPrimitiveError as error:
        raise UnsupportedPrimitiveError(
            f"`alpha_bb` could not bound the Hessian of the function. {error}"
        ) from error
    convex, concave = alpha_bb_sides(
        jnp.asarray(function(variables.convex, *arguments), relaxation.dtype),
        variables.convex,
        variables.lower,
        variables.upper,
        hessian_lower,
        hessian_upper,
    )
    alpha_bb = Relaxation(
        lower=relaxation.lower,
        upper=relaxation.upper,
        convex=convex,
        concave=concave,
    )
    return intersect_relaxations(relaxation, alpha_bb)


def interval(
    function: Callable[..., PyTree[Array]],
    *,
    tighten_bounds: bool = False,
    enclosures: bool = False,
) -> Callable[..., Any]:
    """Returns a function that computes interval bounds of `function` over a box.

    The returned function takes the lower and upper bounds of the box, then the
    constant arguments of `function`, and returns the lower and upper bounds of its
    output. The convex and concave sides are not needed, so XLA removes them, unless
    `tighten_bounds`, which is passed to `relax`, needs them.

    With `enclosures=True`, the returned function takes the enclosures of the roots after
    the box, as `relax` does, and returns the verified ones after the bounds.
    """
    if enclosures:
        relaxed_with_enclosures = relax(
            function, tighten_bounds=tighten_bounds, enclosures=True
        )

        @functools.wraps(function)
        @jax.jit
        def bounds_with_enclosures(
            lower: ArrayLike, upper: ArrayLike, enclosure: ArrayLike, *arguments: Any
        ) -> tuple[PyTree[Array], PyTree[Array], Array]:
            lower, upper = as_float_array(lower), as_float_array(upper)
            relaxations, found = relaxed_with_enclosures(
                (lower + upper) / 2, lower, upper, enclosure, *arguments
            )
            is_leaf = lambda leaf: isinstance(leaf, Relaxation)  # noqa: E731
            return (
                jax.tree.map(lambda r: r.lower, relaxations, is_leaf=is_leaf),
                jax.tree.map(lambda r: r.upper, relaxations, is_leaf=is_leaf),
                found,
            )

        return bounds_with_enclosures

    relaxed = relax(function, tighten_bounds=tighten_bounds)

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
    function: Callable[..., Array],
    side: Literal["convex", "concave"],
    mode: Mode,
    alpha_bb: bool,
    tighten_bounds: bool,
    enclosures: bool = False,
) -> Callable[..., tuple[Any, ...]]:
    """Returns a function that computes one side of the relaxation and its gradient."""
    if enclosures:
        relaxed_with_enclosures = relax(
            function,
            alpha_bb=alpha_bb,
            tighten_bounds=tighten_bounds,
            enclosures=True,
        )

        def value_with_enclosures(
            reference: Array,
            lower: Array,
            upper: Array,
            enclosure: Array,
            *arguments: Any,
        ) -> tuple[Array, tuple[Array, Array]]:
            relaxation, found = relaxed_with_enclosures(
                reference, lower, upper, enclosure, *arguments
            )
            side_value = getattr(relaxation, side)
            return side_value, (side_value, found)

        differentiated_with_enclosures = {
            "forward": jax.jacfwd,
            "reverse": jax.jacrev,
        }.get(mode)
        if differentiated_with_enclosures is None:
            raise ValueError(f"`mode` must be 'forward' or 'reverse', not {mode!r}.")
        differentiate = differentiated_with_enclosures(
            value_with_enclosures, has_aux=True
        )

        @functools.wraps(function)
        @jax.jit
        def value_gradient_enclosures(
            reference: ArrayLike,
            lower: ArrayLike,
            upper: ArrayLike,
            enclosure: ArrayLike,
            *arguments: Any,
        ) -> tuple[Array, Array, Array]:
            gradient, (side_value, found) = differentiate(
                as_float_array(reference), lower, upper, enclosure, *arguments
            )
            return side_value, gradient, found

        return value_gradient_enclosures

    relaxed = relax(function, alpha_bb=alpha_bb, tighten_bounds=tighten_bounds)

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
    function: Callable[..., Array],
    mode: Mode = "forward",
    *,
    alpha_bb: bool = False,
    tighten_bounds: bool = False,
    enclosures: bool = False,
) -> Callable[..., tuple[Any, ...]]:
    """Returns a function that computes the convex relaxation and a subgradient.

    The returned function takes the same arguments as the one from `relax` and returns
    the value of the convex relaxation of `function` at the reference point and its
    gradient there. `function` has to return a single array. Computing one side alone
    lets XLA remove the computations that only the other side needs. `alpha_bb` and
    `tighten_bounds` are passed to `relax`.

    With `enclosures=True`, the function takes the enclosures of the roots after the box
    and returns the verified ones as a third result. See `relax`.
    """
    return one_side(function, "convex", mode, alpha_bb, tighten_bounds, enclosures)


def concave_side(
    function: Callable[..., Array],
    mode: Mode = "forward",
    *,
    alpha_bb: bool = False,
    tighten_bounds: bool = False,
    enclosures: bool = False,
) -> Callable[..., tuple[Any, ...]]:
    """Returns a function that computes the concave relaxation and a supergradient.

    This mirrors `convex_side`.
    """
    return one_side(function, "concave", mode, alpha_bb, tighten_bounds, enclosures)


def affine_relaxation(
    function: Callable[..., Array],
    mode: Mode = "forward",
    *,
    alpha_bb: bool = False,
    tighten_bounds: bool = False,
) -> Callable[..., AffineRelaxation]:
    """Returns a function that computes affine under- and overestimators of `function`.

    The returned function takes the same arguments as the one from `relax` and returns
    an `AffineRelaxation`. Its planes touch the convex and concave relaxations at the
    reference point and bound `function` on the whole box, since a tangent plane of a
    convex function stays below it. `function` has to return a single array. Both sides
    are differentiated in one pass, in forward mode by default. `alpha_bb` and
    `tighten_bounds` are passed to `relax`.
    """
    relaxed = relax(function, alpha_bb=alpha_bb, tighten_bounds=tighten_bounds)

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
