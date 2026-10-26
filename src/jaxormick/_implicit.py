"""Relaxations of implicit functions, whose value solves a system of equations.

The solution `x(p)` of `r(p, x) = 0` is enclosed by the parametric Krawczyk operator,
after Krawczyk (1969) and Neumaier (1990). An enclosure of the solution over the box
comes from ε-inflation around the solution at the middle of the box. Relaxed Krawczyk
steps then tighten constant relaxations of the solution, after Stuber, Scott and Barton
(2015).
"""

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.lax.control_flow.solves import CustomRoot
from jax.extend.core import ClosedJaxpr, JaxprEqn, Var, jaxpr_as_fun
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from ._errors import UnsupportedPrimitiveError
from ._relaxation import Relaxation, intersect_relaxations
from ._rules import Operand, as_relaxation

INFLATION_STEPS = 10
"""The number of ε-inflation steps that try to verify an enclosure of the solution."""

RELAXATION_STEPS = 5
"""The number of relaxed Krawczyk steps that tighten the relaxations of the solution."""

INFLATION_FACTOR = 1.1
"""The factor by which each ε-inflation step widens the candidate enclosure."""

Residual: TypeAlias = Callable[[Sequence[Array], Array], Array]
"""A residual `r(p, x)` of flat solutions `x`, whose root is the implicit function."""

Evaluate: TypeAlias = Callable[[ClosedJaxpr, Sequence[Operand]], list[Operand]]
"""Relaxes a jaxpr on a mix of relaxed and plain operands."""


def is_custom_root(equation: JaxprEqn) -> bool:
    """Returns whether an equation is a `lax.custom_root` that exposes its residual."""
    return equation.primitive.name == "call_hi_primitive" and isinstance(
        equation.params["_prim"], CustomRoot
    )


def middle_value(value: Operand) -> Array:
    """Returns the middle of the interval of a relaxed value, or zero where infinite.

    A plain value is its own middle.
    """
    if not isinstance(value, Relaxation):
        return value
    middle = (value.lower + value.upper) / 2
    return jnp.where(jnp.isfinite(middle), middle, 0).astype(value.dtype)


def relax_custom_root(
    equation: JaxprEqn, inputs: Sequence[Operand], evaluate: Evaluate
) -> list[Operand]:
    """Relaxes the root of `lax.custom_root` as a function of the residual's constants.

    The solver is assumed to find the root that the enclosure verifies. The initial
    guess does not change the root, so its relaxation is not used.
    """
    root = equation.params["_prim"]
    if root.has_aux:
        raise UnsupportedPrimitiveError(
            "`lax.custom_root` with `has_aux=True` is not supported."
        )
    subtrees = jax.tree_util.treedef_children(root.in_tree)
    solve_consts, guess, residual_consts, _ = split_operands(inputs, subtrees)
    middle_solution = root.solve(
        jax.tree.unflatten(
            subtrees[0], [middle_value(value) for value in solve_consts]
        ),
        jax.tree.unflatten(subtrees[1], [middle_value(value) for value in guess]),
    )
    flat_solution, unravel = ravel_pytree(middle_solution)

    def residual(parameters: Sequence[Array], solution: Array) -> Array:
        """Returns the flat residual of the root's equations."""
        values = root.f(jax.tree.unflatten(subtrees[2], parameters), unravel(solution))
        flat_values, _ = ravel_pytree(values)
        return flat_values

    solution = relax_solution(residual, residual_consts, flat_solution, evaluate)
    return unravel_relaxation(solution, unravel)


def split_operands(
    operands: Sequence[Operand], subtrees: Sequence[jax.tree_util.PyTreeDef]
) -> list[list[Operand]]:
    """Returns the operands split into the leaves of consecutive subtrees."""
    groups, start = [], 0
    for subtree in subtrees:
        groups.append(list(operands[start : start + subtree.num_leaves]))
        start += subtree.num_leaves
    return groups


def relax_linear_solve(
    equation: JaxprEqn, inputs: Sequence[Operand], evaluate: Evaluate
) -> list[Operand]:
    """Relaxes the solution of `lax.custom_linear_solve` as a function of its operands.

    The solution depends on the constants of `matvec` and on the right-hand side. The
    constants of the solvers only help to solve the system, so they are not used.
    """
    lengths, jaxprs = equation.params["const_lengths"], equation.params["jaxprs"]
    matvec_consts = list(inputs[: lengths.matvec])
    right_hand_side = list(inputs[sum(lengths) :])
    matvec = jaxpr_as_fun(jaxprs.matvec)
    flat_solution, unravel = ravel_pytree(
        [zeros_of(variable) for variable in equation.outvars]
    )

    def residual(parameters: Sequence[Array], solution: Array) -> Array:
        """Returns the flat residual `A x − b` of the linear system."""
        products, _ = ravel_pytree(
            matvec(*parameters[: lengths.matvec], *unravel(solution))
        )
        targets, _ = ravel_pytree(list(parameters[lengths.matvec :]))
        return products - targets

    solution = relax_solution(
        residual, [*matvec_consts, *right_hand_side], flat_solution, evaluate
    )
    return unravel_relaxation(solution, unravel)


def unravel_relaxation(
    relaxation: Relaxation, unravel: Callable[[Array], Any]
) -> list[Operand]:
    """Returns the relaxation of a flat solution split into the leaves of the solution."""
    fields = [
        jax.tree.leaves(unravel(field))
        for field in (
            relaxation.lower,
            relaxation.upper,
            relaxation.convex,
            relaxation.concave,
        )
    ]
    return [Relaxation(*leaf_fields) for leaf_fields in zip(*fields, strict=True)]


def relax_solution(
    residual: Residual,
    parameters: Sequence[Operand],
    start: Array,
    evaluate: Evaluate,
) -> Relaxation:
    """Relaxes the solution of `residual(parameters, x) = 0` over the box.

    `start` approximates the solution at the middle of the parameters. Two Newton steps
    refine it. The result is infinite where no enclosure is verified.
    """
    middle_parameters = [middle_value(value) for value in parameters]
    jacobian = jax.jacfwd(residual, argnums=1)
    preconditioner = finite_or_zero(jnp.linalg.inv(jacobian(middle_parameters, start)))
    solution = start
    for _ in range(2):
        solution = solution - preconditioner @ residual(middle_parameters, solution)
    solution = finite_or_zero(solution)
    identity = jnp.eye(solution.size, dtype=solution.dtype)

    def krawczyk(
        parameters: Sequence[Array],
        middle: Array,
        jacobian_point: Array,
        iterate: Array,
    ) -> Array:
        """Returns the Krawczyk operator, which maps the solution to itself."""
        contraction = identity - preconditioner @ jacobian(parameters, jacobian_point)
        return (
            middle
            - preconditioner @ residual(parameters, middle)
            + contraction @ (iterate - middle)
        )

    shapes = [
        jax.ShapeDtypeStruct(jnp.shape(value), dtype_of(value)) for value in parameters
    ]
    point = jax.ShapeDtypeStruct(solution.shape, solution.dtype)
    operator = jax.make_jaxpr(krawczyk)(shapes, point, point, point)

    def image(enclosure: Relaxation, iterate: Relaxation) -> Relaxation:
        """Relaxes the Krawczyk operator with the solution in `enclosure`."""
        middle = (enclosure.lower + enclosure.upper) / 2
        [output] = evaluate(operator, [*parameters, middle, enclosure, iterate])
        return jax.tree.map(
            lambda field: field.astype(solution.dtype), as_relaxation(output)
        )

    def inflation_step(
        _: int, state: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Inflates the candidate enclosure and checks whether Krawczyk contracts it."""
        lower, upper, verified_lower, verified_upper, is_verified = state
        middle, radius = (lower + upper) / 2, (upper - lower) / 2
        slack = jnp.sqrt(jnp.finfo(solution.dtype).eps) * (1 + jnp.abs(middle))
        radius = INFLATION_FACTOR * radius + slack
        lower, upper = middle - radius, middle + radius
        contracted = image(constant(lower, upper), constant(lower, upper))
        is_inside = jnp.all(contracted.lower > lower) & jnp.all(
            contracted.upper < upper
        )
        is_new = is_inside & ~is_verified
        return (
            jnp.where(jnp.isfinite(contracted.lower), contracted.lower, lower),
            jnp.where(jnp.isfinite(contracted.upper), contracted.upper, upper),
            jnp.where(is_new, contracted.lower, verified_lower),
            jnp.where(is_new, contracted.upper, verified_upper),
            is_verified | is_inside,
        )

    _, _, lower, upper, is_verified = lax.fori_loop(
        0,
        INFLATION_STEPS,
        inflation_step,
        (solution, solution, solution, solution, jnp.array(False)),
    )

    def relaxation_step(_: int, iterate: Relaxation) -> Relaxation:
        """Intersects the relaxation of the solution with its Krawczyk image."""
        return intersect_relaxations(iterate, image(constant(lower, upper), iterate))

    relaxation = lax.fori_loop(
        0, RELAXATION_STEPS, relaxation_step, constant(lower, upper)
    )
    infinity = jnp.full_like(solution, jnp.inf)
    return Relaxation(
        lower=jnp.where(is_verified, relaxation.lower, -infinity),
        upper=jnp.where(is_verified, relaxation.upper, infinity),
        convex=jnp.where(is_verified, relaxation.convex, -infinity),
        concave=jnp.where(is_verified, relaxation.concave, infinity),
    )


def constant(lower: Array, upper: Array) -> Relaxation:
    """Returns the relaxation by the bounds alone, valid for any value between them."""
    return Relaxation(lower=lower, upper=upper, convex=lower, concave=upper)


def finite_or_zero(value: Array) -> Array:
    """Returns the value with zero where it is not finite.

    A zero preconditioner makes the Krawczyk operator the identity, which never
    verifies an enclosure, so a singular Jacobian fails safely.
    """
    return jnp.where(jnp.isfinite(value), value, 0)


def zeros_of(variable: Var) -> Array:
    """Returns zeros of the shape and dtype of a jaxpr variable."""
    aval = variable.aval
    return jnp.zeros(getattr(aval, "shape"), getattr(aval, "dtype"))


def dtype_of(value: Operand) -> jnp.dtype:
    """Returns the dtype of a relaxed or plain value."""
    return value.dtype if isinstance(value, Relaxation) else jnp.result_type(value)
