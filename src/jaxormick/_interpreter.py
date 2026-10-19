"""A jaxpr interpreter that evaluates relaxed operands with the registered rules."""

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, TypeGuard, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.core import AbstractValue, Tracer
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var
from jaxtyping import Array

from ._errors import UnsupportedPrimitiveError
from ._relaxation import Relaxation
from ._rewrite import EquationTable
from ._rules import RULES, Operand, as_relaxation, is_relaxed
from ._tightening import Box, Tracked, tightened_bounds

CALL_JAXPR_PARAMS = {
    "jit": "jaxpr",
    "pjit": "jaxpr",
    "closed_call": "call_jaxpr",
    "core_call": "call_jaxpr",
    "custom_jvp_call": "call_jaxpr",
    "custom_vjp_call": "call_jaxpr",
    "remat": "jaxpr",
    "remat2": "jaxpr",
    "checkpoint": "jaxpr",
}
"""Call primitives, mapped to the parameter that holds the jaxpr they call.

Custom derivative rules are ignored, because only the primal computation is relaxed.
"""

IMPLICIT_PRIMITIVES = ("custom_linear_solve", "while")
"""Primitives whose result is defined by a solve or a loop of unknown length."""


TrackedOperand: TypeAlias = Operand | Tracked
"""An operand of a jaxpr whose intermediate bounds are tightened."""

Value = TypeVar("Value", Operand, TrackedOperand)
"""The type of the operands of an evaluated jaxpr."""


def eval_jaxpr(
    jaxpr: Jaxpr | ClosedJaxpr, arguments: Sequence[Operand]
) -> list[Operand]:
    """Evaluates a jaxpr on a mix of relaxed and plain operands and returns its outputs."""
    return run_jaxpr(jaxpr, arguments, eval_equation)


def eval_tracked_jaxpr(
    jaxpr: Jaxpr | ClosedJaxpr, arguments: Sequence[TrackedOperand], box: Box
) -> list[TrackedOperand]:
    """Evaluates a jaxpr on tracked and plain operands, tightening every bound."""
    return run_jaxpr(
        jaxpr,
        arguments,
        lambda equation, inputs: eval_tracked_equation(equation, inputs, box),
    )


def run_jaxpr(
    jaxpr: Jaxpr | ClosedJaxpr,
    arguments: Sequence[Value],
    evaluate: Callable[[JaxprEqn, list[Value]], list[Value]],
) -> list[Value]:
    """Evaluates a jaxpr with `evaluate` for each equation and returns its outputs.

    Equations whose inputs depend on the variables are rewritten first, and equal
    equations run only once.
    """
    consts: Sequence[Operand] = ()
    if isinstance(jaxpr, ClosedJaxpr):
        # From JAX 0.11 on, every jaxpr holds its constants and is its own `jaxpr`.
        jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
    environment: dict[Var, Value] = {}

    def read(atom: Var | Literal) -> Value:
        """Returns the value bound to a jaxpr atom."""
        return atom.val if isinstance(atom, Literal) else environment[atom]

    environment.update(zip(jaxpr.constvars, consts, strict=True))
    environment.update(zip(jaxpr.invars, arguments, strict=True))
    table = EquationTable()
    for equation in jaxpr.eqns:
        earlier = table.earlier_outputs(equation)
        if earlier is not None:
            # An equal equation already ran, so its outputs are reused.
            table.merge(equation, earlier)
            outputs = [environment[variable] for variable in earlier]
        else:
            inputs = [read(atom) for atom in equation.invars]
            evaluated = equation
            if any(depends_on_variables(value) for value in inputs):
                evaluated = table.rewrite(equation)
                inputs = [read(atom) for atom in evaluated.invars]
            outputs = evaluate(evaluated, inputs)
            table.record(equation, evaluated)
        environment.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


def depends_on_variables(value: TrackedOperand) -> bool:
    """Returns whether a value is relaxed, with or without tracking."""
    return isinstance(value, Tracked) or is_relaxed(value)


def eval_tracked_equation(
    equation: JaxprEqn, inputs: list[TrackedOperand], box: Box
) -> list[TrackedOperand]:
    """Evaluates one equation at the reference point and at the middle of the box.

    The subgradients at the middle are pushed forward through the equation, and every
    relaxed output gets the bounds that they tighten. Call primitives are evaluated
    equation by equation, so their intermediate values are tightened too. The bodies
    of `scan` and `cond` are not, but their outputs are.
    """
    name = equation.primitive.name
    plain_inputs = [value for value in inputs if not isinstance(value, Tracked)]
    if len(plain_inputs) == len(inputs):
        plain_outputs: list[TrackedOperand] = [*bind_plain(equation, plain_inputs)]
        return plain_outputs
    if name in CALL_JAXPR_PARAMS:
        return eval_tracked_jaxpr(equation.params[CALL_JAXPR_PARAMS[name]], inputs, box)
    at_reference = eval_equation(
        equation,
        [
            value.at_reference if isinstance(value, Tracked) else value
            for value in inputs
        ],
    )
    # An input that appears twice stays one object, as rules such as `ne` expect.
    tracked = list(
        {id(value): value for value in inputs if isinstance(value, Tracked)}.values()
    )

    def at_middle(sides: list[tuple[Array, Array]]) -> list[Operand]:
        """Evaluates the equation at the middle, with the given sides of the inputs."""
        replaced = {
            id(value): eqx.tree_at(
                lambda relaxation: (relaxation.convex, relaxation.concave),
                value.at_middle,
                value_sides,
            )
            for value, value_sides in zip(tracked, sides, strict=True)
        }
        middle_inputs = [
            replaced[id(value)] if isinstance(value, Tracked) else value
            for value in inputs
        ]
        return eval_equation(equation, middle_inputs)

    def differentiable_sides(
        sides: list[tuple[Array, Array]],
    ) -> tuple[list[tuple[Array, Array]], list[Operand]]:
        """Returns the sides of the floating-point relaxed outputs, and all outputs."""
        outputs = at_middle(sides)
        return [
            (output.convex, output.concave)
            for output in outputs
            if is_differentiable(output)
        ], outputs

    sides = [(value.at_middle.convex, value.at_middle.concave) for value in tracked]
    # Without subgradients of every input, those of the outputs are unknown.
    is_known = all(is_differentiable(value.at_middle) for value in tracked)
    if is_known:
        _, linearized, middle_outputs = jax.linearize(
            differentiable_sides, sides, has_aux=True
        )
        known_tangents = iter(
            jax.vmap(linearized)(
                [(value.convex_tangents, value.concave_tangents) for value in tracked]
            )
        )
    else:
        middle_outputs, known_tangents = at_middle(sides), iter(())
    outputs: list[TrackedOperand] = []
    for reference_output, middle_output in zip(
        at_reference, middle_outputs, strict=True
    ):
        match middle_output:
            case Relaxation() if is_known and is_differentiable(middle_output):
                convex_tangents, concave_tangents = next(known_tangents)
                lower, upper = tightened_bounds(
                    middle_output, convex_tangents, concave_tangents, box
                )
            case Relaxation():
                convex_tangents = concave_tangents = unknown_tangents(
                    middle_output, box
                )
                lower, upper = middle_output.lower, middle_output.upper
            case _:
                outputs.append(reference_output)
                continue
        outputs.append(
            Tracked(
                at_reference=with_bounds(as_relaxation(reference_output), lower, upper),
                at_middle=with_bounds(middle_output, lower, upper),
                convex_tangents=convex_tangents,
                concave_tangents=concave_tangents,
            )
        )
    return outputs


def unknown_tangents(relaxation: Relaxation, box: Box) -> Array:
    """Returns tangents of NaN, which mark the subgradients of a value as unknown."""
    dtype = relaxation.dtype
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = box.middle.dtype
    return jnp.full((box.middle.size, *relaxation.shape), jnp.nan, dtype)


def with_bounds(relaxation: Relaxation, lower: Array, upper: Array) -> Relaxation:
    """Returns the relaxation with its interval replaced."""
    return Relaxation(
        lower=lower,
        upper=upper,
        convex=relaxation.convex,
        concave=relaxation.concave,
    )


def is_differentiable(value: Operand) -> TypeGuard[Relaxation]:
    """Returns whether a value is a relaxation of floating-point dtype."""
    return isinstance(value, Relaxation) and jnp.issubdtype(value.dtype, jnp.inexact)


def eval_equation(equation: JaxprEqn, inputs: list[Operand]) -> list[Operand]:
    """Evaluates one equation and returns its outputs as a list."""
    primitive = equation.primitive
    if not any(is_relaxed(value) for value in inputs):
        return bind_plain(equation, inputs)

    name = primitive.name
    if is_implicit(equation):
        raise UnsupportedPrimitiveError(
            f"`{name}` defines its result implicitly, by a solve or a loop of unknown "
            "length, and implicit functions are not supported."
        )
    match name:
        case "scan":
            outputs = eval_scan(equation, inputs)
        case "cond":
            outputs = eval_cond(equation, inputs)
        case _ if name in CALL_JAXPR_PARAMS:
            outputs = eval_jaxpr(equation.params[CALL_JAXPR_PARAMS[name]], inputs)
        case _ if name in RULES:
            output = RULES[name](primitive, inputs, equation.params)
            outputs = list(output) if isinstance(output, Sequence) else [output]
        case _:
            raise UnsupportedPrimitiveError(
                f"There is no relaxation rule for `{name}` with an operand that "
                "depends on the variables."
            )
    return [
        conform(output, variable.aval)
        for output, variable in zip(outputs, equation.outvars, strict=True)
    ]


def is_implicit(equation: JaxprEqn) -> bool:
    """Returns whether an equation is an implicit function such as `lax.custom_root`.

    `lax.custom_root` produces a `custom_jvp_call` whose primal jaxpr only runs the
    solver. Relaxing that jaxpr would bound the solver's iterates instead of the root,
    so it is recognized by the function that defines its derivative.
    """
    if equation.primitive.name in IMPLICIT_PRIMITIVES:
        return True
    derivative_rule = equation.params.get("jvp_jaxpr_fun")
    debug_info = getattr(derivative_rule, "debug_info", None)
    source = getattr(debug_info, "func_src_info", None) or ""
    return source.startswith("_root_jvp ")


def bind_plain(equation: JaxprEqn, inputs: list[Operand]) -> list[Operand]:
    """Binds an equation's primitive to plain inputs, as `jax.core.eval_jaxpr` does.

    Inputs that are concrete are folded while tracing, so rules such as `pow` see the
    constant exponents that JAX computes, as in the derivative of `x ** 1.5`.
    """
    primitive = equation.primitive
    parameters = primitive.get_bind_params(equation.params)
    if any(isinstance(value, Tracer) for value in inputs):
        outputs = primitive.bind(*inputs, **parameters)
    else:
        with jax.ensure_compile_time_eval():
            outputs = primitive.bind(*inputs, **parameters)
    return list(outputs) if primitive.multiple_results else [outputs]


def conform(value: Operand, aval: AbstractValue) -> Operand:
    """Casts the fields of a relaxed value to the dtype that the jaxpr expects.

    Rules may return a different dtype or weak type than the primitive, for example
    after mixing in Python scalars. `lax.scan` and `lax.switch` require the exact types,
    so relaxed values are cast. Plain values already have them.
    """
    if not is_relaxed(value):
        return value
    dtype = getattr(aval, "dtype")
    return jax.tree.map(lambda field: lax.convert_element_type(field, dtype), value)


def promote(value: Operand, is_promoted: bool, aval: AbstractValue) -> Operand:
    """Returns the value as a conformed `Relaxation` if `is_promoted`, else unchanged."""
    return conform(as_relaxation(value), aval) if is_promoted else value


def relaxed_outputs(
    function: Callable[..., Sequence[Operand]], *arguments: object
) -> list[bool]:
    """Returns which outputs of `function` are relaxed, without running it."""
    shapes = jax.eval_shape(function, *arguments)
    return [isinstance(shape, Relaxation) for shape in shapes]


def eval_scan(equation: JaxprEqn, inputs: list[Operand]) -> list[Operand]:
    """Relaxes a `scan` as another `scan` whose carry and outputs may be relaxed.

    A carry that starts plain can become relaxed in the body, for example the iterate
    of a fixed-point iteration that starts at a constant. Such carries are promoted to
    a `Relaxation` before the loop starts. Which ones they are follows from repeated
    abstract evaluation of the body until no further carry becomes relaxed. The body is
    traced once, whatever the number of iterations.
    """
    params = equation.params
    body: ClosedJaxpr = params["jaxpr"]
    consts_count, carry_count = scan_counts(params)
    consts = inputs[:consts_count]
    initial = inputs[consts_count : consts_count + carry_count]
    sequences = inputs[consts_count + carry_count :]
    carry_avals = [variable.aval for variable in equation.outvars[:carry_count]]

    def step(
        carry: list[Operand], slices: list[Operand]
    ) -> tuple[list[Operand], list[Operand]]:
        """Runs the body once and returns the new carry and the stacked outputs."""
        outputs = eval_jaxpr(body, [*consts, *carry, *slices])
        return outputs[:carry_count], outputs[carry_count:]

    slice_shapes = [
        jax.tree.map(
            lambda field: jax.ShapeDtypeStruct(field.shape[1:], field.dtype), x
        )
        for x in sequences
    ]
    is_promoted = [is_relaxed(value) for value in initial]
    while True:
        carry = [
            promote(value, promoted, aval)
            for value, promoted, aval in zip(initial, is_promoted, carry_avals)
        ]
        updated = relaxed_outputs(lambda c, s: step(c, s)[0], carry, slice_shapes)
        widened = [new or old for new, old in zip(updated, is_promoted)]
        if widened == is_promoted:
            break
        is_promoted = widened

    def promoted_step(
        carry: list[Operand], slices: list[Operand]
    ) -> tuple[list[Operand], list[Operand]]:
        """Runs the body and promotes the new carry like the initial one."""
        new_carry, outputs = step(carry, slices)
        return [
            promote(value, promoted, aval)
            for value, promoted, aval in zip(new_carry, is_promoted, carry_avals)
        ], outputs

    final, stacked = lax.scan(
        promoted_step,
        carry,
        sequences,
        length=params["length"],
        reverse=params["reverse"],
        unroll=params["unroll"],
    )
    return [*final, *stacked]


def scan_counts(params: dict[str, Any]) -> tuple[int, int]:
    """Returns the numbers of constants and carries among the operands of a `scan`.

    From JAX 0.11 on, `scan` describes its three groups of operands by flat trees
    instead of counts.
    """
    if "num_consts" in params:
        return params["num_consts"], params["num_carry"]
    consts, carry, _ = params["ft_in"].unpack()
    return len(list(consts.vals)), len(list(carry.vals))


def eval_cond(equation: JaxprEqn, inputs: list[Operand]) -> list[Operand]:
    """Relaxes a `cond` whose branch index does not depend on the variables.

    A concrete index selects its branch while tracing. Otherwise the relaxed branches
    run under `lax.switch`, with every output as a `Relaxation` so that all branches
    return the same types.
    """
    index, *operands = inputs
    if isinstance(index, Relaxation):
        raise UnsupportedPrimitiveError(
            "`cond` with a branch index that depends on the variables is not supported."
        )
    branches: Sequence[ClosedJaxpr] = equation.params["branches"]
    if not isinstance(index, Tracer):
        return eval_jaxpr(branches[int(index)], operands)
    avals = [variable.aval for variable in equation.outvars]

    def relaxed_branch(branch: ClosedJaxpr) -> Callable[..., list[Operand]]:
        """Returns a function that evaluates a branch with relaxed outputs."""

        def evaluate(*branch_operands: Operand) -> list[Operand]:
            outputs = eval_jaxpr(branch, branch_operands)
            return [promote(output, True, aval) for output, aval in zip(outputs, avals)]

        return evaluate

    return lax.switch(index, [relaxed_branch(branch) for branch in branches], *operands)
