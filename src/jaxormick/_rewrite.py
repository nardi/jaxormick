"""Rewrites of jaxpr equations that relax the same values more tightly.

A product of two relaxations is relaxed as if its factors were independent. When both
factors hold the same value, as in `x * x`, the product is a power, and the rule for
`integer_pow` relaxes it exactly. JAX does not merge repeated equations, so `x[0] *
x[0]` slices `x` twice. Repeated equations are therefore merged first, which lets the
rewrites recognize equal factors.
"""

import dataclasses
from collections.abc import Hashable

import numpy as np
from jax import lax
from jax.extend.core import JaxprEqn, Literal, Var

POWER_PRIMITIVES = ("integer_pow", "square")
"""Primitives that raise their operand to a constant positive integer power."""


@dataclasses.dataclass
class EquationTable:
    """Records the equations of one jaxpr, to find repeated equations and powers."""

    canonical: dict[Var, Var] = dataclasses.field(default_factory=dict)
    """Maps a variable to the earlier variable that holds the same value."""

    outputs: dict[Hashable, list[Var]] = dataclasses.field(default_factory=dict)
    """Maps the key of an equation to the variables it binds."""

    powers: dict[Var, tuple[Var, int]] = dataclasses.field(default_factory=dict)
    """Maps a variable that holds a positive integer power to its base and exponent."""

    def canonical_atom(self, atom: Var | Literal) -> Var | Literal:
        """Returns the earliest variable with the value of `atom`, or the literal."""
        if isinstance(atom, Literal):
            return atom
        return self.canonical.get(atom, atom)

    def key(self, equation: JaxprEqn) -> Hashable | None:
        """Returns a key that equal equations share, or `None` if there is none.

        Equations with effects and equations whose parameters are not hashable have no
        key and are never merged.
        """
        if equation.effects:
            return None
        operands = tuple(
            literal_key(self.canonical_atom(atom)) for atom in equation.invars
        )
        key = (equation.primitive, operands, tuple(sorted(equation.params.items())))
        try:
            hash(key)
        except TypeError:
            return None
        return key

    def earlier_outputs(self, equation: JaxprEqn) -> list[Var] | None:
        """Returns the variables of an earlier equal equation, or `None`."""
        key = self.key(equation)
        return None if key is None else self.outputs.get(key)

    def merge(self, equation: JaxprEqn, earlier: list[Var]) -> None:
        """Records that the outputs of `equation` hold the values of `earlier`."""
        for variable, earlier_variable in zip(equation.outvars, earlier, strict=True):
            self.canonical[variable] = self.canonical.get(
                earlier_variable, earlier_variable
            )

    def power(self, atom: Var | Literal) -> tuple[Var, int] | None:
        """Returns the base and exponent of a variable, with exponent 1 by default."""
        canonical = self.canonical_atom(atom)
        if isinstance(canonical, Literal):
            return None
        return self.powers.get(canonical, (canonical, 1))

    def rewrite(self, equation: JaxprEqn) -> JaxprEqn:
        """Returns an equation with the same value whose relaxation is no looser.

        A product of two powers of the same base becomes one power, and a power of a
        power becomes one power. The univariate rule for a power is the tightest
        relaxation of it, so it is never looser than a product or a composition.
        """
        match equation.primitive.name, equation.invars:
            case "mul", [left, right]:
                match self.power(left), self.power(right):
                    case (base, left_exponent), (right_base, right_exponent) if (
                        base is right_base
                    ):
                        return power_equation(
                            equation, base, left_exponent + right_exponent
                        )
            case name, [operand] if name in POWER_PRIMITIVES:
                match self.power(operand):
                    case (base, exponent) if exponent > 1:
                        return power_equation(
                            equation, base, exponent * power_exponent(equation)
                        )
        return equation

    def record(self, equation: JaxprEqn, evaluated: JaxprEqn) -> None:
        """Records an equation and the rewritten form it was evaluated as."""
        key = self.key(equation)
        if key is not None:
            self.outputs[key] = list(equation.outvars)
        match evaluated.primitive.name, evaluated.invars, evaluated.outvars:
            case name, [operand], [output] if name in POWER_PRIMITIVES:
                exponent = power_exponent(evaluated)
                if exponent > 0 and (power := self.power(operand)) is not None:
                    base, base_exponent = power
                    self.powers[output] = (base, base_exponent * exponent)


def literal_key(atom: Var | Literal) -> Hashable:
    """Returns a hashable key for an atom, using the value of a literal."""
    if isinstance(atom, Var):
        return atom
    value = np.asarray(atom.val)
    return (value.dtype.str, value.shape, value.tobytes())


def power_exponent(equation: JaxprEqn) -> int:
    """Returns the exponent of an `integer_pow` or `square` equation."""
    return 2 if equation.primitive.name == "square" else equation.params["y"]


def power_equation(equation: JaxprEqn, base: Var, exponent: int) -> JaxprEqn:
    """Returns `equation` with its value computed as `base ** exponent`."""
    return equation.replace(
        primitive=lax.integer_pow_p, invars=[base], params={"y": exponent}
    )
