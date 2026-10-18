from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias

from jax.extend.core import Primitive
from jaxtyping import Array

from .._relaxation import Relaxation

Operand: TypeAlias = Relaxation | Array
"""A primitive operand, relaxed when it depends on the variables and plain otherwise."""

Params: TypeAlias = Mapping[str, Any]
"""The parameters of a primitive.

Their types depend on the primitive. They hold shapes, dtypes, axes, jaxprs and more, so
no type narrower than `Any` covers every value.
"""


class RelaxationRule(Protocol):
    """A function that relaxes one primitive.

    The interpreter calls a rule only when at least one operand is a `Relaxation`. A
    rule returns one operand, or a sequence of them for a primitive with multiple
    results.
    """

    def __call__(
        self, primitive: Primitive, operands: Sequence[Operand], params: Params, /
    ) -> Operand | Sequence[Operand]: ...


RULES: dict[str, RelaxationRule] = {}
"""Relaxation rules keyed by primitive name."""


def register_rule(primitive: Primitive | str, rule: RelaxationRule) -> RelaxationRule:
    """Registers `rule` as the relaxation rule of `primitive` and returns the rule.

    A rule that is already registered for the primitive is replaced.
    """
    RULES[primitive if isinstance(primitive, str) else primitive.name] = rule
    return rule


def relaxes(*names: str) -> Callable[[RelaxationRule], RelaxationRule]:
    """Returns a decorator that registers a rule for each of the named primitives."""

    def register(rule: RelaxationRule) -> RelaxationRule:
        for name in names:
            register_rule(name, rule)
        return rule

    return register
