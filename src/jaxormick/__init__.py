"""McCormick relaxations of JAX functions."""

from ._errors import UnsupportedPrimitiveError as UnsupportedPrimitiveError
from ._relaxation import Relaxation as Relaxation
from ._relaxation import intersect_relaxations as intersect_relaxations
from ._rules import RULES as RULES
from ._rules import Operand as Operand
from ._rules import Params as Params
from ._rules import RelaxationRule as RelaxationRule
from ._rules import as_relaxation as as_relaxation
from ._rules import compose as compose
from ._rules import compose_concave as compose_concave
from ._rules import compose_convex as compose_convex
from ._rules import mid as mid
from ._rules import register_rule as register_rule
from ._rules import secant as secant
from ._transforms import AffineRelaxation as AffineRelaxation
from ._transforms import Mode as Mode
from ._transforms import affine_relaxation as affine_relaxation
from ._transforms import concave_side as concave_side
from ._transforms import convex_side as convex_side
from ._transforms import interval as interval
from ._transforms import relax as relax
