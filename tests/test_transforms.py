"""The public transformations and how they compose with JAX."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import Primitive

import jaxormick as jmk

LOWER = jnp.array([-1.0, 0.5])
"""Lower bounds of the box used in these tests."""

UPPER = jnp.array([1.5, 2.0])
"""Upper bounds of the box used in these tests."""

REFERENCE = jnp.array([0.25, 1.0])
"""Reference point inside the box."""


def bilinear(x: jax.Array) -> jax.Array:
    """Returns a product with a known relaxation."""
    return x[0] * x[1]


def test_relax_returns_mccormick_envelopes_of_product():
    """Checks the relaxation of `x · y` against McCormick's formulas."""
    relaxation = jmk.relax(bilinear)(REFERENCE, LOWER, UPPER)
    (x, y), (xl, yl), (xu, yu) = REFERENCE, LOWER, UPPER
    convex = max(xl * y + yl * x - xl * yl, xu * y + yu * x - xu * yu)
    concave = min(xu * y + yl * x - xu * yl, xl * y + yu * x - xl * yu)
    np.testing.assert_allclose(relaxation.convex, convex, rtol=1e-6)
    np.testing.assert_allclose(relaxation.concave, concave, rtol=1e-6)
    np.testing.assert_allclose(relaxation.lower, -2.0)
    np.testing.assert_allclose(relaxation.upper, 3.0)


def test_relax_keeps_output_structure_and_constants():
    """Checks pytree outputs, constant outputs and pytree arguments."""

    def function(x: jax.Array, parameters: dict[str, jax.Array]):
        return {"product": x[0] * parameters["scale"], "constant": parameters["offset"]}

    parameters = {"scale": jnp.asarray(2.0), "offset": jnp.asarray(3.0)}
    relaxations = jmk.relax(function)(REFERENCE, LOWER, UPPER, parameters)
    assert set(relaxations) == {"product", "constant"}
    np.testing.assert_allclose(relaxations["product"].convex, 0.5)
    for field in jax.tree.leaves(relaxations["constant"]):
        np.testing.assert_allclose(field, 3.0)


def test_relax_accepts_integer_and_list_inputs():
    """Checks that inputs are converted to the default floating-point dtype."""
    relaxation = jmk.relax(bilinear)([0, 1], [-1, 0], [1, 2])
    assert relaxation.dtype == jnp.float32
    assert relaxation.shape == ()


def test_relax_composes_with_jit_vmap_and_grad():
    """Checks that relaxations can be batched, compiled and differentiated."""
    relaxed = jmk.relax(bilinear)
    references = np.stack([REFERENCE, LOWER, UPPER])
    batched = jax.jit(jax.vmap(relaxed, in_axes=(0, None, None)))(
        references, LOWER, UPPER
    )
    assert batched.convex.shape == (3,)
    gradient = jax.grad(lambda point: relaxed(point, LOWER, UPPER).convex)(REFERENCE)
    assert np.all(np.isfinite(gradient))


def test_interval_matches_relax():
    """Checks that `interval` returns the bounds of `relax`."""
    lower, upper = jmk.interval(bilinear)(LOWER, UPPER)
    relaxation = jmk.relax(bilinear)(REFERENCE, LOWER, UPPER)
    assert lower == relaxation.lower and upper == relaxation.upper


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_sides_match_affine_relaxation(mode):
    """Checks that `convex_side` and `concave_side` agree with `affine_relaxation`."""
    function = lambda x: jnp.exp(x[0]) * x[1]  # noqa: E731
    planes = jmk.affine_relaxation(function, mode)(REFERENCE, LOWER, UPPER)
    convex, convex_gradient = jmk.convex_side(function, mode)(REFERENCE, LOWER, UPPER)
    concave, concave_gradient = jmk.concave_side(function, mode)(
        REFERENCE, LOWER, UPPER
    )
    np.testing.assert_allclose(convex, planes.convex)
    np.testing.assert_allclose(convex_gradient, planes.convex_gradient)
    np.testing.assert_allclose(concave, planes.concave)
    np.testing.assert_allclose(concave_gradient, planes.concave_gradient)


def test_invalid_mode_raises():
    """Checks that an unknown differentiation mode is rejected."""
    with pytest.raises(ValueError, match="mode"):
        jmk.affine_relaxation(bilinear, cast(jmk.Mode, "sideways"))


def test_register_rule_adds_primitive():
    """Checks that a rule registered for a new primitive is used."""
    cube = Primitive("cube_for_test")
    cube.def_impl(lambda x: x**3)
    cube.def_abstract_eval(lambda x: x)

    def cube_rule(primitive, operands, params):
        (operand,) = operands
        return jmk.relax(lambda x: x**3)(operand.convex, operand.lower, operand.upper)

    assert jmk.register_rule(cube, cube_rule) is cube_rule
    assert jmk.RULES["cube_for_test"] is cube_rule
    relaxation = jmk.relax(lambda x: cube.bind(x[0]))(REFERENCE, LOWER, UPPER)
    np.testing.assert_allclose(relaxation.upper, 1.5**3)
    del jmk.RULES["cube_for_test"]


def test_relaxation_constructors():
    """Checks `Relaxation.constant`, `Relaxation.variable` and `tightened`."""
    constant = jmk.Relaxation.constant(2.0)
    assert all(float(field) == 2.0 for field in jax.tree.leaves(constant))
    variable = jmk.Relaxation.variable(0.5, 0.0, 1.0)
    assert (variable.convex, variable.concave) == (0.5, 0.5)
    loose = jmk.Relaxation(
        lower=jnp.asarray(0.0),
        upper=jnp.asarray(1.0),
        convex=jnp.asarray(-1.0),
        concave=jnp.asarray(2.0),
    )
    tight = loose.tightened()
    assert (tight.convex, tight.concave) == (0.0, 1.0)


@pytest.mark.parametrize("exponent", [0, 1])
def test_trivial_integer_powers_are_exact(exponent):
    """Checks that `x ** 0` and `x ** 1` relax to themselves."""
    function = lambda x: jax.lax.integer_pow(x[0], exponent)  # noqa: E731
    relaxation = jmk.relax(function)(REFERENCE, LOWER, UPPER)
    expected = (1.0, 1.0) if exponent == 0 else (LOWER[0], UPPER[0])
    assert (relaxation.lower, relaxation.upper) == expected
    assert relaxation.convex == relaxation.concave == function(REFERENCE)
