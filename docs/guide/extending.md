# Custom rules

## Rules and primitives

JAX functions break down into primitives such as `exp`, `mul` or `reduce_sum`. A
relaxation rule computes the `Relaxation` of one primitive's output from the relaxations
of its inputs. To find the primitive behind a function, print its jaxpr.

<!-- name: test_extending -->
```python
import jax
import jax.numpy as jnp

print(jax.make_jaxpr(jnp.cosh)(1.0))
```

This prints `{ lambda ; a:f32[]. let b:f32[] = cosh a in (b,) }`, so `jnp.cosh` is the
primitive `cosh`, which has no rule yet.

## A rule for a convex function

A rule is a function of the primitive, its operands and its parameters, as described by
`jaxormick.RelaxationRule`. Operands that depend on the variables arrive as `Relaxation`
objects and the others as arrays. The interpreter calls the rule only when at least one
operand is a `Relaxation`.

`cosh` is convex with its minimum at zero. `jaxormick.compose_convex` relaxes a convex
univariate function, given the point of the interval where the function is smallest.

<!-- name: test_extending -->
```python
import jaxormick as jmk


def cosh_rule(primitive, operands, params):
    operand = jmk.as_relaxation(operands[0])
    smallest = jnp.clip(0.0, operand.lower, operand.upper)
    return jmk.compose_convex(operand, jnp.cosh, smallest)


jmk.register_rule("cosh", cosh_rule)

relaxation = jmk.relax(lambda x: jnp.cosh(x) - x)(
    jnp.array(0.5), jnp.array(-1.0), jnp.array(2.0)
)
assert relaxation.convex <= jnp.cosh(0.5) - 0.5 <= relaxation.concave
```

`jaxormick.compose_concave` does the same for concave functions, given the point where
the function is largest. For other shapes, `jaxormick.compose` takes a convex
underestimator and a concave overestimator on the interval, each as a function of one
array, with their minimizer and maximizer and the interval bounds of the result.

## Rules from the four fields

A rule can also build the `Relaxation` directly. Its fields are the interval bounds and
the values of the relaxations at the reference point, all as arrays of the output's
shape. A rule for `atan`, which is increasing, concave right of zero and convex left of
it, could start with the interval bounds alone, using the bounds as constant
relaxations:

<!-- name: test_extending -->
```python
def atan_bounds_rule(primitive, operands, params):
    operand = jmk.as_relaxation(operands[0])
    lower, upper = jnp.arctan(operand.lower), jnp.arctan(operand.upper)
    return jmk.Relaxation(lower=lower, upper=upper, convex=lower, concave=upper)


jmk.register_rule("atan", atan_bounds_rule)
low, high = jmk.interval(jnp.arctan)(jnp.array(-1.0), jnp.array(1.0))
assert jnp.isclose(low, -jnp.pi / 4) and jnp.isclose(high, jnp.pi / 4)
```

Constant relaxations are valid but loose, which makes them a safe first version of a
rule. A later version can tighten the convex and concave sides.

## Tests for a rule

A rule must satisfy `lower ≤ convex ≤ f(x) ≤ concave ≤ upper` at every reference point
of every box, with `convex` convex and `concave` concave in the reference point. The
helper below samples points in a box and checks these inequalities.

<!-- name: test_extending -->
```python
def check(function, lower, upper, count=1000):
    points = jnp.linspace(lower, upper, count)
    relaxations = jax.vmap(jmk.relax(function), in_axes=(0, None, None))(
        points, lower, upper
    )
    values = jax.vmap(function)(points)
    assert jnp.all(relaxations.lower <= relaxations.convex)
    assert jnp.all(relaxations.convex <= values + 1e-6)
    assert jnp.all(values <= relaxations.concave + 1e-6)
    assert jnp.all(relaxations.concave <= relaxations.upper)


check(jnp.cosh, jnp.array(-1.0), jnp.array(2.0))
check(jnp.arctan, jnp.array(-3.0), jnp.array(1.0))
```

Registering a rule for a primitive that already has one replaces it. The dictionary
`jaxormick.RULES` holds the rules, keyed by primitive name.

## Custom primitives

For a primitive created with `jax.extend.core.Primitive`, pass the primitive itself to
`register_rule`. Its rule can often use the monotonicity and convexity of the function
it computes, as the `cosh` rule above does.
