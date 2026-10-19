# Custom rules

## Rules and primitives

JAX functions break down into primitives such as `exp`, `mul` or `reduce_sum`. A
relaxation rule computes the `Relaxation` of one primitive's output from the relaxations
of its inputs. To find the primitive behind a function, print its jaxpr.

<!-- name: test_extending -->
```python
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln

print(jax.make_jaxpr(gammaln)(1.0))
```

This prints `{ lambda ; a:f32[]. let b:f32[] = lgamma a in (b,) }`, so
`jax.scipy.special.gammaln` is the primitive `lgamma`, which has no rule yet.

## A rule for a convex function

A rule is a function of the primitive, its operands and its parameters, as described by
`jaxormick.RelaxationRule`. Operands that depend on the variables arrive as `Relaxation`
objects and the others as arrays. The interpreter calls the rule only when at least one
operand is a `Relaxation`.

`lgamma` is convex on positive arguments, with its minimum near 1.4616.
`jaxormick.compose_convex` relaxes a convex univariate function, given the point of the
interval where the function is smallest. The rule below assumes a positive interval.

<!-- name: test_extending -->
```python
import jaxormick as jmk


def lgamma_rule(primitive, operands, params):
    operand = jmk.as_relaxation(operands[0])
    smallest = jnp.clip(1.4616321, operand.lower, operand.upper)
    return jmk.compose_convex(operand, gammaln, smallest)


jmk.register_rule("lgamma", lgamma_rule)

relaxation = jmk.relax(lambda x: gammaln(x) - x)(
    jnp.array(2.5), jnp.array(0.5), jnp.array(3.0)
)
assert relaxation.convex <= gammaln(2.5) - 2.5 <= relaxation.concave
```

`jaxormick.compose_concave` does the same for concave functions, given the point where
the function is largest. For other shapes, `jaxormick.compose` takes a convex
underestimator and a concave overestimator on the interval, each as a function of one
array, with their minimizer and maximizer and the interval bounds of the result.

## Rules from the four fields

A rule can also build the `Relaxation` directly. Its fields are the interval bounds and
the values of the relaxations at the reference point, all as arrays of the output's
shape. A rule for `digamma`, which is increasing and concave on positive arguments,
could start with the interval bounds alone, using the bounds as constant relaxations:

<!-- name: test_extending -->
```python
def digamma_bounds_rule(primitive, operands, params):
    operand = jmk.as_relaxation(operands[0])
    lower, upper = digamma(operand.lower), digamma(operand.upper)
    return jmk.Relaxation(lower=lower, upper=upper, convex=lower, concave=upper)


jmk.register_rule("digamma", digamma_bounds_rule)
low, high = jmk.interval(digamma)(jnp.array(1.0), jnp.array(2.0))
assert jnp.isclose(high - low, 1.0)
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


check(gammaln, jnp.array(0.5), jnp.array(3.0))
check(digamma, jnp.array(1.0), jnp.array(3.0))
```

Registering a rule for a primitive that already has one replaces it. The dictionary
`jaxormick.RULES` holds the rules, keyed by primitive name.

## Custom primitives

For a primitive created with `jax.extend.core.Primitive`, pass the primitive itself to
`register_rule`. Its rule can often use the monotonicity and convexity of the function
it computes, as the `lgamma` rule above does.
