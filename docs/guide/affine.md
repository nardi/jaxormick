# Affine relaxations

## Tangent planes of the relaxations

`jaxormick.affine_relaxation` returns the values and gradients of both relaxations at
the reference point, as an `AffineRelaxation`. The planes

\[
f^{cv}(\bar{x}) + s^{cv} \cdot (x - \bar{x}) \quad\text{and}\quad f^{cc}(\bar{x}) + s^{cc} \cdot (x - \bar{x})
\]

bound the function from below and above on the whole box. They are the cuts a linear
programming relaxation needs, as described in the [theory
section](../theory/relaxations.md#affine-relaxations).

<!-- name: test_affine -->
```python
import jax
import jax.numpy as jnp

import jaxormick as jmk


def f(x):
    return x[0] * jnp.exp(x[1]) - x[1] ** 2


lower = jnp.array([0.0, -1.0])
upper = jnp.array([1.0, 1.0])
reference = jnp.array([0.5, 0.0])

planes = jmk.affine_relaxation(f)(reference, lower, upper)


def below(x):
    return planes.convex + planes.convex_gradient @ (x - reference)


def above(x):
    return planes.concave + planes.concave_gradient @ (x - reference)


points = jax.random.uniform(jax.random.key(0), (1000, 2), minval=lower, maxval=upper)
values = jax.vmap(f)(points)
assert jnp.all(jax.vmap(below)(points) <= values)
assert jnp.all(values <= jax.vmap(above)(points))
```

## A lower bound from one plane

The smallest value of an affine function on a box sits at a corner, chosen per
coordinate by the sign of the gradient. The value at that corner is a lower bound, often
tighter than the interval bound, at the cost of one gradient. For \(g(x) = x^2 - x\) on
\([0, 1]\), interval arithmetic computes \([0, 1] - [0, 1] = [-1, 1]\), while one plane
at the center finds the true minimum \(-0.25\).

<!-- name: test_affine -->
```python
def plane_lower_bound(planes, reference, lower, upper):
    gradient = planes.convex_gradient
    corner = jnp.where(gradient > 0, lower, upper)
    return planes.convex + gradient @ (corner - reference)


def g(x):
    return x[0] ** 2 - x[0]


box = jnp.array([0.0]), jnp.array([1.0])
center = jnp.array([0.5])
interval_bound, _ = jmk.interval(g)(*box)
plane_bound = plane_lower_bound(jmk.affine_relaxation(g)(center, *box), center, *box)
assert interval_bound == -1.0 and plane_bound == -0.25
```

The planes from several reference points, for example the box center and corners,
together form a linear program. Its optimum is a lower bound at least as tight as the
best single plane.

## One side only

When only the lower bounds matter, `jaxormick.convex_side` returns the value and
gradient of the convex relaxation. `jaxormick.concave_side` does the same for the
concave relaxation. Each computes less than `affine_relaxation`.

<!-- name: test_affine -->
```python
value, gradient = jmk.convex_side(f)(reference, lower, upper)
assert value == planes.convex
assert jnp.allclose(gradient, planes.convex_gradient)
```

## Forward and reverse mode

The gradients come from automatic differentiation. Forward mode is the default. It
pushes one tangent vector per variable through the relaxation and stores nothing for a
backward pass, which keeps the compiled program small. Reverse mode costs one backward
pass per output and is cheaper for functions with many variables, roughly from a few
dozen on.

<!-- name: test_affine -->
```python
reverse = jmk.affine_relaxation(f, mode="reverse")(reference, lower, upper)
assert jnp.allclose(reverse.convex_gradient, planes.convex_gradient)
```

Both modes return the same gradients. The [benchmarks](performance.md) show where each
is faster.

## Gradients at the box boundary

The relaxations are not differentiable everywhere, and the planes are only valid if the
gradient is a true subgradient of the convex side. Where the convex and concave sides of
an intermediate value tie, which happens often on the boundary of the box, the rules
choose one side explicitly instead of letting automatic differentiation average them.
Planes at boundary points, where linear programs put their solutions, stay valid.

<!-- name: test_affine -->
```python
corner = jmk.affine_relaxation(f)(upper, lower, upper)
corner_values = corner.convex + (points - upper) @ corner.convex_gradient
assert jnp.all(corner_values <= values)
```

The relaxed function needs a single output array for these transformations. For several
outputs, call them once per output, or use `jax.jacfwd` on the fields of `relax`
directly.
