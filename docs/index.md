# JAXormick

McCormick relaxations of [JAX](https://docs.jax.dev/en/latest/) functions.

`jaxormick` turns a JAX function into functions that compute, over a box of inputs,
bounds on its values and convex and concave relaxations with their subgradients. These
are the lower bounds and cuts that a branch-and-bound solver needs to prove a global
optimum of a nonconvex function. The relaxations are JAX functions too, so they compile
with `jax.jit`, batch over many boxes with `jax.vmap` and differentiate with `jax.grad`.

The transformations are:

- [`relax`][jaxormick.relax], which returns interval bounds and the values of the convex
  and concave relaxations at a reference point,
- [`interval`][jaxormick.interval], which returns the interval bounds only,
- [`affine_relaxation`][jaxormick.affine_relaxation], which returns the tangent planes
  of both relaxations at a reference point, and
- [`convex_side`][jaxormick.convex_side] and [`concave_side`][jaxormick.concave_side],
  which return one relaxation and its gradient.

`intersect_relaxations` combines relaxations of several forms of a function.
`alpha_bb=True` intersects the McCormick relaxation with an αBB relaxation, and
`tighten_bounds=True` narrows intermediate intervals with subgradients. See [tighter
relaxations](guide/tightening.md).

## Installation

```bash
pip install jaxormick
```

## Quick example

The example bounds \(f(x) = x_1 e^{x_2} - x_2^2\) on the box \([0, 1] \times [-1, 1]\).

<!-- name: test_index -->
```python
import jax.numpy as jnp

import jaxormick as jmk


def f(x):
    return x[0] * jnp.exp(x[1]) - x[1] ** 2


lower = jnp.array([0.0, -1.0])
upper = jnp.array([1.0, 1.0])
reference = jnp.array([0.5, 0.0])

relaxation = jmk.relax(f)(reference, lower, upper)
print(relaxation.lower, relaxation.upper)
print(relaxation.convex, relaxation.concave)

planes = jmk.affine_relaxation(f)(reference, lower, upper)
print(planes.convex_gradient)
```

The function lies in \([-1, 2.718]\) on the whole box. At the reference point, the
convex relaxation is \(-0.816\) and the concave one \(1.359\), around the function value
\(0.5\). The plane through \(-0.816\) with slope `planes.convex_gradient` stays below
\(f\) everywhere on the box.

## Further reading

- [Convex relaxations](theory/relaxations.md) explains what relaxations are and how
  global optimization uses them.
- [McCormick relaxations](theory/mccormick.md) shows how they are built, with a worked
  example.
- [Relaxations in JAX](theory/jax.md) explains why JAX suits them.
- The [user guide](guide/basic.md) covers the transformations, control flow and the
  [supported operations](guide/supported.md).

`jaxormick` computes the bounds in floating point without outward rounding, so they are
valid up to rounding errors. See the [known limits](guide/supported.md#known-limits).
