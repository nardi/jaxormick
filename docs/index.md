# JAXormick

McCormick relaxations of [JAX](https://docs.jax.dev/en/latest/) functions.

`jaxormick` turns a JAX function into functions that compute, over a box of inputs,
bounds on its values and convex and concave relaxations with their subgradients. These
are the lower bounds and cuts that a branch-and-bound solver needs to prove a global
optimum of a nonconvex function. The relaxations are JAX functions too, so they can be
compiled with `jax.jit`, batched over many boxes with `jax.vmap` and differentiated with
`jax.grad`.

The transformations are:

- [`relax`][jaxormick.relax], which returns interval bounds and the values of the convex
  and concave relaxations at a reference point,
- [`interval`][jaxormick.interval], which returns the interval bounds only,
- [`affine_relaxation`][jaxormick.affine_relaxation], which returns the tangent planes
  of both relaxations at a reference point, and
- [`convex_side`][jaxormick.convex_side] and [`concave_side`][jaxormick.concave_side],
  which return one relaxation and its gradient.

`intersect_relaxations` combines relaxations of several forms of a function. With
`alpha_bb=True`, the McCormick relaxation is intersected with an αBB relaxation. With
`tighten_bounds=True`, the intervals of intermediate values are narrowed with
subgradients. See [tighter relaxations](guide/tightening.md).

The solutions of equations that `jnp.linalg.solve`, `lax.custom_linear_solve` and
`lax.custom_root` compute are relaxed from the residual of the equations, not from the
steps of the solver. See [implicit functions](theory/implicit.md).

## Installation

```bash
pip install jaxormick
```

JAXormick needs Python 3.11 or newer and works with JAX 0.10 and 0.11. Before JAX 0.11,
`lax.custom_root` is not supported. See [supported operations](guide/supported.md).

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

The interval bounds show that the function is in \([-1, 2.718]\) on the whole box. At
the reference point, the convex relaxation is \(-0.816\) and the concave one \(1.359\),
around the function value \(0.5\). The plane through \(-0.816\) at the reference point
with slope `planes.convex_gradient` is below \(f\) everywhere on the box.

## Further reading

- [Convex relaxations](theory/relaxations.md) explains what relaxations are and how
  global optimization uses them.
- [McCormick relaxations](theory/mccormick.md) shows how they are built, with a worked
  example.
- [Relaxations in JAX](theory/jax.md) explains how `jaxormick` relaxes the jaxpr of a
  function and computes subgradients by automatic differentiation.
- The [user guide](guide/basic.md) describes the transformations, control flow and the
  [supported operations](guide/supported.md).

`jaxormick` computes the bounds in floating point without outward rounding, so they are
valid up to rounding errors. See the [known limits](guide/supported.md#known-limits).
