# JAXormick

McCormick relaxations of [JAX](https://docs.jax.dev/en/latest/) functions.

`jaxormick` turns a JAX function into functions that compute, over a box of inputs,
bounds on its values and convex and concave relaxations with their subgradients. Global
optimization solvers use these as lower bounds and linear cuts. The relaxations are JAX
functions, so they compile with `jax.jit`, batch over boxes with `jax.vmap` and
differentiate with `jax.grad`.

## Installation

```bash
pip install jaxormick
```

## Example

<!-- name: test_readme -->
```python
import jax.numpy as jnp

import jaxormick as jmk


def f(x):
    return x[0] * jnp.exp(x[1]) - x[1] ** 2


lower = jnp.array([0.0, -1.0])
upper = jnp.array([1.0, 1.0])
reference = jnp.array([0.5, 0.0])

relaxation = jmk.relax(f)(reference, lower, upper)
assert relaxation.lower <= relaxation.convex <= f(reference)
assert f(reference) <= relaxation.concave <= relaxation.upper

planes = jmk.affine_relaxation(f)(reference, lower, upper)
print(planes.convex, planes.convex_gradient)
```

`relaxation.lower` and `relaxation.upper` bound `f` on the whole box. `planes` holds
tangent planes of the convex and concave relaxations at the reference point, which bound
`f` from below and above on the box.

## Documentation

The [documentation](https://nardi.github.io/jaxormick/) has the theory behind McCormick
relaxations, a user guide and the API reference.

## License

[Hippocratic License 3.0](LICENSE.md).
