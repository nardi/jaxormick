# Parameters and batches

## Constant arguments

The relaxed function passes arguments after the variables to the function unchanged.
They count as constants, so `jaxormick` relaxes over the variables only. Because they
are runtime arguments to a compiled function, changing their values does not cause a
retrace or a recompile.

<!-- name: test_parameters -->
```python
import jax
import jax.numpy as jnp

import jaxormick as jmk


def model(x, theta):
    return theta[0] * jnp.exp(-theta[1] * x[0]) + x[1] ** 2


lower = jnp.array([0.0, -1.0])
upper = jnp.array([1.0, 1.0])
center = (lower + upper) / 2

relaxed_model = jmk.relax(model)
first = relaxed_model(center, lower, upper, jnp.array([2.0, 0.5]))
second = relaxed_model(center, lower, upper, jnp.array([3.0, 2.0]))
```

Constant arguments can be any pytree of arrays, such as a dictionary of weights. Close
over Python scalars and other static values with a `lambda` or `functools.partial`
instead.

## Many parameter values

`jax.vmap` maps the relaxation over a batch of parameters. Here the box stays fixed.

<!-- name: test_parameters -->
```python
thetas = jnp.array([[1.0, 0.5], [2.0, 1.0], [3.0, 2.0]])
relaxations = jax.vmap(relaxed_model, in_axes=(None, None, None, 0))(
    center, lower, upper, thetas
)
assert relaxations.lower.shape == (3,)
```

## Many boxes

Branch and bound keeps a list of open boxes. Evaluating them one at a time spends most
of the time in Python and dispatch. Mapping over a batch of boxes runs them in one
compiled call.

<!-- name: test_parameters -->
```python
lowers = jnp.stack([jnp.linspace(0.0, 0.75, 4), jnp.full(4, -1.0)], axis=1)
uppers = lowers + jnp.array([0.25, 2.0])
theta = jnp.array([2.0, 0.5])

batched_interval = jax.jit(jax.vmap(jmk.interval(model), in_axes=(0, 0, None)))
lows, highs = batched_interval(lowers, uppers, theta)
assert lows.shape == highs.shape == (4,)
assert jnp.all(lows <= highs)
```

Keep the batch size fixed, for example by padding with copies of a box, so the compiled
program is reused across iterations.

## Gradients with respect to parameters

The relaxed functions are differentiable in every argument, including the constant ones.
The gradient of a bound with respect to the parameters is its sensitivity, which bilevel
problems and fits under worst-case bounds need.

<!-- name: test_parameters -->
```python
def upper_bound(theta):
    return relaxed_model(center, lower, upper, theta).upper


sensitivity = jax.grad(upper_bound)(theta)
assert sensitivity.shape == (2,)
```
