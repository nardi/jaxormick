# Control flow

## Function calls

`jaxormick` relaxes nested `jax.jit` functions, `jax.checkpoint`, `jax.custom_jvp` and
`jax.custom_vjp` by relaxing the function they wrap. It ignores custom derivative rules,
because the relaxations take only their values from the function. Their gradients come
from differentiating the relaxation itself.

<!-- name: test_control_flow -->
```python
import jax
import jax.numpy as jnp
from jax import lax

import jaxormick as jmk


@jax.jit
def inner(x):
    return jax.nn.softplus(x)


relaxation = jmk.relax(lambda x: inner(x) - x)(
    jnp.array(0.0), jnp.array(-1.0), jnp.array(1.0)
)
assert relaxation.convex <= jax.nn.softplus(0.0) <= relaxation.concave
```

## Loops with a fixed length

`jaxormick` relaxes `lax.scan` and `lax.fori_loop` with constant bounds as loops. It
relaxes the body once and runs it inside a `lax.scan`, so compile time does not depend
on the number of iterations. A carry that starts as a constant and later depends on the
variables, like the iterate below, becomes a `Relaxation` from the first step on.

<!-- name: test_control_flow -->
```python
def square_root(x):
    target = x[0]
    return lax.fori_loop(0, 5, lambda step, guess: (guess + target / guess) / 2, 1.0)


relaxation = jmk.relax(square_root)(
    jnp.array([2.0]), jnp.array([1.0]), jnp.array([4.0])
)
assert relaxation.convex <= jnp.sqrt(2.0) <= relaxation.concave
print(relaxation.lower, relaxation.upper)
```

The bounds are valid but loose, here about `[0.26, 8.6]` for a square root on `[1, 4]`.
Each step relaxes the previous iterate, and the errors add up. Iterative algorithms
often relax better as the closed form of their result, if one exists, or with fewer
iterations from a better starting point.

`lax.while_loop` raises `UnsupportedPrimitiveError` when its body depends on the
variables, since the number of steps is not known while tracing. A `fori_loop` with
traced bounds turns into a `while_loop` and has the same limit.

## Conditionals

`lax.cond` and `lax.switch` are supported when the branch index does not depend on the
variables. With a Python constant as index, tracing picks the branch. With a traced
constant argument as index, the relaxed branches run under `lax.switch`.

<!-- name: test_control_flow -->
```python
def piecewise(x, use_square):
    return lax.cond(use_square, lambda y: y**2, jnp.abs, x[0])


relaxed = jmk.relax(piecewise)
square = relaxed(jnp.array([0.5]), jnp.array([-1.0]), jnp.array([1.0]), True)
absolute = relaxed(jnp.array([0.5]), jnp.array([-1.0]), jnp.array([1.0]), False)
assert square.convex == 0.25 and absolute.convex == 0.5
```

A condition that depends on the variables would make the function jump, and McCormick's
rules need continuous functions. Such branches raise `UnsupportedPrimitiveError`.
`jnp.where` with a condition on the variables raises in the same way. Many such
conditions are a maximum or a minimum in disguise and can be rewritten with
`jnp.maximum`, `jnp.minimum`, `jnp.abs` or `jax.nn.relu`.

<!-- name: test_control_flow -->
```python
def ramp(x):
    return jnp.maximum(x[0], 0.0)


relaxation = jmk.relax(ramp)(jnp.array([0.0]), jnp.array([-1.0]), jnp.array([1.0]))
assert relaxation.convex == 0.0 and relaxation.concave == 0.5
```
