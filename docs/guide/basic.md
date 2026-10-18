# Basic usage

## The `relax` transformation

`jaxormick.relax` takes a function of an array of variables and returns a function of a
reference point and a box. It returns a `Relaxation` with four arrays of the shape of
the output:

- `lower` and `upper` bound the function on the whole box,
- `convex` is the value of a convex underestimator at the reference point,
- `concave` is the value of a concave overestimator at the reference point.

<!-- name: test_basic -->
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
print(relaxation.convex, relaxation.concave)
```

This prints `-0.8160603 1.3591409`, the values of the [worked
example](../theory/mccormick.md#a-worked-example).

The bounds hold for every point of the box. The convex and concave values describe the
relaxations at the reference point only. To see the relaxations as functions, evaluate
them at several reference points in the same box.

<!-- name: test_basic -->
```python
import jax

points = jnp.stack([jnp.linspace(0, 1, 5), jnp.zeros(5)], axis=1)
relaxations = jax.vmap(jmk.relax(f), in_axes=(0, None, None))(points, lower, upper)
assert jnp.all(relaxations.convex <= jax.vmap(f)(points))
```

## Interval bounds alone

When only the bounds matter, `jaxormick.interval` skips the reference point. It returns
the lower and upper bounds of the output over the box.

<!-- name: test_basic -->
```python
low, high = jmk.interval(f)(lower, upper)
assert low == relaxation.lower and high == relaxation.upper
```

The rules still trace the convex and concave sides, but the bounds do not depend on
them, so XLA removes them from the compiled program.

## Function forms for tight relaxations

The relaxation depends on how the function is written, not only on its values. Each rule
sees only its own operands, so it relaxes a variable that occurs twice as if the two
occurrences could differ. A square written as a product has a much looser relaxation
than a power.

<!-- name: test_basic -->
```python
product = jmk.relax(lambda x: x * x)(jnp.array(0.0), jnp.array(-1.0), jnp.array(1.0))
power = jmk.relax(lambda x: x**2)(jnp.array(0.0), jnp.array(-1.0), jnp.array(1.0))
assert product.convex == -1.0 and product.lower == -1.0
assert power.convex == 0.0 and power.lower == 0.0
```

These rewrites tighten the relaxations:

- Write `x ** 2` instead of `x * x`, and `x ** 3` instead of `x * x * x`.
- Collect occurrences of a variable. `x * (1 - x)` relaxes worse than
  `0.25 - (x - 0.5) ** 2`.
- Use `jnp.tanh`, `jax.nn.sigmoid` and `jnp.log1p` directly instead of building them
  from `jnp.exp`.

## Inputs and outputs

The variables are a single array of any shape. The reference point and the bounds have
the same shape. `relax` converts integer inputs to the default floating-point dtype.

The function may return a pytree of arrays, such as a tuple or a dictionary. The result
has the same structure with a `Relaxation` at each leaf.

<!-- name: test_basic -->
```python
def two_outputs(x):
    return {"sum": x[0] + x[1], "product": x[0] * x[1]}


relaxations = jmk.relax(two_outputs)(reference, lower, upper)
assert relaxations["product"].lower == -1.0
```

`relax` compiles the relaxation with `jax.jit`. The first call traces and compiles it,
and later calls with the same shapes and dtypes reuse the program. See
[performance](performance.md) for how to keep calls fast.

## Unsupported operations

An operation without a relaxation rule raises `jaxormick.UnsupportedPrimitiveError` when
JAX traces the relaxed function. The message names the primitive. Comparisons of values
that depend on the variables are one example, since their results jump.

<!-- name: test_basic -->
```python
import pytest

with pytest.raises(jmk.UnsupportedPrimitiveError, match="gt"):
    jmk.relax(lambda x: jnp.where(x > 0, x, 0.0))(
        jnp.array(0.5), jnp.array(-1.0), jnp.array(1.0)
    )
```

`jax.nn.relu(x)` and `jnp.maximum(x, 0)` have rules. The list of [supported
operations](supported.md) has the rest.
