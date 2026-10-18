# Performance

## Compilation and reuse

Every transformation returns a function compiled with `jax.jit`. The first call with new
shapes or dtypes traces the function, runs the rules and compiles the result. Later
calls reuse the program. Create the relaxed function once, outside the loop that calls
it.

<!-- name: test_performance -->
```python
import jax
import jax.numpy as jnp

import jaxormick as jmk


def f(x):
    return jnp.sum(jnp.tanh(x) * x**2)


relaxed = jmk.relax(f)
for width in [1.0, 0.5, 0.25]:
    relaxation = relaxed(jnp.zeros(3), jnp.full(3, -width), jnp.full(3, width))
```

Calling `jmk.relax(f)` inside the loop would compile again on each iteration, since each
call of `jmk.relax` returns a new function. Keep shapes and dtypes fixed for the same
reason. Constant arguments, as in [parameters and batches](parameters.md), can change
without recompiling.

## Batches of boxes

A single call of a small relaxation takes a few microseconds, most of it dispatch.
Mapping over a batch of boxes with `jax.vmap` spreads that over the whole batch, and on
a GPU the boxes run in parallel.

<!-- name: test_performance -->
```python
lowers = jax.random.uniform(jax.random.key(0), (1024, 3), minval=-1.0, maxval=0.0)
uppers = lowers + 0.5
batched = jax.jit(jax.vmap(jmk.interval(f)))
lows, highs = batched(lowers, uppers)
assert lows.shape == (1024,)
```

## Cost of each transformation

`interval`, `convex_side` and `concave_side` compute less than `relax` and
`affine_relaxation`, since XLA removes the parts of the rules that their outputs do not
need. Asking for a gradient costs a pass of automatic differentiation on top of the
relaxation.

Forward mode, the default for gradients, is faster for functions of up to a few dozen
variables. Reverse mode is faster for many variables, as for the quadratic form in 100
variables below.

## Measured costs

The benchmarks in `benchmarks/` time each transformation on five functions and divide by
the time of the plain function on the same machine. The ratios below are medians from a
4-core Intel Xeon CPU at 2.1 GHz, with `float32`.

| Function | Variables | `interval` | `relax` | `convex_side` | `affine_relaxation`, forward | `affine_relaxation`, reverse |
| --- | --- | --- | --- | --- | --- | --- |
| Six-hump camel | 2 | 1.3 | 2.2 | 1.4 | 5.6 | 5.5 |
| Sum of 50 exponentials | 50 | 7.2 | 10.2 | 11.6 | 17.9 | 9.3 |
| Newton iteration, 20 steps | 1 | 1.6 | 2.1 | 2.0 | 2.8 | 11.8 |
| Quadratic form | 100 | 2.4 | 3.1 | 4.9 | 8.8 | 3.3 |
| `sin`, `cos` and `tanh` | 2 | 1.4 | 10.8 | 10.1 | 13.8 | 13.8 |

Single calls of the plain functions take 7 to 25 microseconds here, mostly dispatch,
which hides part of the cost of the rules. A batch of 1024 boxes shows the cost more
clearly, with `relax` taking 6 to 66 times as long as the plain function, and
`affine_relaxation` in forward mode taking 15 to 390 times as long. The upper ends are
the trigonometric rules and the forward-mode gradient of the quadratic form in 100
variables.

Compiling a relaxation takes 2 to 13 times as long as compiling the plain function, and
adding its gradients 3 to 49 times. The trigonometric envelopes are the most expensive
to compile, since they evaluate several cases with a bisection each.

To run the benchmarks and compare with the committed baseline:

```bash
uv run pytest benchmarks -o addopts="" --benchmark-only --benchmark-json=results.json
uv run python benchmarks/compare.py results.json benchmarks/baseline.json
```
