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

## Products of many factors

A product of power factors with a different exponent for each factor is one `jnp.prod`
of a power with an exponent array. The rule for `pow` relaxes the elements with the same
integer exponent, or the same class of non-integer exponent, together, and `reduce_prod`
combines the factors in a logarithmic number of levels. The exponent array has to be
known while tracing, as the one here is.

<!-- name: test_performance -->
```python
def product(x, scale):
    return jnp.prod((scale * x) ** jnp.linspace(0.01, 0.06, x.shape[0]))


size = 24
value, gradient = jmk.convex_side(product)(
    jnp.ones(size), jnp.full(size, 0.1), jnp.full(size, 10.0), jnp.ones(size)
)
assert gradient.shape == (size,)
```

A Python loop of `*` over the same factors compiles much more slowly. Each `mul` adds
one product step to a chain as long as the product, and XLA's compile time grows much
faster than the program does. The table gives the lines of StableHLO of
`jmk.convex_side` and the time to lower and compile it on the CPU of the machine above,
with `float64`. The left fold of `*` is not measured beyond 24 factors, and the stack of
separate scalar powers beyond 30.

| Factors | Left fold of `*` | `jnp.prod` of stacked powers | `jnp.prod` with an array exponent |
| --- | --- | --- | --- |
| 8 | 2366 lines, 3.2 s | 1958 lines, 2.2 s | 966 lines, 1.1 s |
| 16 | 4918 lines, 14.4 s | 3409 lines, 3.8 s | 1225 lines, 1.8 s |
| 24 | 7470 lines, 69.8 s | 4851 lines, 6.2 s | 1463 lines, 2.3 s |
| 30 | | 5766 lines, 7.3 s | 1484 lines, 2.6 s |
| 1000 | | | 2779 lines, 8.7 s |

Stacking the separate powers and calling `jnp.prod` helps, but it builds one relaxation
for each factor. Prefer the array exponent, or one scalar exponent when the factors
share it.

## Cost of each transformation

`interval`, `convex_side` and `concave_side` compute less than `relax` and
`affine_relaxation`, since XLA removes the parts of the rules that their outputs do not
need. Asking for a gradient costs a pass of automatic differentiation on top of the
relaxation.

Forward mode, the default for gradients, is faster for functions of up to a few dozen
variables. Reverse mode is faster for many variables, as for the quadratic form in 100
variables below.

`intersect_relaxations` costs one relaxation for each form it combines, plus an
elementwise maximum and minimum.

An implicit function with \(n\) unknowns costs 15 relaxed evaluations of its residual
and of the residual's \(n \times n\) Jacobian, plus the inverse of one Jacobian. 10 of
the evaluations verify the enclosure and 5 tighten the relaxations.

`alpha_bb=True` adds an interval evaluation of the Hessian. The table compares `relax`
with and without it on batches of 1024 different boxes of the benchmark functions, with
each coordinate of the box 0.1 or 1 wide. The last two columns give the mean gap between
the concave and convex sides with αBB, relative to the gap without it.

| Function | Run time | Compile time | Gap, width 1 | Gap, width 0.1 |
| --- | --- | --- | --- | --- |
| Six-hump camel | 1.5–2.1× | 5.8× | 1.00 | 0.63 |
| Sum of 50 exponentials | 30–34× | 4× | 1.00 | 1.00 |
| Newton iteration, 20 steps | 2.4–3.2× | 6–8× | 1.00 | 0.55 |
| Quadratic form | 5.5–6.1× | 4–4.5× | 0.94 | 0.94 |
| `sin`, `cos` and `tanh` | 1.1× | 2.3× | 1.00 | 0.97 |

αBB helps most on small boxes, where its quadratic term is small. Branch and bound
reaches such boxes after a few splits. The sum of 50 exponentials shows the cost of a
Hessian with 2500 entries, for no gain on these boxes.

`tighten_bounds=True` relaxes every value a second time at the middle of the box and
pushes one subgradient per variable through each rule. The table compares `relax` with
and without it on the same batches. The last two columns give the mean width of the
interval of the output with tightened bounds, relative to the width without them.

| Function | Run time | Compile time | Width, width 1 | Width, width 0.1 |
| --- | --- | --- | --- | --- |
| Six-hump camel | 4.5–5.4× | 4.8× | 0.97 | 0.81 |
| Sum of 50 exponentials | 54–55× | 2.2× | 1.00 | 0.79 |
| Newton iteration, 20 steps | 2.6–3.0× | 3.4–3.8× | 0.97 | 0.37 |
| Quadratic form | 30–38× | 2.5× | 1.00 | 0.36 |
| `sin`, `cos` and `tanh` | 1.3–1.4× | 3.1–3.6× | 0.99 | 0.91 |

The run time grows with the number of variables, as for forward-mode differentiation.
The quadratic form has 100 variables and the sum of exponentials 50. The bounds narrow
most on small boxes, where the affine relaxations are close to the function. On these
benchmarks the gap between the convex and concave sides changes by at most 1%.

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
