# Supported operations

## Arithmetic

| Operation | JAX functions | Relaxation |
| --- | --- | --- |
| `neg`, `add`, `sub` | `-x`, `x + y`, `x - y` | Exact composition of the sides |
| `mul` | `x * y` | Scaling when one side is constant. A product of powers of the same value, such as `x * x`, becomes one power. McCormick's product rule otherwise |
| `div` | `x / y` | Product with the reciprocal. When the interval of `y` contains zero, the result has infinite bounds |
| `dot_general` | `x @ y`, `jnp.dot`, `jnp.einsum` | Scaling with a constant matrix, the product rule per term when both operands depend on the variables |
| `reduce_sum`, `cumsum` | `jnp.sum`, `jnp.cumsum` | Exact composition of the sides |

## Powers and roots

| Operation | JAX functions | Relaxation |
| --- | --- | --- |
| `integer_pow`, `square` | `x ** 2`, `x ** 3`, `jnp.square` | Envelopes. Even powers are convex, odd powers use the envelope of [Liberti and Pantelides (2003)](https://doi.org/10.1023/A:1021924706467) when the interval contains zero. Negative powers have infinite bounds when the interval contains zero. A power of a power, such as `(x ** 2) ** 2`, becomes one power |
| `pow` | `x ** 1.5` | Constant exponents only, on positive intervals for non-integer exponents |
| `sqrt`, `rsqrt` | `jnp.sqrt`, `lax.rsqrt` | Envelopes on nonnegative and positive intervals |
| `one_minus_square` | \(1 - x^2\) in the derivative of `jnp.tanh` in JAX 0.11 | Envelopes |

## Elementary functions

| Operation | JAX functions | Relaxation |
| --- | --- | --- |
| `exp`, `expm1` | `jnp.exp`, `jnp.expm1` | Envelopes |
| `log`, `log1p` | `jnp.log`, `jnp.log1p` | Envelopes. An interval that reaches the edge of the domain has a lower bound of `-inf` |
| `abs` | `jnp.abs` | Envelopes |
| `tanh`, `logistic` | `jnp.tanh`, `jax.nn.sigmoid` | Envelopes of convex-concave functions |
| `sin`, `cos` | `jnp.sin`, `jnp.cos` | Envelopes on intervals of any width |
| `max`, `min` | `jnp.maximum`, `jnp.minimum`, `jnp.clip`, `jax.nn.relu` | Through \(\max(a, b) = (a + b + \lvert a - b \rvert) / 2\), with the larger of two convex sides |

Functions such as `jax.nn.softplus` and `jax.nn.silu` relax through these rules, since
JAX builds them from the operations above.

## Array operations

Reshaping, broadcasting, transposing, slicing, splitting, indexing, concatenation,
padding, reversal and dtype conversion act on the four fields separately and are exact.
This covers `reshape`, `broadcast_in_dim`, `transpose`, `squeeze`, `expand_dims`,
`slice`, `split`, `dynamic_slice`, `gather`, `concatenate`, `stack`, `pad`, `rev`,
`copy` and `convert_element_type`. Indices must not depend on the variables.

`jnp.where` and `lax.select_n` are supported when the condition does not depend on the
variables.

## Control flow

Calls to `jax.jit` functions, `jax.checkpoint`, `jax.custom_jvp` and `jax.custom_vjp`,
loops with a fixed length through `lax.scan` and `lax.fori_loop`, and `lax.cond` and
`lax.switch` with an index that does not depend on the variables. See [control
flow](control_flow.md).

## Known limits

- **No outward rounding.** `jaxormick` computes the bounds and relaxations in floating
  point without directed rounding. They hold in exact arithmetic, and in floating point
  they can be off by a few units in the last place. A rigorous global solver needs to
  widen them, for example by a relative tolerance on the order of `1e-6` in `float32`.
  Use `float64` (`jax.config.update("jax_enable_x64", True)`) when the bounds need to be
  tight to many digits.
- **Finite boxes.** Every bound of the box must be finite.
- **Implicit functions.** `lax.custom_root`, `lax.custom_linear_solve` and the solvers
  built on them, such as `jnp.linalg.solve`, raise `UnsupportedPrimitiveError`. Relaxing
  them needs the generalized relaxations of [Scott, Stuber and Barton
  (2011)](https://doi.org/10.1007/s10898-011-9664-7), which a later version may add.
- **Loops of unknown length.** `lax.while_loop` and `fori_loop` with traced bounds are
  not supported when the loop depends on the variables.
- **Comparisons and conditions.** Comparisons, `jnp.where` and `lax.cond` with a
  condition that depends on the variables raise, since the result would jump.
- **Reductions to a maximum.** `jnp.max`, `jnp.min` and `jnp.argmax` over an axis are
  not supported yet. A loop of `jnp.maximum` over the elements works.
- **αBB of nonsmooth functions.** `alpha_bb=True` needs a rule for every primitive in
  the Hessian. Hessians of `jnp.abs`, `jnp.maximum`, `jnp.clip` and `jax.nn.softplus`
  contain comparisons and raise `UnsupportedPrimitiveError`.
- **Tightened bounds in loops and branches.** `tighten_bounds=True` tightens the outputs
  of `lax.scan` and `lax.cond`, but not the values inside their bodies.
- **Other functions.** `jnp.arctan`, `jnp.sinh`, `jnp.cosh`, `jax.scipy.special.erf` and
  other functions without a rule raise `UnsupportedPrimitiveError`. [Custom
  rules](extending.md) shows how to add one.
