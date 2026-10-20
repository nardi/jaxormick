# Supported operations

## Arithmetic

| Operation | JAX functions | Relaxation |
| --- | --- | --- |
| `neg`, `add`, `sub` | `-x`, `x + y`, `x - y` | Exact composition of the sides |
| `mul` | `x * y` | Scaling when one side is constant. A product of powers of the same value, such as `x * x`, becomes one power. McCormick's product rule otherwise |
| `div` | `x / y` | Product with the reciprocal. When the interval of `y` contains zero, the result has infinite bounds |
| `dot_general` | `x @ y`, `jnp.dot`, `jnp.einsum` | Scaling with a constant matrix, the product rule per term when both operands depend on the variables |
| `reduce_sum`, `cumsum` | `jnp.sum`, `jnp.cumsum` | Exact composition of the sides |
| `reduce_prod` | `jnp.prod` | The product rule on pairs of factors, in \(\lceil \log_2 n \rceil\) rounds |

## Powers and roots

| Operation | JAX functions | Relaxation |
| --- | --- | --- |
| `integer_pow`, `square` | `x ** 2`, `x ** 3`, `jnp.square` | Envelopes. Even powers are convex, odd powers use the envelope of [Liberti and Pantelides (2003)](https://doi.org/10.1023/A:1021924706467) when the interval contains zero. Negative powers have infinite bounds when the interval contains zero. A power of a power, such as `(x ** 2) ** 2`, becomes one power |
| `pow` | `x ** 1.5`, `x ** jnp.linspace(0.01, 0.06, n)` | Exponents known while tracing, as a scalar or as an array that broadcasts against the base. Elements with the same integer exponent or the same non-integer class (above 1, between 0 and 1, below 0) are relaxed together. Non-integer exponents need positive intervals |
| `sqrt`, `rsqrt` | `jnp.sqrt`, `lax.rsqrt` | Envelopes on nonnegative and positive intervals |
| `one_minus_square` | \(1 - x^2\) in the derivative of `jnp.tanh` in JAX 0.11 | Envelopes |

## Elementary functions

| Operation | JAX functions | Relaxation |
| --- | --- | --- |
| `exp`, `expm1`, `exp2` | `jnp.exp`, `jnp.expm1`, `jnp.exp2` | Envelopes |
| `log`, `log1p`, `log2` | `jnp.log`, `jnp.log1p`, `jnp.log2` | Envelopes. An interval that reaches the edge of the domain has a lower bound of `-inf` |
| `abs` | `jnp.abs` | Envelopes |
| `tanh`, `logistic` | `jnp.tanh`, `jax.nn.sigmoid` | Envelopes of convex-concave functions |
| `sinh`, `cosh` | `jnp.sinh`, `jnp.cosh` | Envelopes |
| `asinh`, `atan`, `cbrt` | `jnp.arcsinh`, `jnp.arctan`, `jnp.cbrt` | Envelopes of convex-concave functions |
| `asin`, `acos`, `acosh`, `atanh` | `jnp.arcsin`, `jnp.arccos`, `jnp.arccosh`, `jnp.arctanh` | Envelopes on intervals inside the open domain. An interval that reaches the edge of the domain only gets bounds, which are infinite for `atanh` |
| `erf`, `erfc`, `erf_inv` | `jax.scipy.special.erf`, `jax.scipy.special.erfc`, `jax.scipy.special.erfinv` | Envelopes. `erf_inv` handles the edge of its domain like `atanh` |
| `sin`, `cos`, `tan` | `jnp.sin`, `jnp.cos`, `jnp.tan` | Envelopes on intervals of any width. `tan` has infinite bounds on intervals that reach a pole |
| `max`, `min` | `jnp.maximum`, `jnp.minimum`, `jnp.clip`, `jax.nn.relu` | Through \(\max(a, b) = (a + b + \lvert a - b \rvert) / 2\), with the larger of two convex sides |
| `reduce_max`, `reduce_min` | `jnp.max`, `jnp.min` | The rule of `max` or `min` on pairs of terms, in \(\lceil \log_2 n \rceil\) rounds |

Functions such as `jax.nn.softplus` and `jax.nn.silu` relax through these rules, since
JAX builds them from the operations above.

## Array operations

Reshaping, broadcasting, transposing, slicing, splitting, indexing, concatenation,
padding, reversal and dtype conversion act on the four fields separately and are exact.
This covers `reshape`, `broadcast_in_dim`, `transpose`, `squeeze`, `expand_dims`,
`slice`, `split`, `unstack`, `dynamic_slice`, `gather`, `concatenate`, `stack`, `pad`,
`rev`, `copy` and `convert_element_type`. Indices must not depend on the variables.

Updates through `x.at[i].set(y)`, `x.at[i].add(y)` and `lax.dynamic_update_slice` are
exact too. They produce `scatter`, `scatter-add` and `dynamic_update_slice`. With
repeated indices, `scatter` keeps one of the updates for each position, and the four
fields always come from the same update. `x.at[i].multiply`, `x.at[i].min` and
`x.at[i].max` are not supported.

`jnp.where` and `lax.select_n` are supported when the condition does not depend on the
variables.

## Control flow

Calls to `jax.jit` functions, `jax.checkpoint`, `jax.custom_jvp` and `jax.custom_vjp`,
loops with a fixed length through `lax.scan` and `lax.fori_loop`, and `lax.cond` and
`lax.switch` with an index that does not depend on the variables. See [control
flow](control_flow.md).

Implicit functions through `lax.custom_linear_solve`, which covers `jnp.linalg.solve`,
and through `lax.custom_root` from `jax>=0.11` on.

## Known limits

- **No outward rounding.** `jaxormick` computes the bounds and relaxations in floating
  point without directed rounding. They hold in exact arithmetic, and in floating point
  they can be off by a few units in the last place. A rigorous global solver needs to
  widen them, for example by a relative tolerance on the order of `1e-6` in `float32`.
  Use `float64` (`jax.config.update("jax_enable_x64", True)`) when the bounds need to be
  tight to many digits.
- **Finite boxes.** Every bound of the box must be finite.
- **Implicit functions on wide boxes.** The relaxation of a solution of equations needs
  a verified enclosure. On wide boxes the verification may fail, and then the bounds and
  relaxations of the solution are infinite.
- **`lax.custom_root` before JAX 0.11.** Older versions hide the residual, so
  `lax.custom_root` raises `UnsupportedPrimitiveError` there. Roots with `has_aux=True`
  raise on every version.
- **Loops of unknown length.** `lax.while_loop` and `fori_loop` with traced bounds are
  not supported when the loop depends on the variables.
- **Comparisons and conditions.** Comparisons, `jnp.where` and `lax.cond` with a
  condition that depends on the variables raise, since the result would jump.
- **Positions of extrema.** `jnp.argmax` and `jnp.argmin` of values that depend on the
  variables raise, since the result would jump.
- **αBB of nonsmooth functions.** `alpha_bb=True` needs a rule for every primitive in
  the Hessian. Hessians of `jnp.abs`, `jnp.maximum`, `jnp.clip` and `jax.nn.softplus`
  contain comparisons and raise `UnsupportedPrimitiveError`.
- **Tightened bounds in loops and branches.** `tighten_bounds=True` tightens the outputs
  of `lax.scan` and `lax.cond`, but not the values inside their bodies.
- **Other functions.** Functions without a rule raise `UnsupportedPrimitiveError`.
  [Custom rules](extending.md) shows how to add one.
