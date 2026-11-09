# Relaxations in JAX

McCormick's method needs a function as a sequence of elementary operations, and affine
relaxations need subgradients. This page explains how `jaxormick` gets the first from
JAX's tracing and the second from JAX's automatic differentiation, and which limits
follow from compiling with XLA.

## Jaxprs as factorable representations

[JAX](https://docs.jax.dev/en/latest/) traces a Python function into a
[jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) ([Frostig, Johnson and Leary,
2018](https://mlsys.org/Conferences/doc/2018/146.pdf)). A jaxpr lists the elementary
operations of the function, one equation per operation, each applied to the results of
earlier equations. This is the factorable representation that McCormick's method needs.
For the example of the [previous page](mccormick.md):

<!-- name: test_jaxpr -->
```python
import jax
import jax.numpy as jnp


def f(x):
    return x[0] * jnp.exp(x[1]) - x[1] ** 2


print(jax.make_jaxpr(f)(jnp.zeros(2)))
```

prints

```
{ lambda ; a:f32[2]. let
    b:f32[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] a
    c:f32[] = squeeze[dimensions=(0,)] b
    d:f32[1] = slice[limit_indices=(2,) start_indices=(1,) strides=None] a
    e:f32[] = squeeze[dimensions=(0,)] d
    f:f32[] = exp e
    g:f32[] = mul c f
    h:f32[1] = slice[limit_indices=(2,) start_indices=(1,) strides=None] a
    i:f32[] = squeeze[dimensions=(0,)] h
    j:f32[] = integer_pow[y=2] i
    k:f32[] = sub g j
  in (k,) }
```

The factors \(v_1 = e^{x_2}\), \(v_2 = x_1 v_1\) and \(v_3 = x_2^2\) appear as `exp`,
`mul` and `integer_pow`. The jaxpr keeps `x[1] ** 2` as a power instead of a product, so
the tight univariate rule applies.

## Relaxation as an interpreter

`jaxormick.relax` traces the function to a jaxpr and evaluates it with a different
meaning for each operation. The JAX tutorial on [custom jaxpr
interpreters](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
explains this technique in general. Values that depend on the variables are represented
by `Relaxation` objects, which contain the four arrays of McCormick's method. For each
equation with such an operand, the interpreter calls the rule of its primitive. Values
that do not depend on the variables, such as constants and parameters, remain ordinary
arrays, and JAX evaluates the equations on them as usual.

### The example step by step

For the jaxpr above, the interpreter starts with `a` as a `Relaxation` of the box, whose
convex and concave values are the reference point.

1. `slice` and `squeeze` select one entry of each of the four arrays. They give the rows
   \(x_1\) and \(x_2\) of the [worked example](mccormick.md#a-worked-example) as `c` and
   `e`.
2. The rule of `exp` applies the composition rule to `e` and gives `f`, the row \(v_1\).
3. The rule of `mul` applies the product rule to `c` and `f` and gives `g`, the row
   \(v_2\).
4. The rule of `integer_pow` relaxes the square of `i` and gives `j`, the row \(v_3\).
   The slice `h` repeats `d`, so the interpreter merges them, as the next section
   describes, and `i` is the same variable as `e`.
5. The rule of `sub` subtracts the sides of `j` from those of `g` and gives `k`, the
   relaxation of \(f\).

### Merged equations and rewritten powers

JAX does not merge repeated equations, so `x[0] * x[0]` slices `x` twice and multiplies
two different variables. The interpreter merges equations that have the same primitive,
parameters and operands, so that both factors become the same variable. It then rewrites
a product of two powers of the same variable as one `integer_pow`, and a power of a
power as well. Equations with side effects, such as `jax.debug.print`, are never merged.

### JAX transformations of the relaxation

The rules are written in `jax.numpy`, so the relaxed function is itself a JAX function.
Every JAX transformation applies to it.

- `jax.jit` compiles the whole relaxation into one XLA program. The rules run once while
  tracing and not on each call.
- `jax.vmap` evaluates the relaxation on many boxes at once. Branch and bound has many
  open boxes, and a GPU evaluates a batch of them in parallel.
- `jax.grad`, `jax.jacfwd` and `jax.jacrev` differentiate the relaxations, and the
  derivatives are the subgradients for affine relaxations.

The function to relax can itself come from a JAX transformation. `jax.hessian(f)` is a
JAX function, so `interval` bounds the Hessian of `f` over a box. The αBB relaxation of
`relax(f, alpha_bb=True)` uses these bounds. Hessians often contain `split`, which has a
rule for that reason. In JAX 0.11, the derivative of `tanh` contains `one_minus_square`,
which has a rule too.

### Tightened bounds by forward-mode differentiation

With `tighten_bounds=True`, the interpreter computes a second relaxation of every value,
at the middle of the box, together with its subgradients there. It propagates the
subgradients through each rule with `jax.linearize`, one direction per variable. This is
forward-mode differentiation inside the interpreter. The bodies of call primitives such
as `jit` are evaluated equation by equation, so their intermediate values are tightened
too. The bodies of `scan` and `cond` are evaluated as one equation, so only their
outputs are tightened.

### Implicit functions

Implicit functions have their own rules. The interpreter recognizes
`lax.custom_linear_solve`, and from JAX 0.11 on the `CustomRoot` object of
`lax.custom_root`. It relaxes their solutions from the residual with the Krawczyk
operator, which it relaxes with the same interpreter. [Implicit functions](implicit.md)
describes the method.

The function to relax can be the same JAX code that a model or a simulation already
runs, without a separate implementation for the solver.

## Subgradients from automatic differentiation

The rules compute the convex side \(f^{cv}\) with `jnp.exp`, `jnp.where`, products and
sums. Away from the points where a `jnp.where` switches branches, it is differentiable,
and its gradient is the subgradient that [Mitsos, Chachuat and Barton
(2009)](https://doi.org/10.1137/080717341) propagate by hand. At a switch, automatic
differentiation follows the branch that the condition picked and returns one of the
one-sided derivatives. Both are subgradients when the function is convex. [Griewank and
Walther (2008)](https://doi.org/10.1137/1.9780898717761) and the survey of [Baydin et
al. (2018)](https://jmlr.org/papers/v18/17-468.html) explain forward and reverse mode in
detail, and the [autodiff
cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) shows how JAX
exposes them.

Three details of floating-point code need care, so that the gradient is a subgradient at
every point.

### Medians at ties

A median written as `jnp.minimum(jnp.maximum(a, b), c)` has the right value, but where
two arguments tie, `jax.grad` splits the gradient between them. At a tie between the
convex and concave sides of an operand, half the gradient would come from each. The mix
is neither a subgradient of the convex side nor a supergradient of the concave one, and
its tangent plane can cut off part of the function. Ties happen at the box boundary,
where linear programs attain their optima. `jaxormick` therefore writes the median as an
explicit choice of one branch.

### Maxima at near ties

The product rule and the clipping of relaxations to their bounds take maxima and minima
of relaxations too. These ties are harmless in exact arithmetic, since an average of
subgradients of the active pieces is a subgradient of their maximum. In floating point
they are not. XLA may evaluate the comparison that decides a tie in several fused
kernels, each with its own rounding. The gradient contributions of these kernels can
then assume different active pieces, and their sum is no subgradient. `jaxormick`
therefore keeps one fixed argument unless the other is larger by more than a few ulps,
so that rounding cannot flip the choice. The result may fall short of the maximum by
these few ulps, which only loosens it. When a relaxation is clipped to its bound, the
bound is chosen at such a near tie, since a constant is a valid plane even where
rounding pushed the relaxation past it.

### `NaN` in unused branches

`jnp.where(condition, a, b)` computes both `a` and `b`, also where it discards one of
them. If `b` is `NaN` where it is not used, for example the logarithm of a negative
bound, its gradient is still `NaN`, and the gradient of the result becomes `NaN` too.
The rules replace invalid inputs before they reach such a branch.

## Static shapes

XLA compiles programs with fixed shapes, and under `jax.vmap` a loop of varying length
runs until the slowest box finishes. The rules therefore use no loops whose length
depends on the data.

- Envelopes of convex-concave functions find their tangent point by a fixed number of
  bisection steps, enough to reach the precision of the dtype. A slightly wrong tangent
  point still leaves a valid underestimator, since the rule uses the end of the bracket
  where the line is below the function.
- Odd powers use a closed form for the tangent point.
- Case analyses, such as which parts of the period of \(\cos\) an interval contains,
  compute every case and select one with `jnp.where`.

## Parameters and control flow

The arguments after the variables are passed to the function unchanged. They stay
ordinary arrays inside the relaxation, so one compiled relaxation is used for every
parameter value, for example when solving the same model for many data sets.

Loops with a fixed trip count, such as `jax.lax.scan` and `jax.lax.fori_loop`, are
relaxed as loops. `jaxormick` relaxes the body once and runs it inside a `scan` with a
`Relaxation` carry, so compile time does not grow with the number of iterations.
[Mitsos, Chachuat and Barton (2009)](https://doi.org/10.1137/080717341) named this
approach McCormick relaxations of algorithms.

Loops whose length depends on the data, such as `lax.while_loop`, have no such
relaxation. Solvers that `lax.custom_root` or `lax.custom_linear_solve` wraps do, since
the relaxation comes from the residual and not from the loop.

## Other implementations

MC++ in C++, with the Python interface [pymcpp](https://pypi.org/project/pymcpp/), and
the [EAGO](https://doi.org/10.1080/10556788.2020.1786566) solver in Julia implement
McCormick relaxations with operator overloading. Both support more operations than
`jaxormick`, which works on existing JAX code, and whose relaxations can be compiled
with XLA and batched with `jax.vmap`.
