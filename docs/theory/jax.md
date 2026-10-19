# Relaxations in JAX

## Jaxprs as factorable representations

[JAX](https://docs.jax.dev/en/latest/) traces a Python function into a jaxpr ([Frostig,
Johnson and Leary, 2018](https://mlsys.org/Conferences/doc/2018/146.pdf)). A jaxpr lists
the elementary operations of the function, one equation per operation, each applied to
the results of earlier equations. This is the factorable representation that McCormick's
method needs. For the example of the [previous page](mccormick.md):

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
meaning for each operation. Values that depend on the variables become `Relaxation`
objects, which hold the four arrays of McCormick's method. For each equation with such
an operand, the interpreter calls the rule of its primitive. Values that do not depend
on the variables, such as constants and parameters, stay ordinary arrays and run through
JAX as usual.

JAX does not merge repeated equations, so `x[0] * x[0]` slices `x` twice and multiplies
two different variables. The interpreter merges equations that have the same primitive,
parameters and operands, so that both factors become the same variable. It then rewrites
a product of two powers of the same variable as one `integer_pow`, and a power of a
power as well. Equations with side effects, such as `jax.debug.print`, are never merged.

The rules are written in `jax.numpy`, so the relaxed function is itself a JAX function.
Every JAX transformation applies to it.

- `jax.jit` compiles the whole relaxation into one XLA program. The rules run once while
  tracing and not on each call.
- `jax.vmap` evaluates the relaxation on many boxes at once. Branch and bound keeps many
  boxes open, and batching them suits a GPU.
- `jax.grad`, `jax.jacfwd` and `jax.jacrev` differentiate the relaxations, and the
  derivatives are the subgradients for affine relaxations.

The function to relax can be the same JAX code that a model or a simulation already
runs, without a separate implementation for the solver.

## Subgradients from automatic differentiation

The rules compute the convex side \(f^{cv}\) with `jnp.exp`, `jnp.where`, products and
sums. Away from the points where a `jnp.where` switches branches, it is differentiable,
and its gradient is the subgradient that [Mitsos, Chachuat and Barton
(2009)](https://doi.org/10.1137/080717341) propagate by hand. At a switch, automatic
differentiation follows the branch that the condition picked and returns one of the
one-sided derivatives. Both are subgradients when the function is convex.

A median written as `jnp.minimum(jnp.maximum(a, b), c)` has the right value, but where
two arguments tie, `jax.grad` splits the gradient between them. At a tie between the
convex and concave sides of an operand, half the gradient would come from each. The mix
is neither a subgradient of the convex side nor a supergradient of the concave one, and
its tangent plane can cut off part of the function. Ties happen at the box boundary,
where linear programs find their solutions. `jaxormick` therefore writes the median as
an explicit choice of one branch.

The product rule and the clipping of relaxations to their bounds take maxima and minima
of relaxations too. These ties are harmless in exact arithmetic, since an average of
subgradients of the active pieces is a subgradient of their maximum. In floating point
they are not. XLA may evaluate the comparison that decides a tie in several fused
kernels, each with its own rounding. The paths of the gradient then disagree about the
active piece, and their sum is no subgradient. `jaxormick` therefore keeps one fixed
argument unless the other is larger by more than a few ulps, so that rounding cannot
flip the choice. The result may fall short of the maximum by these few ulps, which only
loosens it. When a relaxation is clipped to its bound, the bound wins such a near tie,
since a constant is a valid plane even where rounding pushed the relaxation past it.

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
  point still leaves a valid underestimator, since the rule keeps the end of the bracket
  where the line stays below the function.
- Odd powers use a closed form for the tangent point.
- Case analyses, such as which part of the period of \(\cos\) an interval covers,
  compute every case and select one with `jnp.where`.

## Parameters and control flow

The relaxed function passes arguments after the variables to the function unchanged.
They stay ordinary arrays inside the relaxation, so one compiled relaxation serves every
parameter value, for example when solving the same model for many data sets.

Loops with a fixed trip count, such as `jax.lax.scan` and `jax.lax.fori_loop`, relax as
loops. `jaxormick` relaxes the body once and runs it inside a `scan` with a `Relaxation`
carry, so compile time does not grow with the number of iterations. [Mitsos, Chachuat
and Barton (2009)](https://doi.org/10.1137/080717341) named this approach McCormick
relaxations of algorithms.

## Other implementations

MC++ in C++, with the Python interface [pymcpp](https://pypi.org/project/pymcpp/), and
the [EAGO](https://doi.org/10.1080/10556788.2020.1786566) solver in Julia implement
McCormick relaxations with operator overloading. Both cover more operations than
`jaxormick`, which works on existing JAX code, and its relaxations compile with XLA and
batch under `jax.vmap`.
