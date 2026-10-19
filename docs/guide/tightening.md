# Tighter relaxations

## Intersections of relaxations

The same function can often be written in several forms, and each form relaxes
differently. `jaxormick.intersect_relaxations` combines relaxations of the same function
over the same box into one. It takes the larger convex side, the smaller concave side
and the intersection of the intervals. The result is a valid relaxation and at least as
tight as each of its inputs.

On \([-1, 2]\), \(x^3 - x\) and \(x (x^2 - 1)\) are the same function. The sum of
powers has tighter convex and concave sides, because each power gets its envelope. The
product has a tighter upper bound, \(6\) instead of \(9\), which is the maximum of the
function on the box.

<!-- name: test_tightening -->
```python
import jax.numpy as jnp

import jaxormick as jmk

reference = jnp.array(0.5)
lower = jnp.array(-1.0)
upper = jnp.array(2.0)

expanded = jmk.relax(lambda x: x**3 - x)(reference, lower, upper)
factored = jmk.relax(lambda x: x * (x**2 - 1))(reference, lower, upper)
both = jmk.intersect_relaxations(expanded, factored)

assert both.convex == expanded.convex > factored.convex
assert both.concave == expanded.concave < factored.concave
assert both.upper == factored.upper == 6.0
```

The relaxations may also be pytrees of `Relaxation` objects with the same structure,
such as the outputs of two forms of a function with several outputs. The derivatives of
the result follow the input that attains the maximum or the minimum, so
`jax.grad` of the intersected convex side is a subgradient.

Each form costs a relaxation of its own. Relaxing two forms costs about twice as much as
relaxing one, and both run in the same compiled program when the call is inside
`jax.jit`.

## αBB relaxations

`relax(f, alpha_bb=True)` intersects the McCormick relaxation with the αBB relaxation of
[Adjiman et al. (1998)](https://doi.org/10.1016/S0098-1354(98)00027-1). αBB adds a
quadratic to `f` that is just large enough to make it convex, based on bounds of the
Hessian of `f` over the box. It is often tighter for smooth functions whose factors
relax poorly one at a time. The indefinite quadratic form below is one.

<!-- name: test_alpha_bb -->
```python
import jax.numpy as jnp

import jaxormick as jmk

matrix = jnp.array([[2.0, -1.0], [0.5, 1.0]])


def quadratic_form(x):
    return x @ (matrix @ x)


reference = jnp.array([0.1, -0.2])
lower = -jnp.ones(2)
upper = jnp.ones(2)

mccormick = jmk.relax(quadratic_form)(reference, lower, upper)
both = jmk.relax(quadratic_form, alpha_bb=True)(reference, lower, upper)

assert both.convex > mccormick.convex
assert both.concave < mccormick.concave
```

If the bounds of the Hessian show that `f` is convex on the box, the convex side is `f`
itself. The same holds for the concave side of a concave function.

`convex_side`, `concave_side` and `affine_relaxation` take the same keyword. αBB needs a
function that returns one array, and a rule for every primitive in its Hessian.
Functions built with `jnp.abs`, `jnp.maximum` or `jnp.clip` have comparisons in their
Hessians, and `alpha_bb=True` raises `UnsupportedPrimitiveError` for them. The interval
bounds are the McCormick bounds, since αBB only changes the convex and concave sides.

The Hessian of a function of \(n\) variables has \(n^2\) entries, and each costs an
interval evaluation of a second derivative. See
[performance](performance.md#cost-of-each-transformation) for measurements.

## Subgradient-tightened bounds

`relax(f, tighten_bounds=True)` narrows the interval of every intermediate value with
its affine relaxations at the middle of the box, after [Najman and Mitsos
(2019)](https://doi.org/10.1007/s10898-019-00791-0). A narrower interval of an inner
value also tightens the relaxations of the values computed from it. In \(e^{x - x^2}\)
on \([0, 2]\), the rules bound the exponent by \([-4, 2]\), and the tightened bounds by
\([-2, 1]\).

<!-- name: test_tighten_bounds -->
```python
import jax.numpy as jnp
import numpy as np

import jaxormick as jmk


def function(x):
    return jnp.exp(x[0] - x[0] ** 2)


lower = jnp.array([0.0])
upper = jnp.array([2.0])

plain = jmk.interval(function)(lower, upper)
tightened = jmk.interval(function, tighten_bounds=True)(lower, upper)

np.testing.assert_allclose(plain, (np.exp(-4.0), np.exp(2.0)), rtol=1e-6)
np.testing.assert_allclose(tightened, (np.exp(-2.0), np.e), rtol=1e-6)
```

`interval`, `convex_side`, `concave_side` and `affine_relaxation` take the same keyword.
The tightened bounds are never wider than the plain ones. Values inside the bodies of
`lax.scan` and `lax.cond` keep their plain bounds, but the outputs of the loop or branch
are tightened.

Each relaxed value carries one subgradient per variable, so the cost grows with the
number of variables, like forward-mode differentiation. See
[performance](performance.md#cost-of-each-transformation) for measurements.
