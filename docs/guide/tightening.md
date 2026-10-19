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
