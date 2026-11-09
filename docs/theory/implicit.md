# Implicit functions

Some values in a program have no formula. They are defined by an equation
\(r(p, x) = 0\), and a solver computes them. When the equation has one solution for each
\(p\), that solution is a function \(x(p)\) of the parameters, called an implicit
function. In a relaxation, the parameters depend on the variables of the box, so
\(x(p)\) needs a relaxation as well.

JAX computes such values with
[`lax.custom_root`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html)
and
[`lax.custom_linear_solve`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html),
and `jnp.linalg.solve` calls the second. Both take the residual \(r\) as an argument
next to the solver. `jaxormick` relaxes the solution from the residual and never
relaxes the solver.

Relaxing the solver's iterates would also be valid, as for any [loop of fixed
length](../guide/control_flow.md#loops-with-a-fixed-length). The bounds would be loose,
though, because each step relaxes the previous iterate and the errors add up. The
residual describes the solution itself, so a relaxation built from it does not depend
on how many steps the solver takes.

## Overview of the method

The method has two stages.

1. A box \(X\) is found that contains exactly one solution for every \(p\) in the box
   of parameters \(P\). This box is called an enclosure. Its bounds are valid bounds of
   \(x(p)\), and they are also a first, constant relaxation of it.
2. The solution is a fixed point of a map that is built from the residual. Relaxing
   that map with McCormick's rules, with the current relaxation of \(x(p)\) as input,
   gives a new relaxation of \(x(p)\). Repeating this step tightens the relaxation.

Both stages use the same map, the Krawczyk operator.

## The Krawczyk operator

Newton's method moves a guess towards the solution by a step that divides the residual
by its derivative. The Krawczyk operator is a Newton step with two changes. The
derivative is replaced by a fixed matrix \(Y\), an approximate inverse of the Jacobian
\(\partial_x r\). And the step is applied to a whole box \(X\) at once in [interval
arithmetic](relaxations.md#interval-bounds), so that the result is a box as well. For a
point \(\hat x\) in \(X\), it is

\[
K(p, X) = \hat x - Y r(p, \hat x) + \bigl(I - Y \partial_x r(p, X)\bigr)(X - \hat x).
\]

The first two terms are the Newton step from \(\hat x\). The last term accounts for the
error of the fixed matrix over the box. When \(Y\) is close to the inverse Jacobian and
the box is small, this term is small, and the operator maps the box to a smaller one.

By the mean value theorem, every solution in \(X\) also lies in \(K(p, X)\). [Krawczyk
(1969)](https://doi.org/10.1007/BF02234767) and [Moore
(1977)](https://doi.org/10.1137/0714040) showed more. If \(K(p, X)\) lies strictly
inside \(X\), then \(X\) contains exactly one solution. [Neumaier
(1990)](https://doi.org/10.1017/CBO9780511526473) extended this to all \(p\) in a box
\(P\).

`jaxormick` takes \(\hat x\) as the middle of \(X\) and \(Y\) as the inverse Jacobian at
the middle of the box.

## Enclosures by ε-inflation

To find an enclosure, `jaxormick` starts from the solution at the middle of \(P\), which
the solver of `lax.custom_root` computes. It then repeats two steps. It widens the
candidate box a little, and it applies the operator to it. Once the image lies strictly
inside the widened box, the image is the enclosure. Otherwise the image is the next
candidate. This search is called ε-inflation, and [Rump
(2010)](https://doi.org/10.1017/S096249291000005X) describes it in his survey of
verification methods.

After 10 steps without success, the bounds and relaxations of the solution are
infinite. They are valid then, but useless. On wide boxes this is common, because the
operator does not contract there. Branch and bound splits the box, and on smaller boxes
the search succeeds.

## Relaxed Krawczyk steps

For every \(p\) in \(P\), the solution satisfies \(x(p) = K(p, x(p))\). [Stuber, Scott
and Barton (2015)](https://doi.org/10.1080/10556788.2014.924514) turned this identity
into relaxations. The right-hand side is relaxed with McCormick's rules, using the
current relaxation of \(x(p)\) as the relaxation of its operand. The result is again a
relaxation of \(x(p)\), and the intersection of the old and the new relaxation is a
tighter one.

`jaxormick` starts from the bounds of the enclosure and takes 5 of these steps. The
middle of the enclosure and the matrix \(Y\) depend only on the box, so they are
constants in each step, and the results are convex and concave relaxations.

## Implementation in `jaxormick`

- From JAX 0.11 on, `lax.custom_root` appears in the jaxpr as `call_hi_primitive`, and
  its parameter `_prim` contains the residual. Before JAX 0.11, the residual is only
  part of the derivative rule, so `jaxormick` raises `UnsupportedPrimitiveError` for
  such a root. `lax.custom_root` with `has_aux=True` is not supported.
- `lax.custom_linear_solve` contains the jaxpr of the matrix-vector product on every
  JAX version, so its residual \(A(p) x - b(p)\) is always available. The constants of
  its solvers, such as the LU factors of `jnp.linalg.solve`, are never read.
- An enclosure that is valid on a box is also valid on every smaller box inside it.
  With `enclosures=True`, the functions returned by `relax` and `interval` take the
  enclosures of a parent box and return the ones they verified, so that branch and
  bound can pass them on to the children. `root_enclosure_size` returns the length of
  that array.
- Each step needs the Jacobian of the residual as a dense \(n \times n\) matrix of
  relaxations for a solution with \(n\) entries.
- If the equation has several solutions, the relaxation describes the one that the
  solver returns at the middle of the box.

## Further reading

- [Moore, Kearfott and Cloud (2009)](https://doi.org/10.1137/1.9780898717716) introduce
  interval arithmetic and interval Newton methods at textbook level.
- [Neumaier (1990)](https://doi.org/10.1017/CBO9780511526473) treats the Krawczyk
  operator and other interval methods for systems of equations in depth.
- [Rump (2010)](https://doi.org/10.1017/S096249291000005X) surveys verification methods
  in floating point arithmetic, including ε-inflation.
- [Stuber, Scott and Barton (2015)](https://doi.org/10.1080/10556788.2014.924514)
  develop the relaxations of implicit functions that `jaxormick` implements.
- The JAX documentation of
  [`lax.custom_root`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html)
  describes how a solver and its residual are passed to JAX.
