# Implicit functions

Some values in a program have no formula. They are defined by an equation, and a solver
computes them. This page explains how `jaxormick` relaxes such a value when the program
contains only the equation and a solver for it.

## Values defined by equations

Take parameters \(p\) and an unknown \(x\) that solves the equation

\[
r(p, x) = 0.
\]

The function \(r\) is the residual. When the equation has one solution for each \(p\),
that solution is a function \(x(p)\) of the parameters. It is called an implicit
function, because no formula for it appears in the program. In a relaxation, the
parameters depend on the variables of the box, so \(x(p)\) needs a relaxation as well.

In JAX, such values are computed with
[`lax.custom_root`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html)
and
[`lax.custom_linear_solve`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html).
Both take the residual \(r\) as an argument next to the solver, and `jnp.linalg.solve`
calls the second. `jaxormick` relaxes the solution from the residual and never relaxes
the solver.

Relaxing the solver's iterates would also be valid, as for any loop of fixed length. The
bounds would be loose, though, because each step relaxes the previous iterate and the
errors add up. The [control flow
guide](../guide/control_flow.md#loops-with-a-fixed-length) shows this for Newton's
method on a square root, where five relaxed steps bound \(\sqrt{x}\) on \([1, 4]\) by
about \([0.26, 8.6]\). The residual describes the solution itself, so a relaxation built
from it does not depend on how many steps the solver takes.

The method has three parts, and the rest of this page goes through them.

1. A box \(X\) is found that contains exactly one solution for every \(p\) in the box of
   parameters. This box is called an enclosure, and it bounds \(x(p)\).
2. The bounds of the enclosure are taken as a first relaxation of \(x(p)\). They are
   constant, so they are convex and concave.
3. A map whose fixed point is the solution is relaxed with McCormick's rules, starting
   from that first relaxation. Each application tightens the relaxation.

The Krawczyk operator is the map in both the first and the third part. The next section
builds it up for one unknown with concrete numbers, and the section after it states the
general form.

## A one-variable example

Take the cubic equation

\[
r(p, x) = x^3 + x - p = 0
\]

for \(p\) in \(P = [2, 2.2]\). The left-hand side increases in \(x\), so there is
exactly one real solution for each \(p\). At \(p = 2\) it is \(x = 1\). A program would
compute it with a few steps of Newton's method, as the [`lax.custom_root`
example](../guide/control_flow.md#implicit-functions) of the guide does.

### Newton's method with a fixed slope

Newton's method takes a point \(\hat x\) close to the solution and moves it to

\[
\hat x - \frac{r(p, \hat x)}{\partial_x r(p, \hat x)},
\]

where \(\partial_x r = 3x^2 + 1\) is the derivative of the residual with respect to
\(x\). The Krawczyk operator keeps a single slope fixed instead. It takes a number
\(y\), close to \(1 / \partial_x r\) at the solution, and uses it for all \(p\) and all
points.

The solution at the middle of \(P\), at \(p = 2.1\), is \(\hat x = 1.0245\). The
derivative there is \(3 \hat x^2 + 1 = 4.149\), so \(y = 1 / 4.149 = 0.2410\).

The operator follows from the mean value theorem in three steps.

1. For a solution \(x^*\) of \(r(p, x) = 0\), the mean value theorem gives a point
   \(\xi\) between \(\hat x\) and \(x^*\) with
   \(r(p, x^*) - r(p, \hat x) = \partial_x r(p, \xi) (x^* - \hat x)\).
2. Since \(r(p, x^*) = 0\), multiplying the equation of step 1 by \(y\) gives
   \(-y \, r(p, \hat x) = y \, \partial_x r(p, \xi) (x^* - \hat x)\) for any number
   \(y\). Substituting this into \(x^* = \hat x + (x^* - \hat x)\) gives the identity
   \[
   x^* = \hat x - y \, r(p, \hat x) + \bigl(1 - y \, \partial_x r(p, \xi)\bigr)(x^* - \hat x).
   \]
3. The point \(\xi\) is unknown. If \(x^*\) lies in an interval \(X\) that contains
   \(\hat x\), then \(\xi\) lies in \(X\) too. Replacing \(\xi\) and \(x^*\) by \(X\)
   and evaluating the right-hand side in [interval
   arithmetic](relaxations.md#interval-bounds) gives an interval that contains \(x^*\).

The interval from step 3 is the Krawczyk operator,

\[
K(p, X) = \hat x - y \, r(p, \hat x) + \bigl(1 - y \, \partial_x r(p, X)\bigr)(X - \hat x).
\]

The coefficient \(1 - y \, \partial_x r(p, X)\) is close to zero when \(X\) is narrow,
since \(y\) is close to \(1 / \partial_x r\). The last term is then much narrower than
\(X\). Once \(X\) contains \(\hat x - y \, r(p, \hat x)\) for all \(p\) with some
margin, the image \(K(p, X)\) lies inside \(X\).

### An enclosure in two steps

The parameters vary over \(P\), so \(r(p, \hat x) = \hat x^3 + \hat x - p\) is the
interval \(2.1 - [2, 2.2] = [-0.1, 0.1]\).

`jaxormick` searches for a box \(X\) that \(K\) maps strictly into itself, as a later
section describes in general. The search starts from the point \(\hat x\) and widens it
by about \(3 \cdot 10^{-8}\) on each side. On a box this narrow, the last term of \(K\)
is negligible. The operator gives

\[
K(P, X) \approx 1.0245 - 0.2410 \cdot [-0.1, 0.1] = [1.0004, 1.0486].
\]

This image is wider than \(X\). The width comes from the parameters, since \(x(p)\)
changes by about \(0.05\) over \(P\). The image is not inside \(X\), so it becomes the
next candidate.

The second step widens the candidate by 10%, to \(X = [0.9980, 1.0511]\). Over this box,
\(\partial_x r = 3 X^2 + 1\) lies in \([3.988, 4.314]\), so the coefficient
\(1 - 0.2410 \cdot [3.988, 4.314]\) is \([-0.040, 0.039]\). The last term of \(K\) is at
most \(0.040 \cdot 0.0265\), about \(0.001\), and

\[
K(P, X) = [0.9994, 1.0497].
\]

This image lies strictly inside \(X = [0.9980, 1.0511]\). The theorem of the next
section then says that \(X\) contains exactly one solution for every \(p\) in \(P\), and
that this solution lies in the image. The enclosure is \([0.9994, 1.0497]\). The exact
range of \(x(p)\) over \(P\) is \([1, 1.0482]\).

### Relaxations from the fixed point

The identity of step 2 holds for the solution at every \(p\) in \(P\),

\[
x(p) = \hat x - y \, r(p, \hat x) + \bigl(1 - y \, \partial_x r(p, \xi)\bigr)(x(p) - \hat x)
\]

with \(\xi\) in the enclosure. The right-hand side is a function of \(p\) and of
\(x(p)\) itself. McCormick's rules relax it if the relaxation of \(p\) is known and
\(x(p)\) is replaced by any relaxation of it. The result is a new relaxation of
\(x(p)\).

In the example, the first term \(\hat x - y (\hat x^3 + \hat x - p)\) is affine in
\(p\), with slope \(y = 0.2410\). Its relaxation is exact. It is the tangent line of
\(x(p)\) at \(p = 2.1\), since \(y\) is the derivative of \(x(p)\) there. The second
term is the product of a coefficient of about \(\pm 0.04\) and of \(x(p) - \hat x\),
which is at most about \(0.025\) in size. It widens the relaxation by about \(0.001\) on
each side of the tangent line.

Starting from the constant relaxations \(x^{cv} = 0.9994\) and \(x^{cc} = 1.0497\),
`jaxormick` applies this step 5 times and keeps the tighter of the old and new
relaxations after each. The table lists the result in `float64` at three values of
\(p\).

| \(p\) | \(x(p)\) | Convex | Concave |
| --- | --- | --- | --- |
| \(2.0\) | \(1.0000\) | \(0.9995\) | \(1.0014\) |
| \(2.1\) | \(1.0245\) | \(1.0236\) | \(1.0255\) |
| \(2.2\) | \(1.0482\) | \(1.0477\) | \(1.0496\) |

The bounds are \([0.9995, 1.0496]\). The relaxations are a band of width about \(0.002\)
around the tangent line. The bounds are slightly narrower than the enclosure, since each
step also intersects the intervals.

## The Krawczyk operator

The general case has a vector of parameters \(p\) in a box \(P\) and a vector of \(n\)
unknowns \(x\). The residual \(r(p, x)\) has \(n\) components, and
\(\partial_x r(p, x)\) is its \(n \times n\) Jacobian with respect to \(x\). For a box
\(X\), \(\partial_x r(p, X)\) is the interval matrix that contains this Jacobian at
every point of \(X\).

The scalar \(y\) of the example becomes an invertible \(n \times n\) matrix \(Y\), and
\(I\) is the identity matrix. For a point \(\hat x\) in \(X\), the Krawczyk operator is

\[
K(p, X) = \hat x - Y r(p, \hat x) + \bigl(I - Y \partial_x r(p, X)\bigr)(X - \hat x).
\]

If \(x(p)\) lies in \(X\), then it lies in \(K(p, X)\) too, by the mean value theorem.
The derivation is that of the example, with each row of the Jacobian taken somewhere on
the segment from \(\hat x\) to \(x(p)\). [Krawczyk
(1969)](https://doi.org/10.1007/BF02234767) showed more for a fixed \(p\). If
\(K(p, X)\) lies strictly inside \(X\), then \(X\) contains exactly one solution.
[Neumaier (1990)](https://doi.org/10.1017/CBO9780511526473) extended this to all \(p\)
in a box \(P\), with the operator evaluated in interval arithmetic.

`jaxormick` takes \(\hat x\) as the middle of \(X\) and \(Y\) as the inverse Jacobian at
the middle of the box. Then \(I - Y \partial_x r\) is small on small boxes, and the
operator contracts. On wide boxes it may not.

## An enclosure by ε-inflation

The first task is a box \(X\) that the operator maps strictly into itself. `jaxormick`
searches for one by ε-inflation, a standard method of verified computing that [Rump
(2010)](https://doi.org/10.1017/S096249291000005X) describes in his survey.

1. The search starts from the solution at the middle of \(P\). The solver of
   `lax.custom_root` finds this solution, and two Newton steps with the matrix \(Y\)
   refine it. For a linear system, the Newton steps alone solve it.
2. Each step widens the candidate box. Its half-width is multiplied by 1.1 and increased
   by \(\sqrt{\varepsilon} \, (1 + \lvert m \rvert)\), where \(\varepsilon\) is the
   machine epsilon of the dtype and \(m\) the middle of the box. The operator is then
   applied to the widened box.
3. Once the image lies strictly inside the widened box, the image is the enclosure.
   Otherwise the image is the next candidate.
4. After 10 steps without success, the bounds and relaxations of the solution are
   infinite. They are valid then, but useless. Branch and bound splits the box, and on
   smaller boxes the test succeeds.

The example above needed two of these steps.

## Relaxed Krawczyk steps

For every \(p\) in \(P\), the solution is a fixed point of the operator,
\(x(p) = K(p, x(p))\), with the Jacobian taken anywhere in the enclosure \(X\). [Stuber,
Scott and Barton (2015)](https://doi.org/10.1080/10556788.2014.924514) turned this into
relaxations. Relax \(K\) with McCormick's rules, using a relaxation of \(x(p)\) as one
of its operands. The result is again a relaxation of \(x(p)\), and the maximum of the
two convex sides is a tighter one.

`jaxormick` starts from the bounds of the enclosure, which are constant relaxations. It
then takes 5 relaxed steps and intersects the relaxations after each. The middle of the
enclosure and the matrix \(Y\) depend only on the box, so the relaxations are convex and
concave in the reference point.

## Implementation in `jaxormick`

### Residuals in the jaxpr

From JAX 0.11 on, `lax.custom_root` appears in the jaxpr as `call_hi_primitive`. Its
parameter `_prim` is a `CustomRoot` object that contains the residual and the solver.
Before JAX 0.11, `lax.custom_root` traces to a `custom_jvp_call` whose primal jaxpr runs
only the solver. The residual then appears only inside the derivative rule, so
`jaxormick` raises `UnsupportedPrimitiveError` for such a root.

`lax.custom_linear_solve` contains the jaxpr of the matrix-vector product on every JAX
version. Its residual is \(A(p) x - b(p)\), and the operands of `matvec` and the
right-hand side are the parameters. The constants of its solvers, such as the LU factors
of `jnp.linalg.solve`, are never read. The interpreter therefore replaces the outputs of
an equation without a rule by zeros when they are used only as constants of such
solvers.

### Enclosures from a larger box

An enclosure verified for every \(p\) in \(P\) is also an enclosure for every \(p\) in a
smaller box inside \(P\). Branch and bound only splits boxes, so the enclosure of a box
can be passed on to its halves. With `enclosures=True`, the functions returned by
`relax` and `interval` take a flat array of enclosures after the box and return the
enclosures that they verified. A finite enclosure replaces the ε-inflation, and the
relaxed steps start from it. `root_enclosure_size` returns the length of the array. Only
roots of `lax.custom_root` at the top level of the function take part, and
`tighten_bounds` cannot be combined with enclosures.

### Cost and limits

The residual's Jacobian is a dense matrix, so a solution with \(n\) entries costs an
\(n \times n\) Jacobian of relaxations in each step. The ε-inflation and the relaxed
steps run in `lax.fori_loop`, so the relaxation of the operator is traced twice, once
for each loop body.

- On wide boxes, the operator may not contract, and the bounds and relaxations of the
  solution are infinite.
- A singular Jacobian at the middle of the box has no inverse. The entries of \(Y\) that
  are not finite are set to zero, and with \(Y = 0\) the operator returns \(X\) itself,
  which never lies strictly inside \(X\). The bounds are then infinite as well.
- The relaxation describes the solution in the enclosure around the root that the solver
  returns at the middle of the box. If the equation has several solutions, the solver
  has to return the one that the program means.
- `lax.custom_root` with `has_aux=True` is not supported, and neither is
  `lax.custom_root` before JAX 0.11.

## Further reading

- [Neumaier (1990)](https://doi.org/10.1017/CBO9780511526473) treats the Krawczyk
  operator and other interval Newton methods for systems of equations in depth.
- [Moore, Kearfott and Cloud (2009)](https://doi.org/10.1137/1.9780898717716) introduce
  interval arithmetic and interval Newton methods at textbook level.
- [Moore (1977)](https://doi.org/10.1137/0714040) proves that a Krawczyk image inside
  \(X\) implies a solution in \(X\).
- [Rump (2010)](https://doi.org/10.1017/S096249291000005X) surveys verification methods
  in floating point, including ε-inflation.
- [Stuber, Scott and Barton (2015)](https://doi.org/10.1080/10556788.2014.924514)
  develop the relaxations of implicit functions that `jaxormick` implements.
- The JAX documentation of
  [`lax.custom_root`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html)
  and
  [`lax.custom_linear_solve`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html)
  describes how a solver and its residual are passed to JAX.
