# Implicit functions

Some values have no formula, only an equation that defines them. The solution \(x(p)\)
of \(r(p, x) = 0\) depends on the parameters \(p\), which in turn depend on the
variables of the box. JAX marks such values with `lax.custom_root` and
`lax.custom_linear_solve`. Both keep the residual \(r\) next to the solver, and
`jnp.linalg.solve` builds on the second. `jaxormick` relaxes the solution from the
residual and never relaxes the solver.

Relaxing the solver's iterates would also be valid, as for any loop of fixed length. The
bounds would be loose, though, because each step relaxes the previous iterate and the
errors add up. The residual describes the solution itself.

## The Krawczyk operator

Let \(P\) be the box of the parameters and \(X\) a box that may contain the solution.
Take a point \(\hat x\) in \(X\) and an invertible matrix \(Y\). The Krawczyk operator
is

\[
K(p, X) = \hat x - Y r(p, \hat x) + \bigl(I - Y \partial_x r(p, X)\bigr)(X - \hat x).
\]

If \(x(p)\) lies in \(X\), the mean value theorem puts it in \(K(p, X)\) too, with the
Jacobian somewhere on the segment from \(\hat x\) to \(x(p)\). [Krawczyk
(1969)](https://doi.org/10.1007/BF02234767) showed more for a fixed \(p\). If
\(K(p, X)\) lies strictly inside \(X\), then \(X\) contains exactly one solution.
[Neumaier (1990)](https://doi.org/10.1017/CBO9780511526473) extended this to all \(p\)
in a box \(P\), with the operator evaluated in interval arithmetic.

`jaxormick` takes \(\hat x\) as the middle of \(X\) and \(Y\) as the inverse Jacobian at
the middle of the box. Then \(I - Y \partial_x r\) is small on small boxes, and the
operator contracts. On wide boxes it may not.

## An enclosure by ε-inflation

The first task is a box \(X\) that the operator maps strictly into itself. `jaxormick`
finds the solution at the middle of \(P\) and starts from that point. The solver of
`lax.custom_root` finds this solution, and two Newton steps refine it. For a linear
system, the Newton steps alone solve it.

Each step widens the candidate box by 10% and a margin of the square root of the machine
epsilon. It then applies the operator. Once the image lies strictly inside the widened
box, the image is the enclosure. Otherwise the image is the next candidate. After 10
steps without success, the bounds and relaxations of the solution are infinite. They are
valid then, but useless. Branch and bound splits the box, and on smaller boxes the test
succeeds.

## Relaxed Krawczyk steps

For every \(p\) in \(P\), the solution is a fixed point of the operator,
\(x(p) = K(p, x(p))\), with the Jacobian taken anywhere in the enclosure \(X\). [Stuber,
Scott and Barton (2015)](https://doi.org/10.1080/10556788.2014.924514) turned this into
relaxations. Relax \(K\) with McCormick's rules, using a relaxation of \(x(p)\) as one
of its operands. The result is again a relaxation of \(x(p)\), and the maximum of the
two convex sides is a tighter one.

`jaxormick` starts from the bounds of the enclosure, which are constant relaxations. It
then takes 5 relaxed steps and intersects the relaxations after each. The middle of the
enclosure and the matrix \(Y\) depend only on the box, so the relaxations stay convex
and concave in the reference point.

## How JAX exposes the residual

From JAX 0.11 on, `lax.custom_root` appears in the jaxpr as `call_hi_primitive`. Its
parameter `_prim` holds a `CustomRoot` object with the residual and the solver. Before
JAX 0.11, `lax.custom_root` traces to a `custom_jvp_call` whose primal jaxpr runs only
the solver. The residual then appears only inside the derivative rule, so `jaxormick`
raises `UnsupportedPrimitiveError` for such a root.

`lax.custom_linear_solve` holds the jaxpr of the matrix-vector product on every JAX
version. Its residual is \(A(p) x - b(p)\), and the operands of `matvec` and the
right-hand side are the parameters. The constants of its solvers, such as the LU factors
of `jnp.linalg.solve`, are never read. The interpreter therefore gives zeros to the
outputs of an equation without a rule, as long as only such solver constants use them.

The residual's Jacobian is a dense matrix, so a solution with \(n\) entries costs an
\(n \times n\) Jacobian of relaxations in each step. The enclosure and the relaxed steps
run in `lax.fori_loop`, so the operator is traced only twice.
