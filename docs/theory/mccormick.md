# McCormick relaxations

## Factorable functions

[McCormick (1976)](https://doi.org/10.1007/BF01580665) showed how to relax any function
that is built from sums, products and univariate functions. Such functions are called
factorable. Every function a computer evaluates is factorable, since it runs as a
sequence of elementary operations. Take

\[
f(x_1, x_2) = x_1 e^{x_2} - x_2^2.
\]

It splits into factors, each an elementary operation on earlier ones:

\[
v_1 = e^{x_2}, \qquad v_2 = x_1 v_1, \qquad v_3 = x_2^2, \qquad f = v_2 - v_3.
\]

McCormick's method keeps four numbers for each factor \(v\) over the box \(X\): an
interval \([v^L, v^U]\) that contains \(v\) over the whole box, and the values
\(v^{cv}(x)\) and \(v^{cc}(x)\) of a convex and a concave relaxation at a reference
point \(x\). Each variable starts with an exact relaxation,
\(x_i^{cv} = x_i^{cc} = x_i\). A rule for each operation then computes the four numbers
of its result from those of its operands.

## Sums and constant factors

Sums of convex functions are convex, so relaxations add:

\[
(u + w)^{cv} = u^{cv} + w^{cv}, \qquad (u + w)^{cc} = u^{cc} + w^{cc}.
\]

Multiplying by a constant \(c \ge 0\) scales both sides. A negative constant swaps them,
so \((-u)^{cv} = -u^{cc}\). Subtraction combines the two, with
\((u - w)^{cv} = u^{cv} - w^{cc}\).

## Univariate functions

For \(F(u)\) with a univariate \(F\), take a convex underestimator \(F^{cv}\) of \(F\)
on the interval \([u^L, u^U]\) and a point \(z^{min}\) where it is smallest. McCormick's
composition rule is

\[
F(u)^{cv} = F^{cv}\left(\operatorname{mid}\left(u^{cv}, u^{cc}, z^{min}\right)\right),
\]

where \(\operatorname{mid}\) is the median of its three arguments. The concave side
mirrors it with a concave overestimator \(F^{cc}\) and its maximizer \(z^{max}\).

The median picks the value of \(u\) that makes \(F^{cv}\) smallest among those the
relaxations allow. At the reference point, \(u\) lies somewhere in \([u^{cv}, u^{cc}]\).
\(F^{cv}\) decreases left of \(z^{min}\) and increases right of it. If \(z^{min}\) lies
inside \([u^{cv}, u^{cc}]\), the smallest possible value is \(F^{cv}(z^{min})\). If it
lies right of the range, the closest point is \(u^{cc}\), and if left, \(u^{cv}\). The
result is convex in \(x\) ([McCormick, 1976](https://doi.org/10.1007/BF01580665)).

The univariate relaxations \(F^{cv}\) and \(F^{cc}\) depend on the shape of \(F\) on the
interval.

- When \(F\) is convex, as \(e^u\) or \(u^2\) are, \(F^{cv} = F\) and \(F^{cc}\) is the
  secant through the endpoints \((u^L, F(u^L))\) and \((u^U, F(u^U))\).
- When \(F\) is concave, as \(\log u\) and \(\sqrt{u}\) are, the roles swap.
- When \(F\) is convex left of an inflection point and concave right of it, as
  \(\tanh u\) and \(u^3\) are, the convex envelope follows \(F\) up to a point \(t\) and
  then runs along a line from \((t, F(t))\) to \((u^U, F(u^U))\) that touches \(F\) at
  \(t\). For odd powers \(u^{2k+1}\), \(t\) is a fixed multiple of \(u^L\) ([Liberti and
  Pantelides, 2003](https://doi.org/10.1023/A:1021924706467)). Other functions find
  \(t\) by bisection.
- \(\sin\) and \(\cos\) switch between convex and concave pieces every half period.
  Their envelopes combine the convex-concave case with the minima and maxima inside the
  interval ([Mitsos, Chachuat and Barton, 2009](https://doi.org/10.1137/080717341)).

## Products

For a product \(u w\), the inequality \((u - u^L)(w - w^L) \ge 0\) holds on the box and
rearranges to

\[
u w \ge w^L u + u^L w - u^L w^L.
\]

With \((u^U - u)(w^U - w) \ge 0\) there is a second such bound, and their maximum is the
convex envelope of \(u w\) on the rectangle \([u^L, u^U] \times [w^L, w^U]\). When \(u\)
and \(w\) are themselves relaxed, each bound is linear in \(u\) and \(w\). Every term
then takes \(u^{cv}\) or \(u^{cc}\) depending on the sign of its coefficient. The
concave side uses the two other corner inequalities.

The product rule treats \(u\) and \(w\) as independent. For \(x \cdot x\) on
\([-1, 1]\), it returns the envelope of \(u w\) on \([-1, 1]^2\), whose value at
\(u = w = 0\) is \(-1\). The univariate rule for \(x^2\) uses that both factors are
equal and returns the exact value \(0\). `jaxormick` relaxes `x ** 2` with the
univariate rule, so a square written as a power has tighter bounds than one written as a
product.

## A worked example

Take \(f(x) = x_1 e^{x_2} - x_2^2\) on \(X = [0, 1] \times [-1, 1]\) at the reference
point \(x = (0.5, 0)\).

| Factor | Interval | Convex | Concave |
| --- | --- | --- | --- |
| \(x_1\) | \([0, 1]\) | \(0.5\) | \(0.5\) |
| \(x_2\) | \([-1, 1]\) | \(0\) | \(0\) |
| \(v_1 = e^{x_2}\) | \([0.368, 2.718]\) | \(e^0 = 1\) | secant at \(0\), \(1.543\) |
| \(v_2 = x_1 v_1\) | \([0, 2.718]\) | \(0.184\) | \(1.359\) |
| \(v_3 = x_2^2\) | \([0, 1]\) | \(0\) | secant at \(0\), \(1\) |
| \(f = v_2 - v_3\) | \([-1, 2.718]\) | \(0.184 - 1 = -0.816\) | \(1.359 - 0 = 1.359\) |

The function value at the reference point is \(0.5\), and \(-0.816 \le 0.5 \le 1.359\).
The convex side of \(v_2\) takes the larger of the two product bounds,
\(w^L u^{cv} + 0 = 0.368 \cdot 0.5\) and
\(w^U u^{cv} + u^U w^{cv} - u^U w^U = 2.718 \cdot 0.5 + 1 - 2.718\).

## Subgradients

An affine relaxation needs a subgradient of \(f^{cv}\) at the reference point. [Mitsos,
Chachuat and Barton (2009)](https://doi.org/10.1137/080717341) showed that subgradients
propagate through the same rules, by applying the chain rule to the formulas above. At
the median, the subgradient follows whichever argument the median picked. The formulas
are ordinary functions of \(x\), so automatic differentiation computes the same
subgradients, as the [next page](jax.md) describes.

## Further work

McCormick relaxations extend in several directions that `jaxormick` does not implement
yet. [Scott, Stuber and Barton (2011)](https://doi.org/10.1007/s10898-011-9664-7)
generalized them to solutions of implicit equations and differential equations.
[Tsoukalas and Mitsos (2014)](https://doi.org/10.1007/s10898-014-0176-0) gave a
composition rule for functions of several arguments, which tightens products and
divisions. [Najman and Mitsos (2019)](https://doi.org/10.1007/s10898-019-00791-0)
tightened the intervals of intermediate factors with the affine relaxations themselves.
