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
- When \(F\) is concave left of an inflection point and convex right of it, as
  \(\sinh u\) and \(\arcsin u\) are, \(-F\) is convex-concave. The envelopes of \(F\)
  are the negated envelopes of \(-F\).
- When \(F\) is defined on an open interval, as \(\operatorname{artanh} u\) is on
  \((-1, 1)\), the composition rule applies to intervals inside the domain. An interval
  that reaches the edge only gets the bounds of \(F\) over the part inside the domain,
  which are infinite where \(F\) is.
- \(\tan\) has a branch \((k\pi - \pi/2, k\pi + \pi/2)\) for every integer \(k\), and
  is concave-convex on each. Since \(\tan(u - k\pi) = \tan u\), shifting the interval
  by \(k\pi\) moves it to the branch around zero. An interval that contains a pole gets
  infinite bounds.
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
equal and returns the exact value \(0\).

`jaxormick` therefore rewrites a product of two powers of the same factor, such as
\(x \cdot x\) or \(x^2 \cdot x\), as a single power before it relaxes the product. It
does the same for a power of a power, such as \((x^2)^2\). The univariate rule returns
the envelope of a power, and no other relaxation of it is tighter. A product of
different factors, such as \(x (1 - x)\), still goes through the product rule.

A product of many factors, as `jnp.prod` computes, applies the product rule to pairs.
It multiplies the first half of the factors with the second half, elementwise, and
repeats this until one factor is left. That takes \(\lceil \log_2 n \rceil\) rounds for
\(n\) factors. The maximum of many terms, as `jnp.max` computes, combines pairs the
same way with the rule for \(\max(a, b)\).

## Intersections of relaxations

Different forms of the same function relax differently, and neither form needs to be
tighter everywhere. The maximum of two convex underestimators of \(f\) is convex and
still underestimates \(f\), and the minimum of two concave overestimators is concave and
overestimates \(f\). The intersection of two intervals that contain the range of \(f\)
contains it too. Combining relaxations this way therefore gives a relaxation that is at
least as tight as each of them. A subgradient of the function that attains the maximum
at the reference point is a subgradient of the maximum.

## αBB relaxations

McCormick relaxations follow the factorable form, one factor at a time. A smooth
function with many factors can end up with loose sides, even when the function itself is
close to convex. [Adjiman et al. (1998)](https://doi.org/10.1016/S0098-1354(98)00027-1)
relax a twice differentiable function as a whole instead. Their convex underestimator
adds a quadratic to \(f\).

\[
f^{cv}(x) = f(x) + \sum_i \alpha_i (x_i - x_i^L)(x_i - x_i^U)
\]

Each term of the sum is nonpositive on the box, so \(f^{cv} \le f\). The term adds
\(2 \alpha_i\) to the diagonal of the Hessian. \(f^{cv}\) is convex when the shifted
Hessian is positive semidefinite at every point of the box.

`jaxormick` bounds the Hessian over the box by applying `interval` to `jax.hessian(f)`.
The result is an interval matrix \([H^L, H^U]\) that contains the Hessian at every point
of the box. The scaled Gershgorin theorem gives shifts that make every matrix in it
positive semidefinite. With the widths \(d_i = x_i^U - x_i^L\), the shift of variable
\(i\) is

\[
\alpha_i = \max\Bigl(0,\ -\frac{1}{2}\Bigl(H_{ii}^L - \sum_{j \ne i}
\max\bigl(\lvert H_{ij}^L \rvert, \lvert H_{ij}^U \rvert\bigr)
\frac{d_j}{d_i}\Bigr)\Bigr).
\]

The concave overestimator applies the same construction to \(-f\). If every shift is
zero, the interval Hessian proves \(f\) convex on the box, and the convex side is \(f\)
itself.

`jaxormick` intersects the αBB relaxation with the McCormick relaxation, so the result
is at least as tight as either. Neither is tighter in general. The quadratic term
shrinks with the box. αBB helps most for smooth functions whose factors relax poorly one
at a time, such as indefinite quadratic forms. Where the Hessian bounds are infinite,
the αBB sides are infinite too, and the McCormick sides remain.

## Subgradient-tightened bounds

The interval of a factor can be much wider than its range. For \(v = x - x^2\) on
\([0, 2]\), the rules give \([-4, 2]\), while the range is \([-2, 0.25]\). [Najman and
Mitsos (2019)](https://doi.org/10.1007/s10898-019-00791-0) narrow such intervals with
the relaxations themselves. A convex underestimator lies above each of its tangent
planes. The lowest point of such a plane over the box is therefore a lower bound of the
factor.

\[
v^L \leftarrow \max\Bigl(v^L,\ v^{cv}(\bar x) + \sum_i \min\bigl(s_i (x_i^L - \bar x_i),
\ s_i (x_i^U - \bar x_i)\bigr)\Bigr)
\]

Here \(s\) is a subgradient of \(v^{cv}\) at a point \(\bar x\) of the box. The concave
side gives an upper bound in the same way. The narrower interval then tightens the
relaxations of every factor that depends on \(v\).

The bounds must not depend on the reference point, or the relaxations would stop being
convex. `jaxormick` therefore takes the subgradients at the middle of the box, from a
second relaxation at that point. In the example, the convex side of \(v\) is \(-x\) and
the concave side is \(v\) itself. Their tangent planes at \(\bar x = 1\) give
\(-2 \le v \le 1\).

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
subgradients, as [relaxations in JAX](jax.md) describes.

## Further work

McCormick relaxations extend in several directions that `jaxormick` does not implement
yet. [Scott, Stuber and Barton (2011)](https://doi.org/10.1007/s10898-011-9664-7)
generalized them to solutions of differential equations. Solutions of equations are
covered on the [next page](implicit.md).
[Tsoukalas and Mitsos (2014)](https://doi.org/10.1007/s10898-014-0176-0) gave a
composition rule for functions of several arguments, which tightens products and
divisions.
