# McCormick relaxations

The [previous page](relaxations.md) defined convex and concave relaxations. This page
explains how McCormick's method builds them for a function given as code, one operation
at a time, and how `jaxormick` tightens the result.

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

### Four numbers per factor

McCormick's method keeps four numbers for each factor \(v\) over the box \(X\):

- an interval \([v^L, v^U]\) that contains \(v\) over the whole box,
- the value \(v^{cv}(x)\) of a convex relaxation at a reference point \(x\), and
- the value \(v^{cc}(x)\) of a concave relaxation at the same point.

Each variable starts with an exact relaxation, \(x_i^{cv} = x_i^{cc} = x_i\). A rule for
each operation then computes the four numbers of its result from those of its operands.
The intervals follow from [interval arithmetic](relaxations.md#interval-bounds). The
sections below give the rules for the convex and concave sides.

## Sums and constant factors

Sums of convex functions are convex, so relaxations add:

\[
(u + w)^{cv} = u^{cv} + w^{cv}, \qquad (u + w)^{cc} = u^{cc} + w^{cc}.
\]

Multiplying by a constant \(c \ge 0\) scales both sides. A negative constant swaps them,
so \((-u)^{cv} = -u^{cc}\). Subtraction combines the two, with
\((u - w)^{cv} = u^{cv} - w^{cc}\).

## Univariate functions

### Convex and concave relaxations of \(F\)

For \(F(u)\) with a univariate \(F\), the rule first needs relaxations of \(F\) itself
on the interval \([u^L, u^U]\). For \(F(u) = e^u\) on \([-1, 1]\), \(e^u\) is convex, so
it is its own convex relaxation. The tightest concave relaxation is the secant through
the endpoints \((-1, e^{-1})\) and \((1, e)\).

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
- Other shapes combine these cases. Functions such as \(\sinh u\) are concave-convex,
  \(\tan\) is concave-convex on each branch, and \(\sin\) and \(\cos\) switch
  between convex and concave pieces every half period. [Mitsos, Chachuat and Barton
  (2009)](https://doi.org/10.1137/080717341) give the envelopes of these functions.

### The composition rule

The operand \(u\) is itself a function of \(x\) with relaxations \(u^{cv}\) and
\(u^{cc}\). Take a convex underestimator \(F^{cv}\) of \(F\) on \([u^L, u^U]\) and a
point \(z^{min}\) where it is smallest. McCormick's composition rule is

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

For example, take \(F(u) = u^2\), whose minimum is at \(z^{min} = 0\), and an operand
with \(u^{cv} = 0.3\) and \(u^{cc} = 0.7\) at the reference point. The value of \(u\)
there lies in \([0.3, 0.7]\), so \(u^2\) is at least \(0.3^2\). The median of \(0.3\),
\(0.7\) and \(0\) is \(0.3\), and the rule gives \(F(u)^{cv} = 0.09\).

For an increasing convex \(F\) such as \(e^u\), the minimizer is \(z^{min} = u^L\),
which is never larger than \(u^{cv}\). The median is then always \(u^{cv}\), and the
rule reduces to \(F(u)^{cv} = F(u^{cv})\).

## Products

### The product rule

For a product \(u w\), the bounds come from products of nonnegative factors, in three
steps.

1. On the box, \(u - u^L \ge 0\) and \(w - w^L \ge 0\), so \((u - u^L)(w - w^L) \ge 0\).
2. Expanding the product and moving the terms other than \(u w\) to the right-hand side
   gives
   \[
   u w \ge w^L u + u^L w - u^L w^L.
   \]
3. The same argument with \((u^U - u)(w^U - w) \ge 0\) gives a second bound,
   \(u w \ge w^U u + u^U w - u^U w^U\).

The maximum of the two bounds is the convex envelope of \(u w\) on the rectangle
\([u^L, u^U] \times [w^L, w^U]\). When \(u\) and \(w\) are themselves relaxed, each
bound is linear in \(u\) and \(w\). Every term then takes \(u^{cv}\) or \(u^{cc}\)
depending on the sign of its coefficient. The concave side uses the two other corner
inequalities, \((u - u^L)(w^U - w) \ge 0\) and \((u^U - u)(w - w^L) \ge 0\).

### Products of a factor with itself

The product rule treats \(u\) and \(w\) as independent. For \(x \cdot x\) on
\([-1, 1]\), it returns the envelope of \(u w\) on \([-1, 1]^2\), whose value at
\(u = w = 0\) is \(-1\). The univariate rule for \(x^2\) uses that both factors are
equal and returns the exact value \(0\).

`jaxormick` therefore rewrites a product of two powers of the same factor, such as
\(x \cdot x\) or \(x^2 \cdot x\), as a single power before it relaxes the product. It
does the same for a power of a power, such as \((x^2)^2\). The univariate rule returns
the envelope of a power, and no other relaxation of it is tighter. A product of
different factors, such as \(x (1 - x)\), still goes through the product rule.

### Products of many factors

A product of many factors, as `jnp.prod` computes, applies the product rule to pairs. It
multiplies the first half of the factors with the second half, elementwise, and repeats
this until one factor is left. That takes \(\lceil \log_2 n \rceil\) rounds for \(n\)
factors. The maximum of many terms, as `jnp.max` computes, combines pairs the same way
with the rule for \(\max(a, b)\).

## Subgradients

An affine relaxation needs a subgradient of \(f^{cv}\) at the reference point. [Mitsos,
Chachuat and Barton (2009)](https://doi.org/10.1137/080717341) showed that subgradients
propagate through the same rules, by applying the chain rule to the formulas above. At
the median, the subgradient follows whichever argument the median picked. The formulas
are ordinary functions of \(x\), so automatic differentiation computes the same
subgradients, as [relaxations in JAX](jax.md) describes.

## Tighter relaxations

The rules above relax one factor at a time. The three methods in this section give
tighter relaxations, and `jaxormick` implements each of them as an option.

### Intersections of relaxations

Different forms of the same function relax differently, and neither form needs to be
tighter everywhere. The maximum of two convex underestimators of \(f\) is convex and
still underestimates \(f\), and the minimum of two concave overestimators is concave and
overestimates \(f\). The intersection of two intervals that contain the range of \(f\)
contains it too. Combining relaxations this way therefore gives a relaxation that is at
least as tight as each of them. A subgradient of the function that attains the maximum
at the reference point is a subgradient of the maximum.

### αBB relaxations

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

For example, \(f(x) = -x^2\) on \([-1, 1]\) has the Hessian \(-2\). The shift
\(\alpha = 1\) adds \(2\) to it, and \(f^{cv}(x) = -x^2 + (x + 1)(x - 1) = -1\), which
is the convex envelope of \(-x^2\) on this interval.

`jaxormick` bounds the Hessian over the box by applying `interval` to `jax.hessian(f)`,
and computes shifts \(\alpha_i\) that make every matrix within these bounds positive
semidefinite with the scaled Gershgorin theorem ([Adjiman et al.,
1998](https://doi.org/10.1016/S0098-1354(98)00027-1)). The concave overestimator applies
the same construction to \(-f\).

`jaxormick` intersects the αBB relaxation with the McCormick relaxation, so the result
is at least as tight as either. Neither is tighter in general. The quadratic term
shrinks with the box. αBB helps most for smooth functions whose factors relax poorly one
at a time, such as indefinite quadratic forms. Where the Hessian bounds are infinite,
the αBB sides are infinite too, and the McCormick sides remain.

### Subgradient-tightened bounds

The interval of a factor can be much wider than its range. For \(v = x - x^2\) on
\([0, 2]\), the rules give \([-4, 2]\), while the range is \([-2, 0.25]\). [Najman and
Mitsos (2019)](https://doi.org/10.1007/s10898-019-00791-0) narrow such intervals with
the relaxations themselves. A convex underestimator lies above each of its tangent
planes, so the lowest point of such a plane over the box is a lower bound of the factor.
The concave side gives an upper bound in the same way. The narrower interval then
tightens the relaxations of every factor that depends on \(v\).

The bounds must not depend on the reference point, or the relaxations would stop being
convex. `jaxormick` therefore takes the subgradients at the middle of the box, from a
second relaxation at that point. In the example, the convex side of \(v\) is \(-x\) and
the concave side is \(v\) itself. Their tangent planes at \(\bar x = 1\) give
\(-2 \le v \le 1\).

## Further work

McCormick relaxations extend in several directions that `jaxormick` does not implement
yet. [Scott, Stuber and Barton (2011)](https://doi.org/10.1007/s10898-011-9664-7)
generalized them, and their generalized relaxations are the basis of relaxations of
solutions of equations, the topic of the [next page](implicit.md), and of differential
equations. [Tsoukalas and Mitsos
(2014)](https://doi.org/10.1007/s10898-014-0176-0) gave a composition rule for functions
of several arguments, which tightens products and divisions.

## Further reading

- [Tawarmalani and Sahinidis (2002)](https://doi.org/10.1007/978-1-4757-3532-1) treat
  factorable relaxations, convex envelopes and their use in branch and bound at book
  length.
- [Floudas (2000)](https://doi.org/10.1007/978-1-4757-4949-6) describes αBB and other
  underestimators for deterministic global optimization.
- [Mitsos, Chachuat and Barton (2009)](https://doi.org/10.1137/080717341) state
  McCormick's rules in the form with the median that this page uses, together with their
  subgradients.
- [Wilhelm and Stuber (2022)](https://doi.org/10.1080/10556788.2020.1786566) describe
  EAGO, a global optimizer in Julia that computes McCormick relaxations.
