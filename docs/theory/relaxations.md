# Convex relaxations

## Global optimization and lower bounds

Take a function \(f\) on a box \(X = [x^L, x^U] \subset \mathbb{R}^n\) and the problem

\[
\min_{x \in X} f(x).
\]

A local solver such as gradient descent or Newton's method finds a point where no small
step improves \(f\). When \(f\) is nonconvex, that point can be far from the best one,
and nothing in the local solution says how far. A global solver has to prove that no
point in \(X\) does better, which needs a lower bound on \(f\) over all of \(X\).

Branch and bound finds such a proof by splitting the box. It keeps a list of boxes and
the best value \(f^*\) found so far. For each box \(B\) it computes a lower bound
\(\ell(B) \le \min_{x \in B} f(x)\). When \(\ell(B) > f^*\), no point of \(B\) can beat
\(f^*\), and the solver discards the box. Otherwise it splits the box in two and puts
both halves back on the list. The search ends when every remaining box has
\(\ell(B) \ge f^* - \varepsilon\), and \(f^*\) is then within \(\varepsilon\) of the
global minimum.

The speed of the search depends on the quality of the lower bounds. With loose bounds
the solver discards few boxes, and the list grows exponentially with the dimension.
`jaxormick` computes such lower bounds for JAX functions.

## Interval bounds

The cheapest bounds come from interval arithmetic ([Moore, Kearfott and Cloud,
2009](https://doi.org/10.1137/1.9780898717716)). Each intermediate value gets an
interval that contains it for every \(x \in B\). Sums add the endpoints, products take
the smallest and largest of the four endpoint products, and monotone functions map the
endpoints.

Interval bounds lose information whenever a variable occurs twice. For
\(x \in [-1, 1]\), the expression \(x - x\) is always zero, but interval arithmetic
computes \([-1, 1] - [-1, 1] = [-2, 2]\), because it treats the two occurrences as
independent. Such losses add up over a long computation.

## Convex and concave relaxations

A convex relaxation, also called a convex underestimator, of \(f\) on \(B\) is a convex
function \(f^{cv}\) with

\[
f^{cv}(x) \le f(x) \quad \text{for all } x \in B.
\]

A concave relaxation \(f^{cc}\) is a concave function with \(f^{cc}(x) \ge f(x)\) on
\(B\). The tightest convex relaxation is the convex envelope, the largest convex
function below \(f\).

Minimizing \(f^{cv}\) over \(B\) is a convex problem, so a local solver finds its global
minimum, and that minimum is a lower bound on \(f\). Relaxations remember that \(x - x\)
is zero, since they are functions of \(x\) rather than ranges. They also tighten faster
as the boxes shrink. For McCormick relaxations, the gap \(f(x) - f^{cv}(x)\) at a point
shrinks with the square of the box width ([Bompadre and Mitsos,
2012](https://doi.org/10.1007/s10898-011-9685-2)), while the gap of an interval bound
shrinks linearly. Branch and bound then needs fewer splits near the minimum.

Constraints \(g(x) \le 0\) relax the same way. The set where \(g^{cv}(x) \le 0\) is
convex and contains the feasible set, so the optimum of the relaxed problem is still a
lower bound.

## Affine relaxations

A convex function lies above each of its tangent planes. If \(s\) is a subgradient of
\(f^{cv}\) at a reference point \(\bar{x}\), then

\[
f(x) \ge f^{cv}(x) \ge f^{cv}(\bar{x}) + s^\top (x - \bar{x}) \quad \text{for all } x \in B.
\]

The right-hand side is an affine underestimator. A few of them, taken at different
reference points, form a linear program whose optimum is a lower bound on \(f\). Global
solvers prefer linear programs, since the simplex method solves them reliably and can
start from the solution of the parent box after a split. The same holds on the concave
side, where a supergradient defines affine overestimators and linear cuts for
constraints \(g(x) \ge 0\).

The bound from one affine underestimator is the minimum of a linear function over a box,
which has a closed form:

\[
\min_{x \in B} \left( f^{cv}(\bar{x}) + s^\top (x - \bar{x}) \right)
= f^{cv}(\bar{x}) + \sum_i \min\left(s_i (x^L_i - \bar{x}_i),\ s_i (x^U_i - \bar{x}_i)\right).
\]

This bound can be tighter than the interval bound, and it costs one gradient on top of
the relaxation.

The next page shows how McCormick's rules build \(f^{cv}\) and \(f^{cc}\) for a function
given as code.
