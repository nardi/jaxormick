# Convex relaxations

A solver that looks for the global minimum of a nonconvex function has to prove that no
point does better than the one it found. This page explains why such a proof needs lower
bounds, and how interval bounds and convex relaxations provide them.

## Global optimization and lower bounds

Take a function \(f\) on a box \(X = [x^L, x^U] \subset \mathbb{R}^n\), where \(x^L\)
and \(x^U\) are the vectors of lower and upper bounds of the variables, and the problem

\[
\min_{x \in X} f(x).
\]

A local solver such as gradient descent or Newton's method finds a point where no small
step improves \(f\). When \(f\) is nonconvex, that point can be far from the best one,
and nothing in the local solution says how far. A global solver has to prove that no
point in \(X\) does better, which needs a lower bound on \(f\) over all of \(X\).

### Branch and bound

Branch and bound finds such a proof by splitting the box. It keeps a list of boxes and
the best value \(f^*\) found so far. Each round takes one box \(B\) from the list.

1. It computes a lower bound \(\ell(B) \le \min_{x \in B} f(x)\).
2. When \(\ell(B) > f^*\), no point of \(B\) can beat \(f^*\), and the solver discards
   the box.
3. Otherwise it splits the box in two and puts both halves back on the list.

The search ends when every remaining box has \(\ell(B) \ge f^* - \varepsilon\), and
\(f^*\) is then within \(\varepsilon\) of the global minimum.

The speed of the search depends on the quality of the lower bounds. With loose bounds
the solver discards few boxes, and the list grows exponentially with the dimension.
`jaxormick` computes such lower bounds for JAX functions. The textbooks of [Horst and
Tuy (1996)](https://doi.org/10.1007/978-3-662-03199-5) and [Floudas
(2000)](https://doi.org/10.1007/978-1-4757-4949-6) describe branch and bound for
continuous problems in detail.

## Interval bounds

The cheapest bounds come from interval arithmetic ([Moore, Kearfott and Cloud,
2009](https://doi.org/10.1137/1.9780898717716)). Each intermediate value is assigned an
interval that contains it for every \(x \in B\). Sums add the endpoints, products take
the smallest and largest of the four endpoint products, and monotone functions map the
endpoints.

Interval bounds lose information whenever a variable occurs twice. For
\(x \in [-1, 1]\), the expression \(x - x\) is always zero, but interval arithmetic
computes \([-1, 1] - [-1, 1] = [-2, 2]\), because it treats the two occurrences as
independent. Such losses add up over a long computation.

The same happens in the example that the rest of this page uses, \(f(x) = x - x^2\) on
\([0, 1]\). Its values lie in \([0, 0.25]\). Interval arithmetic computes
\(x^2 \in [0, 1]\) and then \(f \in [0, 1] - [0, 1] = [-1, 1]\), a lower bound of \(-1\)
instead of \(0\).

## Convex and concave relaxations

A convex relaxation, also called a convex underestimator, of \(f\) on \(B\) is a convex
function \(f^{cv}\) with

\[
f^{cv}(x) \le f(x) \quad \text{for all } x \in B.
\]

A concave relaxation \(f^{cc}\) is a concave function with \(f^{cc}(x) \ge f(x)\) on
\(B\). The tightest convex relaxation is the convex envelope, the largest convex
function below \(f\).

In the example, \(f(x) = x - x^2\) is concave. Its convex envelope on \([0, 1]\) is the
line through its values at the endpoints, which is the constant \(0\). Its tightest
concave relaxation is \(f\) itself. The minimum of the convex relaxation is \(0\), the
exact minimum of \(f\), where interval arithmetic gave \(-1\).

### Advantages over interval bounds

Minimizing \(f^{cv}\) over \(B\) is a convex problem, so a local solver finds its global
minimum, and that minimum is a lower bound on \(f\). [Boyd and Vandenberghe
(2004)](https://doi.org/10.1017/CBO9780511804441) treat convex functions and convex
problems in depth.

The relaxations of \(x - x\) are zero, since relaxations are functions of \(x\) rather
than ranges. They also tighten faster as the boxes shrink. For McCormick relaxations,
the gap \(f(x) - f^{cv}(x)\) at a point shrinks with the square of the box width
([Bompadre and Mitsos, 2012](https://doi.org/10.1007/s10898-011-9685-2)), while the gap
of an interval bound shrinks linearly. Branch and bound then needs fewer splits near the
minimum.

### Relaxations of constraints

Constraints \(g(x) \le 0\) are relaxed the same way. The set where \(g^{cv}(x) \le 0\)
is convex and contains the feasible set, so the optimum of the relaxed problem is still
a lower bound.

## Affine relaxations

A convex function lies above each of its tangent planes. If \(s\) is a subgradient of
\(f^{cv}\) at a reference point \(\bar{x}\), then

\[
f(x) \ge f^{cv}(x) \ge f^{cv}(\bar{x}) + s^\top (x - \bar{x}) \quad \text{for all } x \in B.
\]

The right-hand side is an affine underestimator. A few of them, taken at different
reference points, form a linear program whose optimum is a lower bound on \(f\). Global
solvers often use linear programs, since the simplex method solves them reliably and can
start from the solution of the parent box after a split. [Tawarmalani and Sahinidis
(2002)](https://doi.org/10.1007/978-1-4757-3532-1) describe how global solvers build
such linear relaxations. The same holds on the concave side, where a supergradient
defines affine overestimators and linear cuts for constraints \(g(x) \ge 0\).

### A bound from one plane

The bound from one affine underestimator is the minimum of a linear function over a box,
which has a closed form:

\[
\min_{x \in B} \left( f^{cv}(\bar{x}) + s^\top (x - \bar{x}) \right)
= f^{cv}(\bar{x}) + \sum_i \min\left(s_i (x^L_i - \bar{x}_i),\ s_i (x^U_i - \bar{x}_i)\right).
\]

This bound can be tighter than the interval bound, and it costs one gradient on top of
the relaxation. The [affine relaxations
guide](../guide/affine.md#a-lower-bound-from-one-plane) computes it for \(x^2 - x\) on
\([0, 1]\), where one plane at the middle of the box gives the exact minimum \(-0.25\)
and interval arithmetic gives \(-1\).

The next page shows how McCormick's rules build \(f^{cv}\) and \(f^{cc}\) for a function
given as code.
