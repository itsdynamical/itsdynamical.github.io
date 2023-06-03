---
layout: post
title:  "Variational Optimization, and How It Simplifies Manifold Optimization"
subtitle: "(Part I: the continuous side of the story)"
author:
- Lingkai Kong
- Molei Tao
# author_homepage: https://mtao8.math.gatech.edu
# author: Lingkai Kong and Molei Tao
date:   2023-06-01
categories: article
comments_id: 1
---

## TL; DR
Gradient Descent (GD) is one of the most popular optimization algorithms for machine learning, and momentum is often used to accelerate its convergence. In this blog, we will start with a variational formulation of momentum GD, explore its rich connection to mechanics, and demonstrate how it allows natural generalizations of momentum GD to optimizing functions defined on manifolds.

Two specific (classes) of manifolds will be discussed, namely Lie groups [[Tao & Ohsawa, 2020]](https://arxiv.org/pdf/2001.10006.pdf) and the Stiefel manifold [[Kong, Wang & Tao, 2023]](https://arxiv.org/pdf/2205.14173.pdf). Such optimizations are beyond being mathematically interesting, and there are numerous applications  helpful for machine learning practice. For example, it can be used to [improve the performance of Transformer model](#orthogonality-boosts-the-performance-of-transformer-models) and [approximate Wasserstein distances in high dimension](#projection-robust-wasserstein-prw-distance).

Codes for both general optimizers and specific applications can be found [here](https://github.com/konglk1203/VariationalStiefelOptimizer). 

## Gradient Descent with Momentum: A Variational Perspective
Let first consider a smooth optimization problem in Euclidean (i.e. flat) space, 

$$\min_{x\in\mathbb{R}^d} f(x).$$ 

### Gradient Descent
Arguably the most common optimizer in machine learning, gradient descent, uses iteration 

$$x_{k+1}=x_k-h\nabla f(x_k) \tag{1}$$

where $h$ is called the learning rate/step size. It can be understood as a forward Euler discretization of gradient flow ODE in continuous time:

$$\dot{x}=-\nabla f(x)\tag{2}$$

One reknown way to accelerate the convergence of gradent descent is to introduce 'momentum'. We will explain why 'momentum' in machine learning really corresponds to momentum in physics (mechanics).

### Gradient Descent with Momentum: How to View It as a Discretization

Consider for instance Nesterov’s Accelerated Gradient for convex functions (NAG-C, 'C' means it has acceleration for convex $f$), which is a popular momentum GD optimizer. The algorithm is

$$\begin{cases}
x_k&=y_{k-1}-s\nabla f(y_{k-1})\\
y_k &= x_k+\frac{k-1}{k+2}(x_k-x_{k-1})
\end{cases}
\tag{3}$$

starting from $x_0$ with initial condition $y_0=x_0$. [Su, Boyd and Candes, 2014] provided an insightful perspective of viewing it as a discretization of an ODE. To see that, one can introduce coordinate transformation $p_k=\frac{x_k-x_{k-1}}{\sqrt{s}}$ and step size $h=\sqrt{s}$ and rewrite it in the following form

$$
\begin{cases}
    x_{k+1}=x_k+hp_{k+1}\\
    p_{k+1}=p_k-\frac{3}{k+2}p_k-h\nabla f(y_k)
\end{cases}
$$

where the first equation is from the definition of $p_k$ and the second equation is by substituting $x_k$ in the first equation in Eq. (3) to the second equation in Eq. (3). Note that $\|f(y_k)-f(x_k)\|=o(h)$. And this discrete system is the discretization of the following ODE

$$\begin{cases}
 \dot{x}&= p \\
 \dot{p} &= - \gamma(t) p  -\nabla f(x) 
\end{cases}
\tag{4}$$

Note that $t\approx hk=\sqrt{s}k$. This gives the corresponding $\gamma=\frac{3}{t}$.



This ODE is exactly [Newton's second law](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion), which says the rate of change of momentum (in time) is given by the net force, which sums a frictional force $-\gamma p$ and a conservative force $-\nabla f$. $\gamma$ here thus serves as the friction coefficient, and it introduces energy dissipation, which leads $x(t)$ to converge to a local min of $f$ as $t\rightarrow\infty$.


### Quantification of Momentum-Induced Acceleration
It is a common saying that momentum 'accelerates gradient descent'. Let's take a quick look at what this means quantitatively:

Since we are minimizing the function $f$, we quantify the convergence by the 'error of optimization'. Mathematically, it is the difference between the function value we are trying to optimize and the oracle minimum value, i.e., $f(x_k)-f(x^*)$ for discrete cases and $f(x_t)-f(x^*)$ for continuous cases. 

Assuming the $f$ to be convex and $L$-smooth ($L$-smooth means $\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$ for all $x,y$), we have the following convergence rates:

|          |  without momentum | with momentum|
| -------- | ------- |---------|
| continuous case <br> $f(x_t)-f(x^*)=$ | Eq. 2 <br>$\mathcal{O}\left(\frac{1}{t}\right)$   | Eq. 4 ($\gamma(t)=\frac{3}{t}$)<br>$\mathcal{O}\left(\frac{1}{t^2}\right)$ |
| discrete case <br> $f(x_k)-f(x^*)=$ | Eq. 1 ($h\le 1/L$)<br> $\mathcal{O}\left(\frac{1}{k}\right)$   | Eq. 3 ($s\le 1/L$) <br>$\mathcal{O}\left(\frac{1}{k^2}\right)$ |

This means momentum improves the nonasymptotic error bound from linear to quadratic.

In fact, the quadratic convergence speed that momentum GD gives is optimal when, roughly speaking, we only have access to the gradient of the function [Nesterov, 1983].

Many other celebrated GD with momentum algorithms can be viewed as discretizations of this ODE (Eq. 4). For example, when we choose $\gamma$ to be constant, one discretization gives NAG-SC ('SC' stands for strongly convex) and another gives heavy ball [Polyak, 1964].

So, in summary, one of the many reasons we like momentum GD is it has the optimal convergence rate among gradient-based optimizers, while GD without momentum can have a slower convergence speed.


### Generalizing Gradient Descent to Manifold: Why is Nontrivial?
Until now, we have an optimization algorithm in the Euclidean space, let's try to generalize it to optimize a function defined on a manifold. That is, we want manifold gradient descent. But what is the gradient?

Let's take $\mathsf{SO}(2)$, the manifold of 2*2 special orthogonal matrices, as an example, and consider a toy objective function 

$$f:\mathsf{SO}(2)\rightarrow \mathbb{R}, \text{ given by } \begin{pmatrix}
a&b\\
c&d\\
\end{pmatrix} \mapsto ad-bc
$$

If we collect all element-wise derivatives, we get 

$$\begin{pmatrix}
\frac{\partial f}{\partial a}&\frac{\partial f}{\partial b}\\
\frac{\partial f}{\partial c}&\frac{\partial f}{\partial d}\\
\end{pmatrix}=\begin{pmatrix}
d&-c\\
-b&a\\
\end{pmatrix}$$

However, $f$ is actually the determinant, and all special orthogonal matrices have determinant 1, which means the gradient of $f$ should actually be 0 everywhere. Why the contradiction? That's because $\mathsf{SO}(2)$ is a 1-dim manifold, $a,b,c,d$ are not independent from each other, and one can't simply collect all partial-derivatives.

There are many other difficulties, but for a simple exposition, we will not overload the readers with technicalities. Instead, let's stay on the main line and see how variational principle, which is a more fundamental view than Newtonian mechanics, can help manifold optimization.

### Variational Principle and Lagrangian Mechanics
For preparation, let's first start with Eq.4 without friction ($\gamma=0$), where the ODE becomes

$$\begin{cases}
 \dot{x}&= p \\
 \dot{p} &= -\nabla f(x) 
\end{cases}
\tag{5}$$

This is Newtonian mechanics. $x$ is a function of time $t$, which gives a trajectory in position space. Based on thinking mechanics in terms of trajectories, Italian-French mathematician and astronomer Joseph-Louis Lagrange proposed a deeper view of mechanics in 1788, as follows.

Let's consider all possible trajectories, i.e. mappings each represented by $x: [0,T] \rightarrow \mathbb{R}^d$, and associate with each with something called a Lagrangian, which takes a vector-valued function of time and returns a scalar-valued function of time (for advanced readers, it is a dual of energy). If we choose the Lagrangian to be

$$L(x,\dot{x}, t)=\frac{1}{2}\|\dot{x}(t)\|^2-f(x(t)),$$

and consider an "action" functional $\mathcal{S}$ defined as

$$\mathcal{S}[x]:=\int_0^T L(x, \dot{x}, t)$$

Then the critical point of $\mathcal{S}$, i.e. $x$ such that $\delta \mathcal{S} / \delta x=0$, satisfies the Newtonian dynamics (Eq. 5).

For the sake of length, we'll not detail the precise meaning of $\delta \mathcal{S}$. Roughly speaking, $\delta \mathcal{S} / \delta x=0$, known as the Euler-Lagrange equation, means $\lim_{\epsilon\rightarrow 0} \frac{1}{\epsilon}(\mathcal{S}[x+\epsilon \eta]-\mathcal{S}[x])=0$ for any curve $\eta$ fixed at end points. Variational calculus gives a streamlined way of computing the Euler-Lagrange equation for any $L$. '[Stationary-Action Principle](https://en.wikipedia.org/wiki/Stationary-action_principle)' for example could be a good supplementary reading.



### A Dissipative Instance of the Variational Principle: from Mechanics to Optimization

The classical Lagrangian perspective does not change the fact that an isolated mechanical system (i.e. Newtonian dynamics given by Eq. 5) is conservative. Without friction, the total energy, namely the sum of kinetic energy $\frac{1}{2}\|\dot{x}\|^2$ and potential energy $f(x)$, is a constant. What typically happens is an oscillatory behavior, where the kinetic energy and the potential energy will keep on exchanging values with each other, and $f$ will not be minimized.

To track the root of this oscillatory behavior, which is undesired for optimization, let's talk about Noether's theorem. Often considered as the mother of modern mechanics, German mathematician Emmy Noether proved that one symmetry gives one conservation law. In our case, $L$ is invariant under time translation of $x$, i.e., you get the same action if you shift the time via $x(\cdot) \mapsto x(\cdot+C)$, and this gives energy conservation. We can make total energy no longer a constant by breaking the time-translation symmetry: in a seminal paper [[Wibisono, Wilson & Jordan 16]](https://arxiv.org/pdf/1603.04245.pdf) introduced an artificial time dependence, a simplified version of which multiplies the original Lagragnian by an extra given term $r(t)$, i.e.

$$
L(x, \dot{x}, t) := r(t)\left(\frac{1}{2}\|\dot{x}(t)\|^2 - f(x(t))\right)
$$

If we write down the corresponding Euler-Lagrange equation, we will have the ODE with the extra friction term (Eq. 4). $\gamma(t)$ in Eq. 4 is given by $r'(t)/r(t)$, and if we choose $r(t)$ to be positive and monotonically increasing, $\gamma$ will be a positive function and it stands for the friction parameter. Popular choices of $\gamma$ are constant for strongly convex functions and $\frac{3}{t}$ for convex functions. 

A simple Lyapunov argument can help us prove that the system converges to a local minimum when time goes to infinity. 

Like mentioned earlier, one time discretization of this ODE leads to a popular gradient descent algorithm with momentum (Eq. 3), and in fact, popular approaches like heavy-ball, Nesterov Accelerated Gradient method for Convex functions (NAG-C), Nesterov Accelerated Gradient method for Strongly Convex functions (NAG-SC), can all be obtained from different discretization schemes and choices of $\gamma(t)$.


## Variational Formulation of Optimization Makes Its Generalization to Manifold Easy ... in Theory

We mentioned that generalizing momentum GD (Eq.3) or its continuous time limit (ODE Eq.4) is nontrivial (but leading experts in manifold optimization have made a lot of progress!) The deeper layer of variational formulation, however, provides a big hammer. Geometers like to say it is "intrinsic", meaning it doesn't even care what kind of coordinate system you use to parametrize the space that $x$ lives in, be it Euclidean or a Riemannian manifold. Let's see how that works and what obstacles will be on the way of getting a good algorithm.



Let's first list the main steps of the variational optimization methodology:
1. Define a dissipative Lagrangian and a corresponding variational problem
2. Solve the variational problem to get an ODE, which does the optimization in continuous time
3. Design a numerical discretization of the ODE, so that we get an algorithm that optimizes in discrete time

Step 1 seems easy. For Euclidean optimization, we chose $L = r(t)\left(\frac{1}{2}\|\dot{x}(t)\|^2 - f(x(t))\right)$, i.e., {time discount} * ({kinetic energy} - {potential energy}). If $x(\cdot)$ is instead a trajectory on a Riemmanian manifold $\mathcal{M}$, $\dot{x}$ will be in the tangent space, and we can use the Riemannian metric to generalize the kinetic energy. The Lagrangian simply becomes

$$L = r(t)\left(\frac{1}{2}\|\dot{x}(t)\|_\mathcal{M}^2 - f(x(t))\right)$$

and the variational principle is again

$$ \delta \int_0^T L(x,\dot{x},t) dt = 0 .$$

But the difficulty is hidden under the rug. What does $\delta$ mean? Because $x(t)\in\mathcal{M}$, $\dot{x}(t)\in T_{x(t)}\mathcal{M}$, and the variation is actually with respect to all infinitesimal changes of $x$ that keeps it inside a **curved** function space. This means Step 2 is actually nontrivial.

There are advanced tools from geometric mechanics that solve Step 2. From a pure math point of view, that is actually rather elegant. However, the resulting ODE will not appear to be very explicit to a practioner. What is even worse is, Step 3 (time discretization) is still needed so that an algorithm can be constructed, but so far (as in May 2023) we are not aware of any discretization that leads to an explicit algorithm. Instead the iterations are always implicit, meaning one has to solve at least one system of nonlinear equations per GD iteration. This slows down the computation and make the optimizer not well scalable to high dimensional problems often faced in machine learning.


But there are ways to get around these difficulties. We don't have to get everything by brute force.


## Tactically Solving the Variational Problem, by Leveraging Specific Manifold Structures

Here are two examples, where specific structures of the manifold class can be used to solve the variational problem, and that will lead to beautiful ODE that does the optimization (in continuous time).



### When the Manifold is a Lie Group: the Technique of Left Trivialization

#### General Discussion
Lie group is a manifold that also has a group structure, meaning you have a rule that computes the "product" of any two points on the manifold, which will be another point on the manifold. This "multiplication" operation enriches the geometric structure of the manifold. Previously, we mentioned that a challenge of variational optimization on manifold is, it requires taking a "function derivative" with respect to variation in a **curved** function space; a smart utilization  of the group structure, known as *left trivialization*, could alleviate this difficulty.

We'll explain what is *left trivialization* and how it helps manifold optimization. We will try to remain intuitive, but details and rigor can be found in [[Tao & Ohsawa, 2020]](https://arxiv.org/pdf/2001.10006.pdf).

Let's begin by considering the same old question: what is momentum? An expert would distinguish velocity and momentum and state the velocity lives in the tangent space $T_{x(t)}\mathcal{M}$, while momentum lives in the cotangent space $T_{x(t)}^*\mathcal{M}$. In Euclidean space this would just correspond to $v=\dot{x}$ is the velocity, and $p=M\dot{x}$ is the momentum, where $M$ is mass (or more precisely, an inertia matrix, which gives an isomorphism between $T_{x(t)}\mathcal{M}$ and its dual). Let's not worry about these and just consider constant mass, which allows us to mean velocity when saying "momentum".

Then momentum lives in $T_{x(t)}\mathcal{M}$. The problem is, this is a space that is changing when $x$ moves on $\mathcal{M}$. This not only complicates our variational problem, but in fact is a well known issue for other approaches and one reason (among several) why manifold optimization **with momentum** is hard (see e.g., [[Kong, Wang & Tao, 2023]](https://arxiv.org/pdf/2205.14173.pdf) for a review of smart ideas in the literature that address this issue).

However, when we have a Lie group, $\dot{x}$ is in $T_{x(t)}\mathcal{M}$, but $x^{-1}\dot{x}$ is actually in $T_{e}\mathcal{M}$ where $e$ is the identity element of the group. Note $T_{e}\mathcal{M}$ is a fixed linear space, not moving with $x$, and it is a fantastic thing known as the Lie algebra. The idea of left trivialization is, let's consider a new version of "momentum" to be $x^{-1}\dot{x}$, and then we don't have to worry about the nonlinear space the original momentum lives in, as now things are just like the Euclidean case.

#### Concrete Demo via Special Orthogonal Group SO(n)
That was how [[Tao & Ohsawa, 2020]](https://arxiv.org/pdf/2001.10006.pdf) approached the variational optimization problem for general Lie groups. However, to remain concrete, this blog will only focus on an important case of $\mathcal{M}=\mathsf{SO}(n)$. $\mathsf{SO}(n)$ is called the special orthogonal group, defined as the set of all the orthogonal matrices whose determinant is 1, i.e.,

$$\mathsf{SO}(n):=\{X\in \mathbb{R}^{n\times n}: X^\top X=I, \text{det}(X)=1\}$$

For trajectory on this manifold, the velocity has to live in the moving tangent space. To see what that entails, taking the time derivative of $X^T X=I$ gives $\dot{X}^T X+X^T\dot{X}=0$. This means the tangent space at $X$ is $\{\eta\in \mathbb{R}^{n\times n}: \eta^T X + X^T\eta=0\}$. This still looks a bit complicated, but if we *left trivialize* the velocity by letting $\xi=X^{-1}\eta=X^T\eta$, then $\xi$ simply satisfies

$$
    \xi^T+\xi=0,
$$

meaning it is a skew-symmetric matrix. The space in which this left trivialized velocity lives is a fixed tangent space, $T_e \mathsf{SO}(n)$, known as the Lie algebra $\mathfrak{so}(n)$. Note it no longer depends on $X$!

Now let's rename $X$ to be $g$, simply to remind ourselves that the trajectory $g(t)$ lives on a Lie group. In the $\mathsf{SO}(n)$ case, we found that the "position" variable $g$ and the new "velocity"/"momentum" needed to satisfy two constraints

$$g^\top g=I, \quad\xi^\top+\xi=0.$$

Remarkably, they are independent of each other, making the variational problem (i.e. finding the critical point of the action functional, with respect to trajectory variations that maintain the constraints) easier to solve. More precisely, we can again define a Lagrangian as

$$L:=r(t)\left(\frac{1}{2}\langle \xi, \xi\rangle-f(g)\right),$$

where $\langle \xi_1, \xi_2\rangle:=\text{tr}(\xi_1^\top M \xi_2)$ is an inner product defined using standard matrix operations and $M$ is any constant positive definite matrix. We will use $M=I$ from now on for a simple demonstration.

Using tools from geometric mechanics (details are technical and thus omitted, but the treatment is actually intrinsic), one can show that the variational principle $\delta \int L dt = 0$ is equivalent to the following ODEs
    
$$\begin{cases}
\dot{g}=g\xi\\
\dot{\xi}=-\gamma (t)\xi-\left(\frac{\partial f}{\partial g}^\top g-g^\top \frac{\partial f}{\partial g}\right)
\end{cases}
\tag{7}
$$
    
Note here $\frac{\partial f}{\partial g}$ is simply an $n\times n$ matrix that collects all element-wise Euclidean partial derivative, i.e. the one we previously said to be incorrect. The dynamics automatically corrects everything for manifold, and one can just forget about complications due to curved geometry and pretend that $g$ and $\xi$ are matrices in Euclidean space. The ODE will internally keep the geometry right, meaning that $g(t)^\top g(t)=I, \quad\xi(t)^\top+\xi(t)=0$ for all $t>0$ as long as the initial condition is on the manifold, i.e. $g(0)^\top g(0)=I, \quad\xi(0)^\top+\xi(0)=0$. And $\lim_{t\to\infty}g(t)$ will be a local minimizer of $f$.


### When the Manifold is the Stiefel Manifold: the Technique of Function Lagrange Multiplier 

#### The Problem to Solve
The previous section discussed how to optimize $f(g)$ with respect to orthogonal matrices $g$. Orthogonal matrices are square matrices. A practically important generalization would be: how to optimize $f(X)$, when matrix $X$ satisfies orthonormal constraints but is not necessarily square? Note this is a highly nonconvex constraint, and traditional convex optimization tools don't apply.

The geometric space for this problem is the Stiefel manifold.
A Stiefel manifold $\mathsf{St}(n,m)$ is the set of $n\times m$ matrices ($n\ge m$, i.e. tall) with each column orthogonal to all other columns and normalized, i.e.,

$$\mathsf{St}(n,m):=\{X\in \mathbb{R}^{n\times m}: X^\top X=I_m\}$$

When $n=m$, $\mathsf{St}(n,n)$ is almost the same as $\mathsf{SO}(n)$ (but with a negative branch too). However, in general, $n\ge m$, and we no longer have a group structure, and we need a different way to solve the variational problem. Here is how:

#### Solving the Variational Problem via an Alternative Variational Formulation

Our original variation problem (for optimization) is to find the critical point of the action functional
    
$$
    \mathcal{S}[X]=\int r(t)\left(\frac{1}{2}\|\dot{X}(t)\|^2 - f(X(t))\right) dt$$
    
with respect to all trajectories that satisfy
    
$$X(t)^\top X(t)=I, \quad \dot{X}(t)^\top X(t)+X(t)^\top \dot{X}(t)=0, \quad \forall t,$$
    
similar to the $\mathsf{SO}(n)$ case (but note the order of multiplications matter as $X$ is $n\times m$ and constraints are $m\times m$). Variational derivative in this curved function space is challenging but we don't have left-trivialization to help any more.

So we use a different approach, constrained variational principle: introduce a function Lagrange multiplier $\Lambda(t)$ to enforce the constraint at all $t$, let

$$\hat{L}(X, \dot{X}, \Lambda, t)=r(t)\Big[\frac{1}{2}\| \dot{X}(t)\|^2-f(X(t))\Big]-\frac{1}{2}\text{tr}\left(\Lambda(t)^\top(X(t)^\top X(t)-I)\right)$$
    
and consider $\delta \int_0^T \hat{L} dt = 0$ in the flat, unconstrained function space 
    
$$
    \{ X(t), \Lambda(t) | 0\leq t\leq T, \Lambda(t)\in\mathbb{R}^{m\times m}, X(t)\in\mathbb{R}^{n\times m} \}.
$$
    
This variational problem is easier to solve, but its resulting Euler-Lagrange equation will be a Differential-Algebraic Equation system that contains an ODE that describes how $X$ changes in time based on $X$ and $\Lambda$, and a requirement that $\Lambda$ is such that $X^\top X=I$ is maintained. This is still difficult to handle. 

Fortunately, it is possible eliminate $\Lambda$ using techniques borrowed from an astrophysics paper [[Chen, Li & Tao, 2021]](https://arxiv.org/abs/2103.12767). Technicality aside, this leads to the following ODE that does optimization 

$$
\begin{cases}
    \dot{X}=&Q\\
    \dot{Q}=&-\gamma Q-XQ^\top Q
    -\frac{\partial f}{\partial X}+\frac{1}{2}XX^\top\frac{\partial f}{\partial X}+\frac{1}{2}X\frac{\partial f}{\partial X}^\top X
\end{cases}
$$
    
Here "position" variable $X$ and "momentum" variable $Q$ are again simply $n\times m$ matrices, 
$\frac{\partial f}{\partial X}$ is the element-wise derivative (an $n\times m$ matrix too). Like before, although everything is based on matrices in Euclidean space and user needs not to worry about the manifold or constraints, the dynamics internally keeps everything on the manifold, while optimizing $f$.


## What's Next?
Up to this point, we have exploited variational optimization, and obtained explicit Euclidean ODEs, which optimize the objective functions on manifolds. However, these are not optimizers yet as they are just dynamics in continuous time. To obtain optimization algorithms, we need to numerically discretize the time so that iterative solvers can be obtained.

Doing this will be fun, because the ODEs are constructed such that their solutions stay on the curved manifolds. A naive discretization will lead to numerical solutions that go off the manifold. We need better design. Please see the [Part II](variational-optimization-2.html) of this blog.

---

## 📝 How to Cite Me?

Please cite the following 2 publications
```
@inproceedings{tao2020variational,
  title={Variational optimization on {L}ie groups, with examples of leading (generalized) eigenvalue problems},
  author={Molei Tao and Tomoki Ohsawa},
  booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2020}
}

@inproceedings{kong2023momentum,
  title={Momentum {S}tiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport},
  author={Lingkai Kong and Yuqing Wang and Molei Tao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```
If you'd also like to cite this blog, please add a 3rd citation as follows
```
@misc{tao2023blog1,
  title = {Variational Optimization, and How It Works for Manifolds},
  author={Lingkai Kong and Molei Tao},
  howpublished = {\url{https://itsdynamical.github.io/article/2023/06/01/variational-optimization-1.html}},
  note = {From blog <It's dynamical>}
}
```

## Thank you!
