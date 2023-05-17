---
layout: post
title:  "Variational Optimization, and How It Works for Manifolds (Part I)"
author:
- Lingkai Kong
- Molei Tao
# author_homepage: https://mtao8.math.gatech.edu
# author: Lingkai Kong
date:   2023-05-17
categories: article
comments_id: 1
---

## TL; DR
Gradient Descent (GD) is one of the most popular optimization algorithms for machine learning, and momentum is often used to accelerate its convergence. In this blog, we will start with a variational formulation of momentum GD, explore its rich connection to mechanics, and demonstrate how it allows natural generalizations of momentum GD to optimizing functions defined on manifolds.

Two specific (classes) of manifolds will be discussed, namely Lie groups [[Tao & Ohsawa, 2020]](https://arxiv.org/pdf/2001.10006.pdf) and the Stiefel manifold [[Kong, Wang & Tao, 2023]](https://arxiv.org/pdf/2205.14173.pdf). Such optimizations are beyond being mathematically interesting, and there are numerous applications  helpful for machine learning practice. For example, it can be used to [improve the performance of Transformer model](#orthogonality-boosts-the-performance-of-transformer-models) and [approximate Wasserstein distances in high dimension](#projection-robust-wasserstein-prw-distance).

Codes for both general optimizers and specific applications can be found [here](https://github.com/konglk1203/VariationalStiefelOptimizer). 

## Gradient Descent with Momentum: A variational approach
Optimization algorithms are trying to find the minimum of a function, i.e., $\min_{x} f(x)$. The most widely-used type of optimizers in machine learning is called gradient-based optimizers, it assumes the accessibility of the function $f$ and its gradient $\nabla f$. If you are a researcher in machine learning or applied math, you must be heard of the optimization algorithm gradient descent and momentum gradient descent, which are the most famous gradient-based optimizers.

Although the intuition for (momentum) GD is quite straightforward, a different view of momentum GD via a variational approach is provided here. The reason is this approach is easier to generalize to the manifold using this more fundamental approach.

### Gradient Descent
The commonly used gradient descent has iteration 

$$x_{k+1}=x_k-h\nabla f(x_k)$$

where $h$ is called the learning rate/step size. This iterative algorithm comes from discretizing the gradient flow:

$$\dot{x}=-\nabla f(x)$$

In order to accelerate the convergence, people introduce 'momentum' into gradient descent.
### Gradient Descent with Momentum: Newton's View
The commonly used gradient descent with momentum ([torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)) has the following iteration:

$$
\begin{cases}
\theta_t=\theta_{t-1}-\text{lr}\, b_t \\ 
b_t=\mu b_{t-1}+\nabla g(\theta_{t-1})
\end{cases}
$$

$\text{lr}$ is the learning rate. $\mu$ is a user-defined hyperparameter in $[0,1)$. $\theta$ is the parameter we are trying to optimize and $b$ is the 'momentum' (note that this momentum is different from the later momentum in machanics by a constant factor). We can see if we let $\mu=0$, it decays to GD without momentum.

After performing a rescaling of variable by
$\theta=\frac{\sqrt{\text{lr}}}{h}x$, $b_t=-\frac{1}{\sqrt{\text{lr}}}p_k$, $g(\theta_t)=f(x_k)$, $\mu=1-\gamma h$,
with
$t=k+1$,
the iteration for momentum SGD can be written as

$$\begin{cases}
x_{k+1}=x_k+hp_{k+1}\\
p_{k+1}=(1-\gamma h)p_k-h\nabla f(x_k)
\tag{1}
\end{cases}
$$

You may already find the reason why we perform this change of variable: writing GD with momentum in this form can let it be viewed as a discretization of the following ODE with $\text{mass}=1$

$$\begin{cases}
 \dot{x}&= p/\text{mass} \\
 \dot{p} &= - \gamma(t) p  -\nabla f(x) 
\end{cases}
\tag{2}$$

This ODE comes from [Newton's second law](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion), the time rate of change of momentum equals the force ($\nabla f$ in this case) (2nd equation in Eq.2). $p$ is called momentum. $f$ is the potential energy and $\nabla f$ is the conservative force of the potential energy. $\gamma$ here stands for friction. It introduces energy dissipation and lets the dynamical system converge to a saddle point. WLOG, the mass is set to be 1 in all cases.

More intuitively, this ODE Eq. 5 characterizes a particle $x$ moving under the potential $f$. The force will push the particle 'down the hill' and it will find a local minimum. Following this intuition, we can see that the point will stops at a local minimum and the ODE is indeed optimizing $f$.

### Difficulty in Generalizing it directly to the Manifold
Until now, we have an optimization algorithm in the Euclidean space, let's generalize it to the manifold. But wait, it seems there are some difficulties! For example, what is the gradient?

We take $\mathsf{SO}(2)$ as an example. Consider the function 

$$f:\mathsf{SO}(2)\rightarrow \mathbb{R}, \begin{pmatrix}
a&b\\
c&d\\
\end{pmatrix}\rightarrow ad-bc
$$

If we take the element-wise derivative, we get 

$$\begin{pmatrix}
\frac{\partial f}{\partial a}&\frac{\partial f}{\partial b}\\
\frac{\partial f}{\partial c}&\frac{\partial f}{\partial d}\\
\end{pmatrix}=\begin{pmatrix}
d&-c\\
-b&a\\
\end{pmatrix}$$

However, if you take a closer look at the function $ad-bc$, you may find it is just the determinant of the matrix and is a constant function $1$ on $\mathsf{SO}(2)$.

An expert may realize other difficulties, a typical one is it is hard to do numerical discretization in the manifold case. So, we need a more fundamental view than Newton's mechanics.

### Variational Principle: Lagrange Mechanics
First, let's consider the case without friction ($\gamma=0$) and ODE (2) becomes

$$\begin{cases}
 \dot{x}&= p \\
 \dot{p} &= -\nabla f(x) 
\end{cases}
\tag{3}$$

You may have heard of the big name Lagrange even if you are not an expert in physics. He has an elegant view of the ODE (2). Consider the space of all the smooth parametric curves, i.e., all the map $t\rightarrow \mathbb{R}^d, t\in [0,T]$. This is a large space, but Lagrange defines a function called Lagrangian $L(x,\dot{x}, t)=\frac{1}{2}\|\dot{x}(t)\|^2-f(x(t))$. Consider the functional $\mathcal{S}$ defined as

$$\mathcal{S}[x]:=\int_0^T L(x, \dot{x}, t)$$

Here $\mathcal{S}$ takes in a parametric curve and outputs a real number. Lagrange tells us that
> If a curve $x(t), t\in [0,T]$ is a 'critical curve' of the functional $\mathcal{S}$, then it must be a solution of the ODE (Eq. 3).

Mathematically, the parametric curve $x(t), t\in[0,T]$ is the solution of ODE ($\gamma=0$) is equal to say the for any curve $\eta$ with vanish end points, $\lim_{\epsilon\rightarrow 0} \frac{1}{\epsilon}(\mathcal{S}[x+\epsilon \eta]-\mathcal{S}[x])=0$. This is denoted as

$$\delta \int_0^T L(x,\dot{x},t) dt = 0
\tag{4}
$$


Intuitively, this means if we change the curve a little, the Lagrangian is almost unchanged. We can compare it with the local minimum of a function. The local minimum $y^*$ of a function $g$ satisfies $\lim_{\epsilon\rightarrow 0} g(y+\epsilon v)=0\,\forall v$. And the curve satisfies the ODE (2) is the 'best curve' under the cost of $\mathcal{S}$. This is why it is also called '[stationary-action principle](https://en.wikipedia.org/wiki/Stationary-action_principle)'. This is really amazing. It tells us that all the objects in the world are somehow lazy, they are always trying to find the path that cost the least Lagrangian.

How to solve this variational problem is technical and beyond this blog. But the solution for a variational problem is an ODE given by the [Euler–Lagrange equation](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation). Back to our problem, Lagrange proves that the solution of variational problem Eq. 4 is Newton's ODE Eq. 3.

### Dissipative Variational Principle: from Lagrange Mechanics to Variational Optimization
Lagrange mechanics is elegant, however, it has not introduced the friction $\gamma$ yet. Without friction, the total energy, the sum of kinetic energy $\frac{1}{2}\|\dot{x}\|^2$ and potential energy $f(x)$, is a constant. In other word, the kinetic energy and the potential energy will keep exchanging energy with each other and $x$ will never converge.

This means we need friction $\gamma$ to do optimization. But how do we get it from the variational principle? The beautiful idea in [[Wibisono, Wilson & Jordan 16]](https://arxiv.org/pdf/1603.04245.pdf) introduces an extra term $r(t)$, a monotonely increasing positive function, and gives a time-dependent Lagrangian 

$$
L(x, \dot{x}, t) := r(t)\left(\frac{1}{2}\|\dot{x}(t)\|^2 - f(x(t))\right)
$$

This $r(t)$ term breaks this symmetricity of energy and gives us energy dissipation. If we also solve the same variational problem Eq. 3, we have the ODE with friction (Eq. 2). $\gamma(t)$ in Eq. 2 is given by $r'(t)/r(t)$ is a positive function stands for the friction parameter. Popular choices of $\gamma$ are constant for strongly convex functions and $\frac{3}{t}$ for convex functions. Thus, we have that the system converges to a local minimum when time goes to infinity. By discretizing this ODE, we have the popular (stochastic) momentum gradient descent algorithm (Eq. 1).



## Generalization to Manifold
In the following, we will generalize this beautiful approach first to Lie groups, and then to the Stiefel manifold. But wait, we have seen that generalizing Newton's ODE directly to the manifold case can be hard, why generalizing it through Lagrange's variational principle is easier?

To see that, let's list the main steps of designing momentum GD in Euclidean space using a variational approach:
- Define a dissipative Lagrangian
- Solve the variational problem and get an ODE
- Design a numerical discretization

We can see only a Lagrange function $L$ can characterize the motion of a particle. This means, we just need to have a Lagrangian $L$ and the variation principle gives us the ODE characterizing its motion. The Lagrangian can be easily generalized to the manifold. But where does the formerly mentioned difficulty from curved space go? Why we do not face the difficulty we were facing anymore?

The answer is, it is hidden in the second step, solving the variational problem. Since we consider the 'best curve among all possible curves', the curve must be on the manifold and as a result, the manifold structure is automatically preserved. 

This looks really nice and easy in the view of pure math: in the second step, the manifold structure is preserved by the variational principle. However, there are 2 main difficulties in reality:
- In the second step, we need to solve a variational problem on a curved space. 
- In the third step, the numerical discretization is usually non-trivial and needs to be designed specifically for each manifold.

In the rest of this part of the blog, we focus on the first difficulty. The second difficulty is in the second part due to the consideration of length.

## Solving the Variational Problem on the Manifold
Unlike in the flat Euclidean space that the solution is always given by Eq. 2,  the variational problem Eq. 4 is hard for a general manifold. However, for some manifolds, we can have a closed form solution using some techniques. Examples are Lie groups and the Stiefel manifold.


### Solving the Variational Problem on Lie Groups Utilizing Left Trivialization
The paper solves the problem of optimizing on a general Lie group. However, this blog we only focus on an important case of $\mathsf{SO}(n)$. $\mathsf{SO}(n)$ is called the special orthogonal group, defined as the set of all the orthogonal matrices whose determinant is 1, i.e.,

$$\mathsf{SO}(n):=\{X\in \mathbb{R}^{n\times n}: X^\top X=I, \text{det}(X)=1\}$$

It is a Lie group (a smooth manifold with group structure). Thanks to the group structure, we can solve the variational problem directly on the manifold.

To strengthen the Lie group structure, we will use $g$ to represent a point on the Lie group. The main property we will use on the Lie group is called *left trivilization*:
> The tangent space of $g$ is $\{g\xi:\xi\in\mathbb{R}^{n\times n},\,\xi+\xi^\top=0\}$

Don't worry if you are not familiar with those geometry concept of a Lie group. In English, the momentum is $g\xi$ at position $g$ where $\xi$ is skew-symmetric. Consider the 'left-trivilized momentum' $\xi$ directly makes the structure of the manifold simpler, as the following: 

$$g^\top g=I, \quad\xi^\top+\xi=0$$

We can see the 2 constraints are independent of each other, making the procedure following easier.

We define our Lagrangian as

$$L:=r(t)\left(\frac{1}{2}\langle \xi, \xi\rangle-f(g)\right)$$

$\langle \xi_1, \xi_2\rangle:=tr(\xi_1^\top \xi_2)$ is the inner product.

After calculating the variational critical point (the procedure is technical and intrinsically done on the manifold, thus omitted), we have the following ODE 

$$\begin{cases}
\dot{g}=g\xi\\
\dot{\xi}=-\gamma (t)\xi-\left(\frac{\partial f}{\partial g}^\top g-g^\top \frac{\partial f}{\partial g}\right)
\end{cases}
\tag{5}
$$

In this equation, $\frac{\partial f}{\partial g}$ is the element-wise Euclidean derivative, which is an $n\times n$ matrix. $g\xi$ is again the matrix product. Again, since the variational problem is solved on the manifold, the trajectory stays on the manifold automatically though everything is Euclidean.

### Solving the Variational Problem on the Stiefel manifold via Lagrange Multiplier 

A Stiefel manifold $\mathsf{St}(n,m)$ is the set of $n\times m$ matrices ($n\ge m$) with orthonormal columns, i.e.,

$$\mathsf{St}(n,m):=\{X\in \mathbb{R}^{n\times m}: X^\top X=I_m\}$$

In the special case of $n=m$, $\mathsf{St}(n,n)$ is almost the same as $\mathsf{SO}(n)$. However, in the case $n\ge m$, we no longer have the group structure and we need a different way to solve the variational problem.

We denote the position as $X$ and the momentum as $Q$. Both of them are $n\text{-by-}m$ matrices. The manifold structure is

$$X^\top X=I, \quad Q^\top X+X^\top Q=0$$

We define the Lagrangian as

$$L:=r(t)\left(\frac{1}{2}\langle \dot{X}, \dot{X}\rangle-f(X)\right)$$

$\langle Q_1, Q_2\rangle:=tr(Q_1^\top Q_2)$ is the inner product.

Instead of using an abstract, intrinsic way of variational principle in the last section about $\mathsf{SO}(n)$, here we will have a different approach: constraint Lagrangian [[Chen, Li & Tao]](https://arxiv.org/pdf/2103.12767.pdf). The reason is that the Stiefel manifold lacks the group structure of $\mathsf{SO}(n)$ and cannot perform the left-trivialization.


The variational problem on the manifold can be written as

$$\delta\int_0^T L(X(t), \dot{X}(t), t)\,dt=0,\quad s.t.X^\top X=I, \,\forall 0\le t\le T$$

We can see the constraint is nonlinear, and the variational problem can be hard to solve. As a result, we introduce a Lagrange multiplier function $\Lambda$ (an $n\times n$ symmetric matrix that depends on time) and the Lagrangian becomes

$$\hat{L}(X, \dot{X}, \Lambda, t)=r(t)\Big[\frac{1}{2}\text{tr}\left(\dot{X}^\top(I-aXX^\top)\dot{X}\right)-f(X)\Big]-\frac{1}{2}\text{tr}\left(\Lambda^\top(X^\top X-I)\right)$$

Then we have a variational problem in a flat, Euclidean space. We can solve $\Lambda$ explicitly and also solve the variational problem to get the following ODE

$$
\begin{cases}
    \dot{X}=&Q\\
    \dot{Q}=&-\gamma Q-XQ^\top Q-\frac{3a}{2}(I-XX^\top)QQ^\top X
    -\frac{\partial f}{\partial X}+\frac{1+b}{2}XX^\top\frac{\partial f}{\partial X}+\frac{1-b}{2}X\frac{\partial f}{\partial X}^\top X
\end{cases}
$$

$\frac{\partial f}{\partial X}$ is again the element-wise derivative (an $n\times m$ matrix).
Again, though everything is a matrix in Euclidean space with no constraints, due to the technique of the Lagrange multiplier, the manifold structure is also preserved automatically.

We can see from the definition that $\mathsf{SO}(n)$ is a special case of $\mathsf{St}(n,n)$, and naturally, though the ODE optimizing on the Stiefel manifold seems complicated, it also contains the $\mathsf{SO}(n)$ case Eq. 5 as a special case. To see that, we decompose the tangent space $T_X\mathsf{St}$ into $X$ and $X^\perp$ components by $Q=XY+V$ and use $Y,V$ to replace $Q$.
This transformation changes the constraint $X^\top Q+Q^\top X=0$ to $\{ Y^\top+Y=0,\, X^\top V=0 \}$ 
and the ODE becomes

$$
\begin{align}
    &\dot{X}=XY+V\tag{6a}\\
    &\dot{Y}=-\gamma Y-\frac{1-b}{2}\Big(X^\top \frac{\partial f}{\partial X}-\frac{\partial f}{\partial X}^\top X\Big)\tag{6b}\\
    &\dot{V}=-\gamma V+\frac{3a-2}{2}VY-XV^\top V-\left(I-XX^\top\right)\frac{\partial f}{\partial X}\tag{6c}
\end{align}
$$

where $Y\in\mathbb{R}^{m\times m}, V\in\mathbb{R}^{n\times m}$. We can see (Eq. 6a) and (Eq. 6b) are just the ODE (Eq. 5) for optimizing on $\mathsf{SO}(n)$.

Until now, we have seen the Lagrange's beautiful view about mechanics and how to generalize this view to some curved manifolds to have an ODE optimizing a function. However, to have an algorithm, we need to have a numerical integrator the ODE. However, this is nontrivial. The manifold is curved and the commonly used Euclidean numerical integrator will not work. It needs to be specially designed. Please see the [part II](variational-optimization-2.html) of this blog.
