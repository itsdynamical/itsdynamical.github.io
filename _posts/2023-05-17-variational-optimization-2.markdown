---
layout: post
title:  "Variational Optimization, and How It Simplifies Manifold Optimization (Part II: the discrete side of the story)"
author: 
- Lingkai Kong
- Molei Tao
# author_homepage: https://mtao8.math.gatech.edu
# author: Lingkai Kong and Molei Tao
date:   2023-06-01
categories: article
comments_id: 1
---


[Part I](variational-optimization-1.html) of the blog described how to obtain ODEs that are written in Euclidean space but optimize functions on manifolds. To turn them into actual optimization algorithms in discrete time, we need to discretize the time of the ODEs. This is not trivial, because a naive discretization would disrespect the fact that the exact solution remains on the manifold (see fig.1 left). Of course, one can always add an extra step that artificially pulls things back to the manifold, but this operation can be computational costly, and it partially cancels efforts and thus possibly slows down the convergence of the optimization too (see fig.1 right). We'd like to be computationally efficient (to experts: e.g., as little usage of exponential maps or projections as possible, computation complexity more dependence on $m$ but less on $n$ (we could have $n\gg m$), etc.) and avoid such cancellations as much as possible.
<p align="middle">
  <img src="https://github.com/itsdynamical/itsdynamical.github.io/blob/blog/images/Untitled-1.png?raw=true" width="200" />
  <img src="https://github.com/itsdynamical/itsdynamical.github.io/blob/blog/images/Untitled-2.png?raw=true" width="200" /> 
</p>
<p align = "center">
Fig.1 - What will happen if we artifacially pull the point back?
</p>

In this Part II of the blog, we will construct such optimizers. Then we will showcase some interesting applications, such as [generically improving the performance of Transformer models](#orthogonality-boosts-the-performance-of-transformer-models), and [approximating Wasserstein distances in high dimensions](#projection-robust-wasserstein-prw-distance).  Codes of the generic optimizers as well as the applications can be found [here](https://github.com/konglk1203/VariationalStiefelOptimizer). 

## Reminder of the Optimization ODE, and Further Preparation
As a continuation from [Part I](variational-optimization-1.html), we will focus on optimization on Stiefel manifold. The specific case of $\mathsf{SO}(n)$ Lie group will be a special case of the Stiefel manifold $\mathsf{St}(n,m)$ when $n=m$. The optimization dynamics, obtained from variational optimization, is
$$
\begin{cases}
    \dot{X}=&Q\\
    \dot{Q}=&-\gamma Q-XQ^\top Q
    -\frac{\partial f}{\partial X}+\frac{1}{2}XX^\top\frac{\partial f}{\partial X}+\frac{1}{2}X\frac{\partial f}{\partial X}^\top X
\end{cases}
$$

where position $X\in \mathsf{St}(n,m)$ and momentum/velocity $Q\in T_X \mathsf{St}(n,m)$ . Rich geometric information is contained there. In the Lie group case, we used left-trivialization to represent the velocity variable $\dot{g}$ using $\xi$ that lives in the Lie algebra, via $\dot{g}=g\xi$. Now our position is $X$ (as opposed to $g$) and velocity is $Q$ (as opposed to $\dot{g}$), but we don't have a group structure, and if we pretend $Q=XY$ and use $Y$ as a new representation of velocity, we will have big trouble --- $Q$ is $n\times m$ and $Y$ then has to be $m\times m$, but $n > m$ and we would have lost information about the velocity! Instead, we decompose the tangent space $T_X\mathsf{St}$ into $X$ and $X^\perp$ components by $Q=XY+V$, where $XY$ is in the span of $X$, and $V$ is an "orthogonal" remainder. Given $X^\top X=I$ and $X^\top Q+Q^\top X=0$, one can show that this transformation turns these the velocity constraint $X^\top Q+Q^\top X=0$ into $Y^T+Y=0$ and $X^\top V=0$ instead, the latter giving the precise meaning of "orthogonal".

The above is the static geometric picture, but there is more. Remember $X, Q$ are actually functions of time $X(t), Q(t)$. If one does this decomposition for each $t$, what dynamics will the resulting $Y(t), V(t)$ have? It turns out that they are given by some elegant ODEs

$$
\begin{align}
    &\dot{X}=XY+V\tag{8a}\\
    &\dot{Y}=-\gamma Y-\frac{1-b}{2}\Big(X^\top \frac{\partial f}{\partial X}-\frac{\partial f}{\partial X}^\top X\Big)\tag{8b}\\
    &\dot{V}=-\gamma V+\frac{3a-2}{2}VY-XV^\top V-\left(I-XX^\top\right)\frac{\partial f}{\partial X}\tag{8c}
\end{align}
$$

and as long as the initial condition satisfies $X(0)^T X(0)=I, Y(0)^T+Y(0)=0$ and $X(0)^\top V(0)=0$, the solution automatically maintains the new structural constraints $X(t)^T X(t)=I, Y(t)^T+Y(t)=0$ and $X(t)^\top V(t)=0$, for all $t>0$, and of course, $Q(t):=X(t)Y(t)+V(t)$ will exactly satisfy its constraint and remain in $T_{X(t)} \mathsf{St}$ too.

When $n=m$, $V=0$, and we degenerate to the Lie group case (Eq.8a and 8b become just Eq.7 in [Part I](variational-optimization-1.html)).


<span style="color:blue">
Tao: to be conti.
</span>.

## Nontrivial Discretization for Computationally Efficient Structure Preservation
Structure preserving means the manifold structure is preserved, i.e., keep the point always stays on the manifold. In the continuous case, the manifold structure is preserved since the variational problem is done among all the curves on the manifold.

When it comes to the discretization of ODEs on the manifold, things gets harder. Imagine the surface of the unit ball in $\mathbb{R}^n$. If a particle is moving on the surface of the ball following some ODE, then its velocity/momentum must be always tangent to the ball. Otherwise, the particle will have a radical velocity and leaves the surface. 

>So, 'preserving of manifold structure' under the context of gradient-based optimizer means the 2 constraints are satisfied:
>- the position stays on the manifold
>- the momentum stays on the tangent space of the position

If the numerical discretization is not done properly, the manifold structure may fail to be preserved. Imagine a point is moving continuously on the surface of a ball and of course, the trajectory is structure preserving. But after being discretized, at each time step, the point goes one step forward to a tangent direction by forward Euler. In this discrete case, the point is no longer on the ball but becomes a 'satellite' because you are in the space out of the ball. In other words, even the ODE is structure preserving, 
> The discretization error from a Euclidean numerical integrator like forward-Euler or Runge-Kutta may lead to the loss of manifold structure.

We use the example of $\mathsf{SO}(n)$ to give an example of structure preserving numerical integrator. The Stiefel case is too complicated to be shown here. The ODE optimizing on $\mathsf{SO}(n)$ is Eq. 7 and we copy it here again for convenience:

$$\begin{cases}
\dot{g}=g\xi\\
\dot{\xi}=-\gamma (t)\xi-\left(\frac{\partial f}{\partial g}^\top g-g^\top \frac{\partial f}{\partial g}\right)
\end{cases}
$$

It can be viewed as the sum of 3 ODEs

$$\begin{cases}
\dot{g}=g\xi\\
\dot{\xi}=0
\end{cases}
\quad
\begin{cases}
\dot{g}=0\\
\dot{\xi}=-\gamma (t)\xi
\end{cases}
\quad
\begin{cases}
\dot{g}=0\\
\dot{\xi}=-\left(\frac{\partial f}{\partial g}^\top g-g^\top \frac{\partial f}{\partial g}\right)
\end{cases}
$$

The ODEs are carefully to have nice properties, but one of the most important property is
> Each of the ODEs is structure preserving

They are also easy to integrate: all of them are linear and have closed form solutions. Evolving them alternatively gives us a numerical integrator (Algo. 2 in the [Tao & Ohsawa, 2020](https://arxiv.org/pdf/2001.10006.pdf)) by the [Lie-Trotter Theorem](https://en.wikipedia.org/wiki/Lie_product_formula).

Here, the structure perserving comes from the carefully chosen splitting: we found a splitting, such that each of them is structure perserving and also has a closed from solution. The advantage is, even in the discretization, no atifacial step is specially introduced for the curved manifold. We just perform a numerical splitting, which is what we always have to do when the ODE is nonlinear. The advantage is intuitive: it will emperically performs better. What's more, no artifacial step for the manifold makes it super cheap in the sence of computational cost per iteration.

## Some Applications of Optimization on the Stiefel manifold
Going through all these nontrivialities is worthful, as now we can finally have many great applications:
### Subspace persuing: finding the best linear subspace for projection
Subspace pursuing view a Stiefel matrix as a projection from $n$-dim spaces to a $m$-dim subspace with its columns. Suppose we have a dataset $\lbrace x_i \rbrace_{i=1}^k$ with $x_i$ in $\mathbb{R}^n$ and a function $f$. Sometimes, when the dimension $n$ is high, the function can be too costly to evaluate. To solve this difficulty, instead of evaluating our function $f(\lbrace x_i\rbrace_{i=1}^k)$ directly, we consider the optimization problem

$$\max_{U\in \mathsf{St}(n,m)} f(\lbrace U^\top x_i\rbrace_{i=1}^k).$$ 

$U$ can be viewed as a projection to a low-dimensional subspace. We take the maximum of $U$ in the sense that the information is preserved as much as we can with the column of $U$ being a set of the orthonormal basis of the subspace. If we choose $m\ll n$, then this can significantly save computational resources. Here are 2 examples for subspace pursuing.
#### Leading EigenValue (LEV) problem
Given an $n\times n$ matrix $A$, the task is to get the top $m$ eigenvalues. Simply computing all the $n$ eigenvalues and sorting them can be too expensive and wasteful in the case $m\ll n$. Instead, we consider converting it to the following optimization problem by subspace pursuing.

$$\max_{U\in \mathsf{St}(n,m)} \text{tr}(U^\top A U)$$

The maximum is in the sense that after being projected, the matrix $A$ has the largest sum of eigenvalues. If we want to use the Lie group optimizer, we need to write it as the following instead

$$\max_{R\in \mathsf{SO}(n)} \text{tr}(E^\top R^\top A RE)$$

where 

$$E=\begin{pmatrix}
I_m\\
0
\end{pmatrix}
$$

is an $n\times m$ matrix with $I_m$ is the $m\times m$ identity matrix and $0$ is the $(n-m)\times m$ zero matrix. $RE$ is the first $m$ columns of the $\mathsf{SO}(n)$ matrix, which is the $\mathsf{St}(n,m)$ matrix $U$ above.


#### Projection Robust Wasserstein (PRW) Distance
The great idea of [Projection Robust Wasserstein Distance](https://arxiv.org/pdf/2006.07458.pdf)[Paty & Cuturi, 2019][Lin et al. 2020] can be viewed as a special case of subspace pursuing. Given 2 probability measures $\mu,\nu$ on $\mathbb{R}^n$, we denote the set of all couplings as $\Pi(\mu,\nu)$. We first define the Wasserstein distance between $\mu$ and $\nu$ as

$$W_2(\mu,\nu) := \min_{\pi \in \Pi(\mu,\nu)} \left( \int \|x-y\|^2 \,d\pi(x,y) \right)^{1/2}$$

It tries to find a coupling of $\mu$ and $\nu$ whose cost is the lowest. Solving this problem numerically can be costly when the dimension is high. The beautiful idea of PRW bypasses this difficulty by projecting the 2 distributions to lower dimensional subspaces and then computing the $W_2$ distance in this lower dimensional space instead, i.e.,

$$P_m(\mu,\nu) := \max_{U\in \mathsf{St}(n,m)} \min_{\pi \in \Pi(\mu,\nu)} \left( \int \|U^\top x - U^\top y\|^2 \,d\pi(x,y) \right)^{1/2}$$

Same as mentioned before, the maximization is in the sense of keeping as much information as possible. This approach not only makes the problem computationally more manageable when $m\ll n$. What's more, since the dimensions that are relatively less important are omitted after projection, the noise is also reduced and only the essential component is left, which increases the robustness compared to the vanilla $W_2$ distance.

### Orthogonality Boosts the Performance of Transformer Models
Transformer [[Vaswani et al.]](https://arxiv.org/pdf/1706.03762.pdf) is a recent but extremely powerful deep learning architecture. It was first invented for NLP, but Vision Transformer (ViT) [[Dosovitskiy et al]](https://arxiv.org/pdf/2010.11929.pdf) also applies is to computer vision. The key to why the Transformer is powerful is that the attention layer is able to characterize long-distance interactions between elements in the sequence. The 'elements' mean 'words' in NLP tasks and 'patches' in CV tasks. We follow the traditional notations in [Vaswani et al.]. 

The trainable parameters in multi-headed attention layers are $W_i^Q, W_i^K\in \mathbb{R}^{d_{model}\times d_k}$ and $W_i^V\in \mathbb{R}^{d_{model}\times d_v}$ where $i$ stands for label of heads. The widely accepted intuition is

> $W_i^Q$ and $W_i^K$ try to catch the interaction between elements and the information their extract will be less redundant if their columns are forced to be orthonormal.

From the test result shown below, we can see that simply applying our optimizer to vanilla ViT improves the validation accuracy on CIFAR 10 from 9.05% to 8.32% (trained from scratch).

<img src="https://raw.githubusercontent.com/itsdynamical/itsdynamical.github.io/blog/images/ViT.png" width="400">

Blue: best within classes. Underscore: best over all classes. Green: unconstrained baseline

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

## Thank you for reading!


## References
1. Molei Tao, and Tomoki Ohsawa. "Variational optimization on lie groups, with examples of leading (generalized) eigenvalue problems." International Conference on Artificial Intelligence and Statistics (2020).


1. Lingkai Kong, Yuqing Wang, and Molei Tao. "Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport." International Conference on Learning Representations (2023).

1. Renyi Chen, Gongjie Li, and Molei Tao. "GRIT: A package for structure-preserving simulations of gravitationally interacting rigid bodies." The Astrophysical Journal (2021).

1. François-Pierre Paty, and Marco Cuturi. "Subspace robust Wasserstein distances." International Conference on Machine Learning (2019).

1. Tianyi Lin, Chenyou Fan, Nhat Ho, Marco Cuturi, and Michael Jordan. "Projection robust Wasserstein distance and Riemannian optimization." Advances in Neural Information Processing Systems (2020).

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." Advances in Neural Information Processing Systems (2017).

1. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. "An image is worth 16x16 words: Transformers for image recognition at scale." International Conference on Learning Representations (2021).

1. Andre Wibisono, Ashia C. Wilson, and Michael I. Jordan. "A variational perspective on accelerated methods in optimization." Proceedings of the National Academy of Sciences (2016).

1. Weijie Su, Stephen Boyd, and Emmanuel Candes. "A differential equation for modeling Nesterov’s accelerated gradient method: theory and insights." Advances in neural information processing systems 27 (2014).

1.  Boris T. Polyak. "Some methods of speeding up the convergence of iteration methods." Ussr computational mathematics and mathematical physics 4.5 (1964): 1-17.
