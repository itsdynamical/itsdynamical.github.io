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


[Part I](variational-optimization-1.html) of the blog described how to obtain ODEs that are written in Euclidean space but optimize functions on manifolds. To turn them into actual optimization algorithms, we need to discretize the time of the ODEs, so that evolution in discrete time corresponds to iterations of an optimizer. 

This is nontrivial, because a naive discretization would destroy the fact that the exact solution remains on the manifold (see Fig.1 left panel). Of course, one can always add an extra step that artificially pulls things back to the manifold (e.g., a projection), but this operation can be computational costly. In addition, it partially cancels efforts and thus possibly slows down the convergence of the optimization too (see Fig.1 right panel). We'd like to be computationally efficient and avoid such cancellations as much as possible (to experts, these mean, for example, as little usage of exponential maps or projections as possible, computation complexity more dependent on $m$ but less on $n$ (we could have $n\gg m$), smaller constants in error bounds, etc.)
<p align="middle">
  <img src="https://github.com/itsdynamical/itsdynamical.github.io/blob/blog/images/Untitled-1.png?raw=true" width="200" />
  <img src="https://github.com/itsdynamical/itsdynamical.github.io/blob/blog/images/Untitled-2.png?raw=true" width="200" /> 
</p>
<p align = "center">
Fig.1 - What will happen if we artifacially pull the point back?
</p>

In this Part II of the blog, we will construct such optimizers. Then we will showcase some interesting applications, such as [a general way to improve the performance of Transformer models](#Sec_Transformer), and [approximating Wasserstein distances in high dimensions](#Sec_PRW).  Codes of the generic optimizers, as well as these applications, can be found [here](https://github.com/konglk1203/VariationalStiefelOptimizer). 


## Reminder of the Optimization ODE, and Further Preparation
As a continuation from [Part I](variational-optimization-1.html), we will focus on optimization on Stiefel manifold. The specific case of $\mathsf{SO}(n)$ Lie group will be a special instance of the Stiefel manifold $\mathsf{St}(n,m)$ when $n=m$. The optimization dynamics, as obtained from variational optimization in [Part I](variational-optimization-1.html), is

$$
\begin{cases}
    \dot{X}=&Q\\
    \dot{Q}=&-\gamma Q-XQ^\top Q
    -\frac{\partial f}{\partial X}+\frac{1}{2}XX^\top\frac{\partial f}{\partial X}+\frac{1}{2}X\frac{\partial f}{\partial X}^\top X
\end{cases}
$$

where position $X\in \mathsf{St}(n,m)$ and momentum/velocity $Q\in T_X \mathsf{St}(n,m)$. 

Rich geometric information can be obtained there. In the aforementioned Lie group case, we used left-trivialization to represent the velocity variable $\dot{g}$, i.e., using $\xi$ that lives in the Lie algebra, via $\dot{g}=g\xi$. Now our position is $X$ (replacing $g$) and velocity is $Q$ (replacing $\dot{g}$), but we don't have a group structure, and if we pretend to do the same thing, namely $Q=XY$ and use $Y$ as a new representation of velocity, we will have big trouble --- $Q$ is $n\times m$ and $Y$ then has to be $m\times m$, but $n > m$, and we would have lost information about the velocity! Instead, we decompose the tangent space $T_X\mathsf{St}$ into $X$ and $X^\perp$ components by $Q=XY+V$, where $XY$ is in the span of $X$, and $V$ is an "orthogonal" remainder. Given $X^\top X=I$ and $X^\top Q+Q^\top X=0$, one can show that this transformation turns these the velocity constraint $X^\top Q+Q^\top X=0$ into $Y^T+Y=0$ and $X^\top V=0$ instead, the latter giving the precise meaning of $V$ being orthogonal to $X$.

The above is the static geometric picture, but there is more. Remember $X, Q$ are actually functions of time $X(t), Q(t)$. If one does this decomposition for each $t$, what dynamics will the resulting $Y(t), V(t)$ obey? It turns out that they are given by some elegant ODEs

$$
\begin{align}
    &\dot{X}=XY+V\tag{8a}\\
    &\dot{Y}=-\gamma Y-\frac{1-b}{2}\Big(X^\top \frac{\partial f}{\partial X}-\frac{\partial f}{\partial X}^\top X\Big)\tag{8b}\\
    &\dot{V}=-\gamma V+\frac{3a-2}{2}VY-XV^\top V-\left(I-XX^\top\right)\frac{\partial f}{\partial X}\tag{8c}
\end{align}
$$

and as long as the initial condition satisfies $X(0)^T X(0)=I, Y(0)^T+Y(0)=0$ and $X(0)^\top V(0)=0$, the solution automatically maintains the new structural constraints $X(t)^T X(t)=I, Y(t)^T+Y(t)=0$ and $X(t)^\top V(t)=0$, for all $t>0$, and of course, $Q(t):=X(t)Y(t)+V(t)$ will exactly satisfy its constraint and remain in $T_{X(t)} \mathsf{St}$ too.







## Nontrivial Discretization for Computationally Efficient Structure Preservation
(Geometric) structure preservation means values of relevant variables stay on their respective manifolds. For our case, namely momentum-accelerated manifold optimization, it corresponds to satisfying 2 constraints:
>- the position variable stays on the manifold
>- the momentum variable stays on the tangent space of the manifold (based at the position variable)

In the continuous case, the manifold structure is preserved, because the variational problem is solved with respect to variations of curves on the manifold. Solving such a problem is nontrivial, but already accomplished (see Part I).

When it comes to discretzing ODEs on the manifold, on the other hand, things become even more difficult. One has to design a delicate numerical discretization, because otherwise the manifold structure may fail to be preserved, despite that the ODE in continuous time is structure preserving. This is often the case for off-the-shelf numerical schemes, such as Euler methods or Runge-Kutta (note both forward and backward Euler methods are special cases of Runge-Kutta).

Nevertheless, it is possible to discretize Eq.8 in a computationally cheap and accurate way, for obtaining iterates that exactly satisfy both constraints for all steps. The construction is a bit convolved because we'd like to maximize the computational efficiency, and we will just give some flavor of the tricks used. 

To fix ideas, let's first start with a simpler example, namely when the manifold is the Lie group $\mathsf{SO}(n)$. This is a special case of the Stiefel problem, in which we let $n=m$ and then $V$ becomes 0, Eq.8a and 8b become just Eq.7 in [Part I](variational-optimization-1.html), and Eq. 8c disappears. We copy Eq.7 here for convenience:

$$\begin{cases}
\dot{g}=g\xi\\
\dot{\xi}=-\gamma (t)\xi-\left(\frac{\partial f}{\partial g}^\top g-g^\top \frac{\partial f}{\partial g}\right)
\end{cases}
\tag{duplicate of Eq.7}
$$

To numerically simulate this ODE, we adopt a vector field splitting approach and strategically decompose its right hand side, known as vector field, as the sum of 3 vector fields, and consider their respective evolution dynamics:

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
\tag{9}
$$

The specific splitting is such that these split ODE systems have nice properties, and one of the most important properties is
> Each of the 3 ODE systems is structure preserving

They are also easy to integrate: all of them admit closed form solutions. Evolving them alternatively gives us a numerical integrator of Eq.(7) (Algo. 2 in the [Tao & Ohsawa, 2020](https://arxiv.org/pdf/2001.10006.pdf)), and the [Lie-Trotter operator splitting theorem](https://en.wikipedia.org/wiki/Lie_product_formula) ensures that this integrator approximates its exact solution.

Because each evolution exactly preserves the manifold structures, so does their composition (i.e., alternatively evolving them). Having simple closed form solutions also ensures a low computational cost (experts may question the cost of exponential map, which is needed for solving $\dot{g}=g\xi$, but even this can be avoided in a more advanced discretization; see 3 paragraphs below).

Now let's describe the full-blown Stiefel version (where $n>m$). The optimization ODE to discretize is Eq.8, and we decompose its right hand side as the sum of 3 carefully chosen vector fields, and their respective evolution dynamics are:

$$
    \begin{cases}
        &\dot{X}=XY\\
        &\dot{Y}=-\gamma Y\\
        &\quad -\frac{1-b}{2}\left(X^\top\frac{\partial f}{\partial X}-\frac{\partial f}{\partial X}^\top X\right)\\
        &\dot{V}=0
    \end{cases}
    \begin{cases}
        &\dot{X}=0\\
        &\dot{Y}=0\\
        &\dot{V}=-\gamma V+\frac{3a-2}{2}VY\\
        &\quad -(I-XX^\top)\frac{\partial f}{\partial X}
    \end{cases}
    \begin{cases}
        \dot{X}=&V\\
        \dot{Y}=&0\\
        \dot{V}=&-XV^\top V
    \end{cases}
    \tag{10}
$$

Again, one can check that each of these 3 ODE systems is structure preserving. Moreover, the first system in Eq. 10 is similar to the $SO(n)$ case (Eq.7) that we just discussed. Even those it does not admit a closed form solution, we can use the same numerical discretization as Eq.7 (given by Eq.9) for it. The second system is a linear ODE and its explicit solution can be cheaply computed. The third system is nonlinear, but a specially designed numerical discretization that preserves the manifold structure can be constructed; it is too complicated to be presented here, but interested experts are referred to  [[Kong, Wang & Tao, 2023]](https://arxiv.org/pdf/2205.14173.pdf) for details.

Alternating these integrators for the 3 ODEs in Eq.10 gives us a numerical optimizer, that exactly preserves the Stiefel manifold and its tangent structure. 

Even better is, the computational cost of this structure preserving optimizer can be further reduced. Here are some technical details for experts: Costly matrix exponentiation operations are needed for computing the exact solutions of linear ODEs such as system 1 in Eq.10, but they can actually be avoided. If we use a cheaper forward Euler integrator to approximate the evolution of system 1, structure preservation will be destroyed by this step. However, it is a small miracle that, if we fist evolve system 2, followed forward Euler for system 1, and finally system 3, then deviation from the manifold created by forward Euler will be corrected! This carefully chosen *ordering* of composition makes the overall iteration still structure preserving, while significantly lowering the computational complexity.

So, in the end, we can obtain a highly-computational-efficient optimization algorithm, that stays exactly on the manifold forever, and faithfully captures the nice convergence property of the continuous-in-time optimization dynamics.


## Some Applications of Optimization on the Stiefel manifold
It is now time to see a subset of useful applications. 
Let us begin with a simple problem, which is nevertheless at the heart of data sciences ---

### Leading EigenValue (LEV) problem
Given an $n\times n$ matrix $A$, the task is to compute its largest $m$ eigenvalues. Simply computing all the $n$ eigenvalues and sorting them can be too expensive and wasteful in the case $m\ll n$, and modern data set often corresponds to huge $n$ (e.g., $\geq 10^6$) such that any method with $\mathcal{O}(n^3)$ computational complexity or storage (needed by traditional eigenvalue methods) is unaffordable.

Instead, we can convert the task to an optimization problem

$$\max_{U\in \mathsf{St}(n,m)} \text{tr}(U^\top A U)$$

where $U$ represents the full bases of an m-dimensional subspace in the n-dimensional space. By searching for the optimal $U$, we look for the best subspace to project $A$ to, such that $A$ restricted to that subspace has maximized sum of eigenvalues. The minimizer $U$ will then give a small $m\times m$ matrix $U^T A U$, whose eigenvalues correspond to $A$'s $m$ leading eigenvalues.

One may think this problem is too easy as the objective function is quadratic, but in fact this optimization problem is not even convex, because there is a nonlinear equality constraint $U^\top U=I_{m\times m}$. Nevertheless, please read on if you'd like to see more complicated objective functions.




<a id="Sec_PRW"> </a>
### Projection Robust Wasserstein (PRW) Distance
Wasserstein distance is a very important notion in machine learning, as it quantifies a distance between two probability distributions. If these distributions are for high-dimension random variables, however, the computation of Wasserstein distance is very challenging; for example, 1) one needs a lot of sample points of the distributions (i.e. data), 2) the computation of the distance can be very expensive.

One way to alleviate these issues is to use Projection Robust Wasserstein Distance (e.g., [[Paty & Cuturi, 2019]](https://arxiv.org/pdf/1901.08949.pdf), [[Lin et al. 2020]](https://arxiv.org/pdf/2006.07458.pdf)). Let's first review Wasserstein distance: given 2 probability measures $\mu,\nu$ on $\mathbb{R}^n$, we denote the set of all couplings as $\Pi(\mu,\nu)$. The Wasserstein distance between $\mu$ and $\nu$ can be defined as

$$W_2(\mu,\nu) := \min_{\pi \in \Pi(\mu,\nu)} \left( \int \|x-y\|^2 \,d\pi(x,y) \right)^{1/2}$$

Imagine $\mu$ and $\nu$ describe the shapes of a sand pile. Wasserstein distance basically tries to the least sand movement plan so that the $\mu$ pile becomes the $\nu$ pile. To deal with the case where the dimension of $x$ and $y$ is high, the beautiful idea of PRW for bypassing the curse of dimensionality is, to project these 2 distributions to lower dimensional subspaces, and then compute the distance in this lower dimensional space instead, and finally use an outer loop to find the best subspace to project to. Mathematically, it is given by a bi-level optimization problem

$$P_m(\mu,\nu) := \max_{U\in \mathsf{St}(n,m)} \min_{\pi \in \Pi(\mu,\nu)} \left( \int \|U^\top x - U^\top y\|^2 \,d\pi(x,y) \right)^{1/2}$$

The maximization is in order to keep as much information as possible. This approach not only makes the problem computationally more manageable, but also less data-hungry, when $m\ll n$. Moreover, since the dimensions that are relatively less important are omitted after projection, data noise is also reduced and only the essential component is left, which increases the robustness compared to the vanilla $W_2$ distance.

This is again a Stiefel optimization problem, and it is important to be exactly on the manifold. Near enforcement of the manifold structure, such as commonly by regularization, will lead to approximate orthogonality which would totally destroy the subspace structure.


### Subspace Pursue: finding the best subspace to approximate a high dim. optimization problem

What do the aforementioned {Leading EigenValue problem} and {Projection Robust Wasserstein Distance example} have in common? They are both based on the idea of approximating a high dimensional problem by looking for an optimal low dimensional projection, and then solving the problem in that low dimensional subspace. In fact, we can make this strategy general, and this results in what we call *Subspace Pursue*. Both LEV and PRWD are instances of Subspace Pursue. Here is a precise, although not the most general, formulation of Subspace Pursue:

Given a dataset $\lbrace x_i \rbrace_{i=1}^k$ and a function $f$, which abstractly denotes the outcome of some algorithm applied to this dataset. Suppose this algorithm can work with various datasets of different dimensions, meaning both $f(\lbrace x_i\rbrace_{i=1}^k)$ with $x_i$ in $\mathbb{R}^n$ and $f(\lbrace y_i\rbrace_{i=1}^k)$ with $y_i$ in $\mathbb{R}^m$
are well-defined. If $f(\lbrace x_i\rbrace_{i=1}^k)$ is computationally too expensive to evaluate in dimension $n$, but not in dimension $m \ll n$, then we can consider instead the optimization problem
$$\max_{U\in \mathsf{St}(n,m)} f(\lbrace U^\top x_i\rbrace_{i=1}^k).$$ 

This is again a Stiefel optimization problem that can be pleasantly solved by optimizers described in this blog. It views a Stiefel matrix $U$ as a projection from $n$-dim spaces to a $m$-dim subspace, spanned by its (orthonormal) columns. The maximization is again to make sure that as much information as possible is captured by a low dimension approximation.


<a id="Sec_Transformer"> </a>
### Orthogonality Boosts the Performance of Transformer Models
Transformer [[Vaswani et al.]](https://arxiv.org/pdf/1706.03762.pdf) is an extremely powerful deep learning architecture. It was first invented for NLP, but then also applied to Computer Vision (e.g., Vision Transformer (ViT) [[Dosovitskiy et al]](https://arxiv.org/pdf/2010.11929.pdf)). One amazing thing of Transformer is, its attention layer is able to characterize long-distance interactions between elements in the sequence, where 'elements' mean 'words' in NLP tasks and 'patches' in CV tasks. 

Can non-Euclidean optimization make the self-attention mechanism even better? The main intuition is, many of the trainable parameters in attention layers aim at capturing correlations between elements, via training. If we require these correlations to be orthogonal to each other, information extracted by the attention mechanism can be less redudant and more accurate.

To try this idea out, one simply replaces the Euclidean optimization in training by Stiefel optimization, and it really works well in all tested cases. For example, for vanilla ViT trained *from scratch* for CIFAR 100, one only needs to modify 2 lines of code to enforce orthogonality, and then test error goes down from 33.1% to 30.2%.


## Thank you for reading!
If you have any comment or question, please don't hesitate to let us know!

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


## References
1. Molei Tao, and Tomoki Ohsawa. "Variational optimization on lie groups, with examples of leading (generalized) eigenvalue problems." International Conference on Artificial Intelligence and Statistics (2020).


1. Lingkai Kong, Yuqing Wang, and Molei Tao. "Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport." International Conference on Learning Representations (2023).

1. Renyi Chen, Gongjie Li, and Molei Tao. "GRIT: A package for structure-preserving simulations of gravitationally interacting rigid bodies." The Astrophysical Journal (2021).

1. François-Pierre Paty, and Marco Cuturi. "Subspace robust Wasserstein distances." International Conference on Machine Learning (2019).

1. Tianyi Lin, Chenyou Fan, Nhat Ho, Marco Cuturi, and Michael Jordan. "Projection robust Wasserstein distance and Riemannian optimization." Advances in Neural Information Processing Systems (2020).

1. François-Pierre Paty, and Marco Cuturi. "Subspace robust Wasserstein distances." International conference on machine learning. PMLR, 2019.


1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." Advances in Neural Information Processing Systems (2017).

1. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. "An image is worth 16x16 words: Transformers for image recognition at scale." International Conference on Learning Representations (2021).

1. Andre Wibisono, Ashia C. Wilson, and Michael I. Jordan. "A variational perspective on accelerated methods in optimization." Proceedings of the National Academy of Sciences (2016).

1. Weijie Su, Stephen Boyd, and Emmanuel Candes. "A differential equation for modeling Nesterov’s accelerated gradient method: theory and insights." Advances in neural information processing systems 27 (2014).

1.  Boris T. Polyak. "Some methods of speeding up the convergence of iteration methods." Ussr computational mathematics and mathematical physics 4.5 (1964): 1-17.
