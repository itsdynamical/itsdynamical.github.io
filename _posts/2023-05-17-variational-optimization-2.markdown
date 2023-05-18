---
layout: post
title:  "Variational Optimization, and How It Works for Manifolds (Part II)"
author: 
- Lingkai Kong
- Molei Tao
# author_homepage: https://mtao8.math.gatech.edu
# author: Lingkai Kong
date:   2023-05-17
categories: article
comments_id: 1
---

In the [part I](variational-optimization-1.html) of the blog, the continuous picture is finished with no artificial step involved. In this blog, we focus on solving the second difficulty: structure preserving discretization. And after solving both of the 2 difficulties, we have a variational optimizer and we will see some interesting applications: [improve the performance of Transformer model](#orthogonality-boosts-the-performance-of-transformer-models) and [estimate the Wasserstein distance in high dimensional cases](#projection-robust-wasserstein-prw-distance).  Code can be found [here](https://github.com/konglk1203/VariationalStiefelOptimizer). 
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

<img src="ViT.png" width="400">

Blue: best within classes. Underscore: best over all classes. Green: unconstrained baseline

## Thank you for reading


## References
1. Molei Tao, and Tomoki Ohsawa. "Variational optimization on lie groups, with examples of leading (generalized) eigenvalue problems." In International Conference on Artificial Intelligence and Statistics, pp. 4269-4280. PMLR, 2020.


1. Lingkai Kong, Yuqing Wang, and Molei Tao. "Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport." International Conference on Learning Representations (2023).

1. Renyi Chen, Gongjie Li, and Molei Tao. "Grit: A package for structure-preserving simulations of gravitationally interacting rigid bodies." The Astrophysical Journal 919, no. 1 (2021): 50.

1. François-Pierre Paty, and Marco Cuturi. "Subspace robust Wasserstein distances." In International conference on machine learning, pp. 5072-5081. PMLR, 2019.

1. Tianyi Lin, Chenyou Fan, Nhat Ho, Marco Cuturi, and Michael Jordan. "Projection robust Wasserstein distance and Riemannian optimization." Advances in neural information processing systems 33 (2020): 9383-9397.

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." Advances in neural information processing systems 30 (2017).

1. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. "An image is worth 16x16 words: Transformers for image recognition at scale." International Conference on Learning Representations (2021).

1. Andre Wibisono, Ashia C. Wilson, and Michael I. Jordan. "A variational perspective on accelerated methods in optimization." proceedings of the National Academy of Sciences 113, no. 47 (2016): E7351-E7358.
