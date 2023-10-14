---
layout: post
title: "Markov Chain Monte Carlo - Metropolis Hastings Algorithm"
categories: journal
tags: [probabilistic,bayesian, mcmc, sampling]
---

Bayesian inference is about continuously updating our beliefs about a probabilistic model parameter $\boldsymbol \theta$. We first start with our prior distribution $p(\boldsymbol \theta)$, which embodies our knowledge\belief\experience about $\theta$ before actually observing any data,and then we observe some data $\mathcal{D}$, and finally we use Bayes rule to update our belief about $\boldsymbol \theta$ given the occurence of data $\mathcal{D}$ *(or more formally, the posterior $p(\boldsymbol \theta\mid\mathcal{D})$)* through Bayes rule:

$$ \overbrace{p(\boldsymbol \theta \mid \mathcal{D})}^{\text{posterior}} = \frac{\overbrace{p(\mathcal{D}\mid\boldsymbol \theta)}^{\text{likelihood}} \ \overbrace{p(\boldsymbol \theta)}^\text{prior}}{\underbrace{\int_{\boldsymbol \theta} p(\mathcal{D}, \boldsymbol \theta) d\boldsymbol \theta}_\text{evidence}} \tag{1}$$

The main challenge in Bayes rule (eqn. 1), is the denominator, also known as the normalization constant, the evidence or the marginal likelihood:
$$p(\mathcal{D}) = \int_{\boldsymbol \theta} p(\mathcal{D}, \boldsymbol \theta) d\boldsymbol \theta = \int_{\boldsymbol \theta}p(\mathcal{D}\mid\boldsymbol \theta) \ p(\boldsymbol \theta) d\boldsymbol \theta \tag{2}$$

In practice, such integration rarely has a closed form solution (for instance when having conjugate priors), so we turn to approximation methods to sample from our posterior  $p(\boldsymbol \theta\mid\mathcal{D})$. One of such methods is Markov Chain Monte Carlo (MCMC) methods.

## MCMC Sampling - The Metroplois-Hastings Way

Metroplois-Hastings algorithm is one of the simplest MCMC sampling methods, but not the most efficient [2]. The key idea is to start with the following:
1. Initial state $\boldsymbol \theta_o$.
2. A target distribution $\tilde{p}(\boldsymbol\theta\mid\mathcal{D})$, proportional to the posterior $p(\boldsymbol \theta \mid \mathcal{D})$, or simply the unnormalized posterior:
$$ \tilde{p}(\boldsymbol \theta \mid\mathcal{D}) = p(\mathcal{D} \mid \boldsymbol \theta) \ p(\boldsymbol \theta)$$
3. A symmetric proposal distribution $q(\boldsymbol \theta' \mid \boldsymbol \theta)$,  from which we sample new parameters $\boldsymbol \theta'$ to jump or move to.

Starting with our initial state or guess $\boldsymbol \theta_o$, we construct the proposal distribution $q$:
$$ q(\boldsymbol\theta'\mid\boldsymbol\theta_o) = \mathcal{N}(\boldsymbol\theta_o,\gamma)$$

where $\gamma$ is the *proposal width*. We then sample a new proposed parameter value $\boldsymbol \theta'$ from $q$, given only the latest state of $\boldsymbol \theta_o$ (hence it's a Markov chain random walk). We check if the proposed value fits the data better than the previous state $\boldsymbol \theta_o$ by evaluating the target distribution $\tilde{p}$:

$$ \alpha = \frac{p(\mathcal{D} \mid \boldsymbol \theta') \ p(\boldsymbol \theta')}{p(\mathcal{D} \mid \boldsymbol \theta_o) \ p(\boldsymbol \theta_o)} $$

where $\alpha$ is the acceptance ratio. If $\boldsymbol \theta'$ is a better state than $\boldsymbol \theta_o$, then $\alpha > 1$ (we jump to the new state). What if $\alpha < 1$? we will jump in a probabilistic manner, by introducing "$r$" factor defined as  $\text{min}(1, \alpha) $, and we finally sample $u$ from $\mathcal{U}(0, 1)$ and if $u \le r$ we jump to $\theta'$, if not, we stay at $\theta_o$ and the repeat the whole process, for $S$ number of samples.

Summarizing as beautifully formatted by James-A. Goulet in [1]:
1. define $\tilde{p}(\boldsymbol \theta)$ (target distribution)
2. define $q(\boldsymbol \theta'\mid\boldsymbol \theta)$ (proposal distribution)
3. define $S$ (number of samples)
4. initialize $\mathcal{S} := \emptyset$ (set of samples)
5. initialize $\boldsymbol \theta_s := \boldsymbol \theta_o$
6. for $s \in \\{0, 1, .., S-1\\}$ do:
    - define $\boldsymbol \theta := \boldsymbol \theta_s$
    - sample $\boldsymbol \theta'$ from $q(\boldsymbol \theta'\mid\boldsymbol \theta)$
    - compute $\alpha$ 
    - compute $r = \text{min}(1, \alpha)$
    - sample $u \sim \mathcal{U}(0, 1)$
    - if $u \le r$: $\boldsymbol \theta_{s+1} := \boldsymbol \theta'$
      else: $\boldsymbol \theta_{s+1} := \boldsymbol \theta_s$
    - $\mathcal{S} \leftarrow \\{ \mathcal{S} \ \cup \\{ \boldsymbol \theta_{s+1} \\} \\}$

## Visual Guide
(To be completed..)

## References
<a id="1">[1]</a> James-A. Goulet (2021) - Probabilistic Machine Learning for Civil Engineers - The MIT Press.

<a id="2">[2]</a> [MCMC Sampling for Dummies (2015) - Thomas Wiecki](https://twiecki.io/blog/2015/11/10/mcmc-sampling/)