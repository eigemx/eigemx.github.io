---
layout: post
title: "Probabilistic Generative and Discriminative Models"
author: "M. Emara"
categories: journal
tags: [probabilistic,generative,discriminative]
---

## Introduction
Suppose we are trying to build a classification model, and we have a training dataset with $M$ training examples, $\mathcal{D} := \\{(x^{(i)}, y_1^{(i)}), ... , (x^{(M)}, y^{(M)})\\}$ where $x \in \mathbb{R}^N$ and $y \in \mathcal{C}$ where $\mathcal{C} = \\{0, .., C\\}$ (i.e.: we have $C + 1$ different classes). We could easily fit a logistic regression or softmax model by modelling $p(y=c|x,\theta)$ where $c \in \mathcal{C}$, and find the optimum point estimate of model parameters $\theta$ for the separator through maximum likelihood estimation $\theta_{MLE}$ or maximum a posteriori $\theta_{MAP}$. Another interesting way to build such model is through generative models.

## Discriminative vs. Generative
A **discriminative** model is a probabilistic model that takes a vector $x$ as an input, and calculate the probability that it belongs to a class $c$: $p(y=c|x,\theta)$ (logistic regrission model fit such description and is discriminative). A **generative** model on the other hand is used to generate/sample new data points (vectors) from a certain class $c$: $p(x|y=c,\theta)$, for example, generating new images of the digit 8 from the MNIST dataset, or sample random sestosa flower sepal and petal dimensions from the Iris dataset.

From Bayes rule we can see that both are closely related:
$$
\begin{aligned}\overbrace{p(y=c|x,\theta)}^\text{Discriminator} &= \frac{\overbrace{p(x|y=c,\theta)}^{\text{Generator}} \ p(y=c)}{p(x)} \\\ &= \frac{\overbrace{p(x|y=c,\theta)}^{\text{Generator}} \ p(y=c)}{\sum_{c^\prime} p(x|y=c^\prime,\theta)\ p(y=c^\prime)}\end{aligned} \tag{1}
$$


Let's start by our generative model, the family of generative models we are going to discuss in this article are all Gaussians, hence the name: Gaussian Discriminant Analysis (GDA) [1], such that: 
$$
p(x|y=c, \theta) \sim \mathcal{N}(x|\mu_c, \Sigma_c) \tag{2}
$$


And as we will see shortly, the way we choose how to set the generator covariance matrix $\Sigma$ would result in two different models: linear discriminant classifier and quadratic discriminant classifier.

To have an example, I have generated a toy dataset $\mathcal{D}$ with three differenct classes $c \in \mathcal{C}$, plotted as black $(c=0)$, red $(c=1)$ and blue $(c=2)$, and $x \in \mathbb{R}^2 = \begin{bmatrix}x_1 \\\ x_2 \end{bmatrix}$.

![alt text](/assets/img/data_scatter_plot.png)

### Generator
In GDA, we assume that $p(x|y=c, \theta)$ is modeled as a multivariate Gaussian $\mathcal{N}(x|\mu_c, \Sigma_c)$, where $\mu_c$ is the mean vector for the subset of points in $\mathcal{D}$ for a specific class $c$ (meaning we have 3 different Gaussians, one for each class), and $\Sigma_c$ is the covariance matrix.


From Murphy[1], we can expand the posterior $p(y=c|x,\theta)$ as:
$$
\begin{aligned} p(y=c|x, \theta)  &\propto p(x|y=c, \theta)\ p(y=c) \\\ &\propto \pi_c \ p(x|y=c, \theta)  \\\ &\propto \pi_c \ \mathcal{N}(x|\mu_c, \Sigma_c)\end{aligned} \tag{3}
$$


where $\pi_c$ is the probability of choosing a class $c$ at random, or as Murphy[1] describes it as our prior belief about a target $c$. We will return to equation (3) later when we try to understand the shape of the decision boundary of LDA and QDA.

Now back to our Gaussians, we have 3 different classes that form 3 different clusters (black, red and blue), for **each cluster** we can approximate $\mu_c$ and $\Sigma_c$ through maximum likelihood estimation as:

$$ \hat{\mu}_c = \frac{1}{N_c} \sum^N _ {n=1} x_n  \tag{4}$$ 
$$ \hat{\Sigma}_c = \frac{1}{N_c} \sum^N _ {n=1} (x_n - \hat{\mu}_c) (x_n - \hat{\mu}_c)^\intercal \tag{5}$$

Now, we have three different Gaussians, where we can sample from each one (given the class) and generate new data points:
![alt text](/assets/img/mle_covariance_contours.png)

### Discriminator
From equation (1), we know that:

$$ p(y=c|x, \theta) \propto \overbrace{p(x|y=c, \theta)}^\text{Generator} \ \overbrace{p(y=c)}^{\text{Class Prior}} $$

And since we are assuming equal class prior $\pi_1 = \pi_2 = \pi_3 = \frac{1}{3}$, we can formulate our discriminator/classifier function $f(x): \mathbb{R}^2 \mapsto \mathcal{C}$ as:
$$ f(x) =  \underset{c}{\mathrm{argmax}} \log p(x|y=c, \theta) \tag{6}$$
$$ \log p(y=c|x, \theta) = \log{\pi_c} - \frac{1}{2}|2\pi \Sigma_c| - \underbrace{\frac{1}{2} (x - \mu_c)^\intercal \Sigma^{-1}_c (x - \mu_c)} _ \text{Quadratic with $x$} \tag{7}$$


And such model is called Quadratic Discriminator giving the name **Quadratic Discriminator Analysis (QDA)**, because as we see from equation (7), $f(x)$ is a quadratic function of $x$. 
And this is the decision boundary for our classifier.
![alt text](/assets/img/qda_decision_boundaries.png)

One of the issues of our QDA classifier is that it's clearly overfitting. but, if we force our covariance matrix $\Sigma_c$ to be shared across all classes $(\Sigma_c = \Sigma)$, defined as:
$$ 
\hat{\Sigma} = \frac{1}{N} \sum^C _ {c=1} \sum^N _ {n=1|c} (x_n - \hat{\mu}_c) (x_n - \hat{\mu}_c)^\intercal 
$$

We will get a discriminator function that's linear with respect to $x$, and our model now is called Linear Discriminator, giving the name **Linear Discriminator Analysis**. 


Let's again go back to our discriminator function in equation (6) and expand the log of the posterior $\log p(y=c \mid x, \theta)$:

$$ 
\begin{aligned}\log p(y=c|x, \theta) &= \overbrace{\log{\pi} - \frac{1}{2} \log |2\pi \Sigma|} ^ {\text{const. for all classes}} - \frac{1}{2} (x - \mu_c)^\intercal \Sigma^{-1} (x - \mu_c) \\\ &=\text{const} - \frac{1}{2}\mu_c^\intercal \Sigma^{-1} \mu_c + \underbrace{x^\intercal \Sigma^{-1} \mu_c} _ {\text{linear with $x$}} - \underbrace{\frac{1}{2}x^\intercal \Sigma^{-1} x}_{\text{const. for all classes}} \end{aligned}  \tag{8}
$$


From equation (8), it's evident that our discriminant function in the case of LDA depends on $x$ linearly, and our decision boundary will be linear. And this is the decision boundary for such case:
![alt text](/assets/img/lda_decision_boundaries.png)

Which seems as if we applied regularization to our classifier/discriminator, and now our classifer can generalize better.

## Summary
1. We discussed difference between discriminative and generative models. 
2. In Gaussian discriminative analysis (GDA), we fit a multivariate Gaussian for each class cluster $\mathcal{N}(x \mid \mu_c, \Sigma_c)$.
3. Maximum likelihood estimation of $\mu_c$ and $\Sigma_c$ of the generator distribution results in a quadratic discriminator (prone to over fitting).
4. Forcing a tied covariance matrix results in a linear discriminator.
5. Generative models are powerful models, that can be used for imputing missing values or as a prior knowledge for further probabilistic analysis [2].

## References
<a id="1">[1]</a> Kevin P. Murphy (2021) - Probabilistic Machine Learning: An Introduction - The MIT Press.

<a id="1">[2]</a> [Volodymyr Kuleshov - Applied Machine Learning Course - Cornell Tech.](https://www.youtube.com/playlist?list=PL2UML_KCiC0UlY7iCQDSiGDMovaupqc83)
