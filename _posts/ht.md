---
layout: post
title: "Bayesian A/B Testing: Part-I the frequentist way"
categories: journal
tags: [probabilistic,bayesian]
image: mu-emp-dist.png
---

*Our roadmap in Part I is to provide a quick introduction to hypothesis testing in the framework of frequentist approach, by introducing test statisitc, p-values and A/B testing, and in Part-II we proceed with the Bayesian way.*

Let's suppose that we are a bunch of newly hired quality specialists in a factory producing welded pipes, and we were given the task of analyzing the daily percentage of defective pipes (with welding cracks). Our factory produces a huge amount of pipes, which makes checking 100% of the pipes impractical, yet we are given a random 20 pipes from production department for testing (at end of each day).

We were told by management that the estimated percentage of daily defective pipes is 15%. We take such information and proceed to verify it. In our case, we have two outcomes for each pipe, either it's defective or non-defective, we define a random variable $X: \Omega \mapsto \mathcal{T}$, where $\Omega$ is the sample space and $\mathcal{T}$ is the target space, defined as:

$$
\mathcal{T} := \{x = 0 \ \text{(non-defective)}, x = 1 \ \text{(defective)}\}
$$ 

We model the outcomes using Bernoulli distribution with parameter $\mu$ as the probability of a pipe being defective (we can also model this a Binomial problem):
$$ x \sim \text{Bernoulli}(x|\mu) $$

Next, we take a daily sample of 20 finished pipes, and we find out that $\mu = 0.12$, what should we do with the information we were provided by management that $\mu = 0.15$? do we reject it? 

Well, at first glance the bernoulli patameter we obtained from the sample is not a strong indication (not too obvious deviation) that management's parameter is wrong, given the nature of randomness in our problem, and knowing that presence of defective welds is subject to many factors, such as welders skill, filler material quality, heating and cooling rates and so on, each parameter with it's own factor of randomness. So, our first analysis did not reject management's parameter because the evidence is not too strong. 

We proceed for the next days and keep logging the daily percentage of defective pipes and find out that there's a fluctuation of the daily percentage rate (we logged the result of testing 600 pipes):

![mu observations](\assets/img/mu-emp-dist.png)

We observe that the bernoulli parameter $\mu$ (daily samples defective proportion) is normally distributed (backed up by central limit theorem), with a mean of $0.135$ and standard deviation $\hat{\sigma} = 0.035$.

So it seems that our initial decision of not rejecting management's assertion was not a bad decision, because at that time the sample statistic we got did not deviate that much. Or, as stated by Walpole et al. [1]:
> "Rejection of a hypothesis implies that the sample evidence refutes it [..] rejection means that there is a small probability of obtaining the sample information observed when, in fact, the hypothesis is true"

We also should note that (from a Bayesian point of view) we are testing a hypothesis about a parameter that we will never truly know, so rejecting a hypothesis must be based on a strong evidence, and when we reject a hypothesis, this decision is based on the sample evidence. And if the data do not provide sufficient evidence the hypothesis is not true, we fail to reject it (innocent until proven guilty).

# Hypothesis testing
We need to formalize our results so far, to prepare our statement to management. But before that we should list the types of error we could make in our results.

## Types of Errors
To demonstrate the type of errors in hypothesis testing, let's give another example.

Suppose we are working in a pharmaceutical company, where R&D comes up with a new drug, but we are skeptical about the safety of such drug. We state the *null hypothesis* as the drug being *unsafe*, we perform some experimenting and data collection to test the *alternate hypothesis* that the drug is *safe*.

There are two types of error we could encounter:
1. **Type I Error:** Reject a true null hypothesis.
2. **Type II Error:** Accept a false null hypotheses.

Given the previous example, we clearly see that Type I is the most dangerous. If our drug is really unsafe (null hypothesis is true) and we conclude that the drug is safe (we reject the true null and the accept alternative hypothesis), we are making a big mistake!

In order to quantitize such situations, we define $\boldsymbol \alpha$ (level of significance) as the probability of committing Type I error, let's return to our pipe factory example to understand $\alpha$:

Let us define a **test statistic** $T$ as the number of defective pipes in a 600 pipe sample, if $T < 30$ we reject the null hypothesis and if $T \ge 30$ we fail to reject the null hypothesis (treat this as if we are setting our own standard to reject or fail to reject the null hypothesis). In such case, $\alpha$ will be the ratio of number times we sample 600 pipes and we find out that $T < 30$ and we reject the null hypothesis when it's actually true. So, basically the test statistic is the critical value [2] based on which we reject or we fail to reject the null hypothesis in favor of the alternative (this will be more clear in the following examples).

That's a lot of words! let's illustrate the concepts we discussed so far by two interesting examples found on [Probability & Statistics for Engineers & Scientists](https://www.pearson.com/us/higher-education/product/Walpole-Probability-and-Statistics-for-Engineers-and-Scientists-9th-Edition/9780321629111.html) book.

### Average Life Span
We collected a random sample of 100 deaths with an average life span $\mu = 71.8$ years, and assuming $\sigma = 8.9$ years for the population, we ask the question: does this indicate that the average life span for the population is greater than 70 years? using a level of significance 0.05.

**Step 1: State null and alternative hypothesis**

$$
\begin{aligned} H_0: \mu = 70 \ \text{years} \\ H_1: \mu > 70 \ \text{years} \end{aligned} 
$$

This is called a one-sided hypothesis.

**Step 2: Calculate the test statistic $T$**
According to central limit theorem, if we have $n$ random samples drawn from a population with mean $\mu$ and standard deviation $\sigma$ the sample mean $\bar{x}$ is distributed as $ \mathcal{N}(\mu, \frac{\sigma}{\sqrt{n}})$.

Since we are given a level of significance $\alpha = 0.05$, and by definition:

$$
\alpha = P(\bar{x} > T) = 1 - \Phi(T) = 0.05
$$

where $\Phi$ is the cumulative distribution function (CDF), then:

$$ 
T = \Phi^{-1} (1-0.05)=71.46 \ \text{years}
$$ 

So, our test statistic is $71.46 \ \text{years}$, and if $\bar{x} > T$ we reject the null hypothesis.

**Step 3: Check the null hypothesis**
The mean of the sample we collected $\bar{x} = 71.8$ years, which is greater than $T$.

**Step 4: Decision**
Based on step 3, we reject the null hypothesis.

**Step 5: Calculate p-value**
Assuming the null hypothesis was true, what is the probability of obtatining our collected sample mean (at least as extreme)?

$$
\begin{aligned} p(\bar{x} > 71.8)  &= 1 - p(\bar{x} \le71.8)= 1- \Phi(71.8) \\\ &=1 - 0.9784 = 0.0261 \end{aligned}
$$

![p-value-1](\assets/img/example1-dist.png)

### Fishing Line Strength
We have been consulted by a company that developed new fishing line, and they claim that the mean breaking strength of the new line is $8 \  \text{kilograms}$ with standard deviation of $0.5 \ \text{kilograms}$, we collected a random sample of 50 lines with a mean breaking strength of $7.8 \ \text{kilograms}$. Using a level of significance of $0.01$.
**Step 1: State null and alternative hypothesis**

$$
\begin{aligned} H_0: \mu = 8 \ \text{kilograms} \\\ H_1: \mu \ne 8 \ \text{kilograms} \end{aligned} 
$$

This is called a two sided hypothesis.

**Step 2: Calculate the test statistic $T$**
Since we are given a level of significance $\alpha = 0.05$, and our alternative hypothesis is two sided, we write:

$$
\frac{\alpha}{2} = P(\bar{x} > T) = 1 - \Phi(T) = 0.005
$$

where $\Phi$ is the cumulative distribution function (CDF), then:

$$
\begin{aligned} T_{\text{max}} = \Phi^{-1} (1-0.005)=8.182\ \text{kilograms} \\\ T_{\text{min}} = \Phi^{-1} (0.005)=7.817 \ \text{kilograms}\end{aligned}
$$ 

So, we reject the null hypothesis if $T < 7.817$ or $T > 8.182$.

**Step 3: Check the null hypothesis**
The mean of the sample we collected $\bar{x} = 7.8$ kg, which is lower than $T_{\text{min}}$.

**Step 4: Decision**
Based on step 3, we reject the null hypothesis.

**Step 5: Calculate p-value**
Assuming the null hypothesis was true, what is the probability of obtatining our collected sample mean (at least as extreme)?

$$
\begin{aligned} p(\bar{x} < 7.8)  &= 1 - p(\bar{x} \le7.8)= 1- \Phi(7.8) \\\ &= 0.0023 \end{aligned}
$$

But because our alternative hypothesis is two-tailed, we need to multiply that value by two, so we conclude:
$$ 
\text{p-value} = 0.0046 
$$

![p-value-2](\assets/img/two-sided-ht.png)


## Summary
"A statistical hypothesis is an assertion or conjecture concerning one or more populations." [1], and unless we examine the whole population in question, we might not ever be absolutley certain about a hypothesis being true or not.

In this post, we tried to give a short introduction to a vast subject through practical examples, to get a good understanding of the procedure and the meaning behind big words like p-value, level of significance and critical regions.

Hope it helps!


## References
<a id="1">[1]</a> Probability & Statistics for Engineers & Scientists - Walpole, Myers, Myers and Ye - Ninth Edition - Pearson 2016

<a id="2">[2]</a> Probability & Statistical Inference- Hogg, Tanis and Zimmerman - Pearson 2014
