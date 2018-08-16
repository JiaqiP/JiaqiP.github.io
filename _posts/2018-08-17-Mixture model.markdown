---
layout:     post
title:      "Mixture Model Exploration"
subtitle:   ""
author:     "Jiaqi"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags: -data science
      -GE_intern

---

## 1. What is Mixture Models?


> Mixture model: a model in which the latent variable takes on one of a finite, unordered set of values. This latent variable represents unobserved subpopulations present in the sample.[^cs.princeton]

![The quick formula for mixture model][1]
file:///C:/Users/dell/Documents/Tencent%20Files/1151664945/FileRecv/MobileFile/IMG_8955.JPG


Variables' quick reference:

| Variable   |Meaning   | Restriction |
| --------   | :-----:  | :----:  |
|K  |number of components | constant
| z     | latent variable: [0 .. 0 1 0 .. 0] |$z$ is to describe each sample:    $z$ ~ Multinomial ($\phi$);  $z_j$ ~ Multinomial($w_j$) |
| $\phi$ |   prior probability of each mixture component   |   $\sum_{i=0}^k \phi^i =1$   |
|i /k  |the $i^{th}$ component |
|j   |the $j^{th}$ sample |
|d   | the $d^{th}$ dimention of a multivaritate distribution
|D  | total number of dimentions for a multivariate distribution | constant
|n/N  |total number of data points or samples | constant
| $\theta_i$        |    set of variables for $i^{th}$ component distribution |   |
|$x_j$ / $x$   | each sample point|
|$w_j^i$ / $\gamma(z_j^i)$ | weight/ responsibility of sample $x_j$ assigned to cluster i | expectation of the assignment variable z; $w_j^i = P(z_j^i=1|x_j,\hat\theta) = E(z_j^i|x_j,\hat\theta)$|
|M| total categories for multinomial distribution|
|m| the $m^{th}$ category in multinomial mixture model |


- Mixture proportion and mixture component 
For a single sample $x$, the likelihood of it coming from the mixture model parameterized by $\theta$ **equals** the likelihood of $x$ coming from each distribution (*mixture component*) $\times$ (*mixture proportion*) $\phi$ 
$$P(x|\theta) = \sum_{i=1}^{k} P(x|z^i=1,\theta_i)P(z^i=1|\phi_i)= \sum_{i=1}^k \phi_iP(x|z^i=1,\theta_i)$$

$$P(x|\theta,\Phi) = \sum_{i=1}^{k} P(x|z^i=1,\theta_i)P(z^i=1|\phi_i)= \sum_{i=1}^k \phi_iP(x|z^i=1,\theta_i)$$

$\color{DarkTurquoise}{I\ am\ here \ to\ try\ some\ color}$ 

 - Clustering:
If you get a new data sample $x^*$, decide it should go to which component: write out the posterior and apply Beyesian rule
$$P(z^i=1|x^*,\hat\theta) = \frac{P(x^*|z^i=1,\hat\theta)\phi_i}{\sum_k\phi_k P(x^*|z^k=1,\hat\theta)} $$  

$$P(z^i=1|x^*,\hat\theta) = \frac{P(x^*|z^i=1,\hat\theta)\phi_i}{\sum_k\phi_k P(x^*|z^k=1,\hat\theta)} $$  


 - Likelihood
 The likelihood (L) and the log likelihood ($l$) of a set of data $N =    \{x_1, . . . , x_n\}$ are as follow:
 $$L(N|\theta) = \prod_{j=1}^n P(x_j|\theta)= \prod_{j=1}^n (\sum_{i=1}^k \phi_i P(x_j^i|\theta_i,z^i=1)$$

 $$l(N|\theta) = \sum_{j=1}^n log(\sum_{i=1}^k \phi_i    P(x_j^i|\theta_i,z^i=1) = \sum_{j=1}^n log(\sum_{i=1}^k \phi_i P(x_j^i|\theta_i))$$  
 
 

 (leave out $z_i$, as it doesn't make a difference in calculation and is easier to understand)  

 The parameters to estimate are $\theta$ and $\phi$. Solve by minimizing the log likelihood.  
    


----------

## 2. How to Solve Mixture Model?

Try to take the derivative of log-likelihood with respect to one parameter $\theta_m$ (the $m^{th}$ component). Note: $f(x_j;\theta_m) = P(x_j^i|\theta_i)$

$$\frac{\partial l}{\partial \theta_m} = \sum_{j=1}^{n}\frac{1}{\sum_{i=1}^k\phi_i f(x_j;\theta_i)} \phi_m \frac{\partial f(x_j;\theta_m)}{\partial \theta_m}$$

$$=\sum_{j=1}^{n}\frac{\phi_m f(x_j;\theta_m)}{\sum_{i=1}^k\phi_i f(x_j;\theta_i)} \frac{1}{f(x_j;\theta_m)} \frac{\partial f(x_j;\theta_m)}{\partial \theta_m}$$

$$=\sum_{j=1}^{n}\frac{\phi_m f(x_j;\theta_m)}{\sum_{i=1}^k\phi_i f(x_j;\theta_i)} \frac{\partial( log f(x_j;\theta_m))}{\partial \theta_m}$$

$$= \sum_{j=1}^n the\  weight\ of\ x_j\ assigned\ to\ the\ m^{th} component * derivative\ of\ ordinary\ log\ likelihood$$

We are doing weighted maximum likelihood, with weights given by posterior cluster probablities. These to repeat, depend on the parameters ($\theta$) we are trying to estimate. ---- A vicious circle. [^stats.cmu] 


### Introduction to EM Algorithm
Alternating between assigning points to clusters and finding cluster centers. ---- K-Means
Find out which component each point is most likely to have come from; re-estimate the components using only the points assigned to it. ----EM

*Algorithm steps:*

|Steps| K-Means   |EM   |
|:--- |:--------   | :-----|
|1  |set k; choose initial centroids| set k; initialize $\theta_1...\theta_k$, $\phi_1...\phi_k$|
|2  |assign data $x_j$ to a cluster i: $z_j^i=1$ by a distance function|**E:**  Using the current parameter guesses, calculate the weight (soft assignment variable $w =\gamma(z)$ : *the posterior probablity of $x_j$ in each cluster* )
|3  |calculate the positions of centroids $\mu_i=\frac{\sum_{j=1}^n(z_j^ix_j)}{\sum_{j=1}^n(z_j^i)}$|**M:** Using the current weights, maximize the weighted likelihood to get new parameter estimates. ---- For each component, take the sample by portion assigned to it, then do parameter estimation as in basic distributions. And then recalculate $\phi$ $\color{Red}{why\ \phi\ calculated\ this\ way}$  
|4  |iterate 2 and 3 until the assignment, z, no longer change| Iterate 2 and 3 until the log-likelihood is below some threshold. Return the final parameter estimates (including mixing proportions $\phi$) and cluster probabilities: $w$ for each sample

*Charasteristics:*

| K-Means   |EM   |
|:--------   | :-----|
|largely dependent on initial assignment, no guarantee|guarentee to converge to local optimal 

### Calculation of EM algorithm (examplified with Gaussian Mixture)
1. Initialize the $\theta_k$:  $( \mu_k, \Sigma_k),\phi_k$ and evaluate the initial value of the log likelihood.
2. **E step** Evaluate the responsibilities/ weight using the current parameter values
$$\gamma (z_j^k) = \frac{\phi_k N(x_j|\mu_k,\Sigma_k)}{\sum_{i=1}^K\phi_i N(x_j|\mu_i,\Sigma_i)}$$
$\color{Red}{connection\ to\ naive\ bayes?}$  
$\color{Red}{Yes！There\ is\ mixtures\ of\ naive\ bayes}$  

3. **M step** Re-estimate the parameters using the current responsibilities
$N_k = \sum_{j=1}^n \gamma(z_j^k)$
$$ \mathbf{\mu_k^{new}} = \frac{1}{N_k}\sum_{j=1}^n \gamma(z_j^k) 、\mathbf{x_j}$$ 

$$ \mathbf{\Sigma_k^{new}} = \frac{1}{N_k} \sum_{j=1}^n \gamma(z_j^k)(\mathbf{x_j-\mu_k^{new}})(\mathbf{x_j-\mu_k^{new}})^T$$

$$\phi_k^{new} = \frac{N_k}{N}$$
4. Evaluate the log likelihood 
 $$l(\mathbf{x_1 \ldots x_n |\mu,\Sigma,\phi}) = \sum_{j=1}^n log(\sum_{i=1}^k \phi_i    P(x_j^i|\theta_i,z^i=1) = \sum_{j=1}^n log(\sum_{i=1}^k \phi_i N(\mathbf{x_j|\mu_k,\Sigma_k})))$$ 

And check for convergence of either the parameters or the log likelihood.

### Bonous: Proof and Details about EM Algorithm
[Read section: more about EM algorithm ][2]

----------


## 3. (Multivariate) Berboulli/ Multinomial Mixture models [^bishop]
https://users.ics.aalto.fi/jhollmen/Publications/courseware.pdf
[mixture model summary][3]

### Bernoulli  Mixture model:
- parttern recognition: hand-written letter recognition
http://users.dsic.upv.es/~ajuan/research/2004/Juan04_08b.pdf
- finding motifs in sequence data (eg. DNA)  
<br>

> **Consider a multivariate Bernoulli distribution:**  
A set of D discrete variables $x_d$, where d = 1, ... D, each of which is
> governed by a Bernoulli distribution with parameter $\mu_i$, so that
 $p(x_1,\ldots x_D|\mu_1 \ldots \mu_D) = \prod_{d=1}^{D}
 \mu_d^{x_d}(1-\mu_d)^{(1-x_d)} $ 
 E[**x**] = **$\mu$** 
 cov[**x**] = diag{$\mu_d(1-\mu_d)$}

$\color{DarkTurquoise}{------}$ 

- Be careful, for a multivariate distribution, the mean is a d-dim vector and covariance is an d by d matrix. 

<font color=blue> **A Problem:** cov here can't really capture the correlation between different dimentions. (If we consider each $x_d$ represents a pixel in a picture and therfore the chance of getting all pixels taking on values exactly the same as they are in this picture is a multinomial Bernoulli distribution)
</font>  
**Solve:** To capture the interpixel correlation: consider a finite mixture of these distributions given by $p(x|\mu,\phi) = \sum_{k=1}^K \phi_k p(x|\mu_k)$

The mean and covariance of this mixture distribution are given by:
E[**x**] = $\sum_{k=1}^K \phi_k\mu_k$
cov[**x**] = $\phi_k \{ \Sigma_k + \mu_k\mu_k^T\} - E[x]E[x]^T$ where $\Sigma_k = diag\{\mu_{kd} (1-\mu_{kd})\}$

$\color{Red}{Task}$  
[*Task: Review and check cov matrix calculations for general and Bernoulli and Multinomial distribution*]

**Problem solved:** The cov matrix is no longer diagonal so the mixture distribution can capture correlations between variables unlike single Bernoulli distribution.
  
  
### Bernoulli Mixture Calculation EM:
Derive the EM algorithm with the below log likelihood for mixture of Bernoulli distributions
$$ln p(X|\mu,\phi) = \sum_{j=1}^nln\{ \sum_{k=1}^K \phi_kp(X_n|\mu_k) \}$$


The complete data likelihood:
$$p(x,z|\mu,\phi) = \prod_{j=1}^n \sum_{k=1}^K (\phi_k z_{jk}\prod_{d=1}^D(\mu_{kd}x_{jd} + (1-\mu_{kd})(1-x_{jd})))$$


TRICK: $ln (\sum_{k=1}^K z_j^k (*)) = \sum_{k=1}^K z_j^k ln(*)$ since only one $z^k$ = 1 and 0 elsewhere  

The complete data log likelihood:
$$ln \ p(x,z|\mu,\phi) = \sum_{j=1}^n \sum_{k=1}^K (z_{jk} \{ln\ \phi_k +\sum_{d=1}^D(ln(\mu_{kd})x_{jd} + ln(1-\mu_{kd})(1-x_{jd})))$$

Take expectation of the complete-data log likelihood with respect to posterior distribution of latent variables $\mathbf{z}$:
$$E_z[ln \ p(x,z|\mu,\phi)  =  \sum_{j=1}^n \sum_{k=1}^K (\gamma (z_{jk}) \{ln\ \phi_k +\sum_{d=1}^D(ln(\mu_{kd})x_{jd} + ln(1-\mu_{kd})(1-x_{jd})))$$ 

where $\gamma (z_{jk}) = E[z_{jk}]$ 

**E step:**
$$\gamma(z_{jk}) = \frac{\sum_{z_{jk}}z_{jk}[\phi_k p(x_j|\mu_k)]^{z_{nk}}}{\sum_{z_{ni}}[\phi_i p(x_j|\mu_i)]^{z_{ni}}}$$

$$= \frac{\phi_k p(x_j|\mu_k)}{\sum_{i=1}^K \phi_i p(x_j|\mu_i)}$$




Consider sum over j in $E_z$ equation, responsibilities enter only through two terms:
$$N_k = \sum_{j=1}^n \gamma(z_{jk})$$

$$\mathbf{\bar x_k} = \frac{1}{N_k} \sum_{j=1}^n \gamma(z_{jk}) \mathbf{x_j} $$

**M step:** Maximize the expected complete-data log likelihood with respect to $\mu_k$ and $\phi$ by setting the derivative w.r.t to $\mu_k$ *That's why the lengthy $E_z[ln \ p(x,z|\mu,\phi)$ need to be calculated*

Obtain $$\mathbf{\mu_k} = \mathbf{\bar x_k}$$

w.r.t. $\phi$, a Lagrange multiplier to enforce constraint $\sum_k \phi_k =1$

Obtain $$\phi_k = \frac{N_k}{N}$$




### Extend to Multinomial Mixture Model:

> Consider a D-dimensional variable **x** each of whose components i is itself a multinomial variable of degree M so that **x** is a binary vector with components $x_{dm}$ where d = 1,...,D and m = 1,...,M,subject to the constraint that $\sum_m x_{dm} = 1$ for all i. Suppose that the distribution of these variables is described by a mixture of the discrete multinomial distributions so that
 $$ p (\mathbf{x}) = \sum_{k=1}^K \phi_k p(\mathbf{x}|\mathbf{\mu_k})$$
where $$p(\mathbf{x}|\mathbf{\mu_k}) = \prod_{d=1}^D\prod_{m=1}^M \mu_{kdm}^{x_{dm}}$$

>Note: 
**x** = [$\mathbf{x_1},...\mathbf{x_D}$] and $\mathbf{x_d} = [x_{d1},...x_{dM}]^T$
$\mu_{kdm} = p(x_{dm} = 1 | \mu_k)$
$\mu_k = [\mu_{k1} ... \mu_{kD}]$ and $\mu_{kd} = [\mu_{kd1},...,\mu_{kdM}]$ constrained by
0 $<=\mu_{kdm} <=$ 1 and $\sum_m\mu_{kdm} = 1$（to ensure multinomial distribution for each component & dimention）

#### Derive mixing coefficients $\phi_k$ and the component parameters $\mu_{kdm}$:
1. Latent variable $Z_n$ corresponding to each observation.
$\color{Red}{TODO:\ Derivation\ not\ understood}$  

**E step**
$$\gamma(z_{nk}) = E[z_{nk}] = \frac{\phi_kp(x_n|\mu_k)}{\sum_{k=1}^K \phi_k p(x_n|\mu_k)}$$

$$where \ \ p(\mathbf{x_n}|\mathbf{\mu_k}) = \prod_{d=1}^D\prod_{m=1}^M \mu_{kdm}^{x_{ndm}}$$

**M step**
$$N_k = \sum_{n=1}^N \gamma(z_{nk})$$
$$\mu_{kdm} = \frac{1}{N_k}\sum_{n=1}^N\gamma(z_{nk}x_{ndm})$$
$$\phi_k = \frac{N_k}{N}$$


For clustering tasks, the $\gamma(z_{n}) = [\gamma(z_{1})...\gamma(z_k)]$ is used.


----
### Clustering with Both Categorical and Numeric Values
Difference in EM lies in $\theta$, and $p(x_n|\theta_k)$.

General Form:
**E-step**
Find $\gamma(z_{nk}) = \frac{\phi_k p(x_n|\theta_k)}{\sum_{k=1}^K \phi_k p(x_n|\theta_k)}$
**M-step**
Estimate parameter $\theta$ using maximum likelyhood estimation

Details
**E-step**
General:
$$\gamma(z_{nk}) = \frac{\phi_k p(x_n|\theta_k)}{\sum_{k=1}^K \phi_k p(x_n|\theta_k)}$$


Assume there is no dependency between numeric columns and categorical columns.
$$p(x_n|\theta_k) = p(x_n|\theta_{k,D_{c}}) p(x_n|\theta_{k,D_{n}})$$




For Gaussian Distribution independently in different dimensions:
$$p(x_n|\theta_k) = \prod^{D_c} \prod_{m=1}^M \mu_{kdm}^{x_{ndm}} \  \prod^{D_n} p(x_n| \mu_{kd}, \sigma_{kd})$$ 
If the numeric dimensions are modeled by multivariate Gaussian Distribution:
$$p(x_n|\theta_k) = (\prod^{D_c} \prod_{m=1}^M \mu_{kdm}^{x_{ndm}} )\  N(x_n| \mu_{kD_n}, \Sigma_{kD_n})$$  

**M-step**
$$N_k = \sum_{i=1}^n \gamma(z_i^k)$$
Gauusian:
$$\mu_{kd} = \frac{1}{N_k} \sum_{i=1}^N \gamma(z_i^k) x_{nd}$$
$$\sigma_{kd} = \frac{1}{N_k} \sum_{i=1}^N \gamma(z_i^k) (x_{id}-\mu_{id})(x_{id}-\mu_{id})^T$$

Multinomial:
$$\mu_{kdm} = \frac{1}{N_k} \sum_{i=1}^N \gamma(z_i^k) x_{ndm}$$

$$\phi_k = \frac{N_K}{N}$$

----

Look for variance and covariance http://wcms.inf.ed.ac.uk/ipab/rlsc/lecture-notes/RLSC-Lec3.pdf
Image: perhaps https://pdfs.semanticscholar.org/6b8e/8ffc6d9ef96fe61bcc92152e1f63ed4c0d59.pdf


----------


## 4. How to Code Mixture Models?
[Python pomegrante library][4]
[R mixtools: mixture of multinomials][5]

## 5. Extentions of Mixture Models

### Conjugate Prior: To get a reasonable prior $p(\theta)$
Introducing prior distribution: similar as getting more valid observations of x, maximize prior probablity w.r.t $\theta$
Then maximize posterior probablity (likelihood) of $\theta$
$p(\theta|x)$ in same distribution family with $p(\theta)$

|Distribition    | Conjuate Prior | Process Involved
|:-- |:-- |:--
|Berboulli | Beta |
|Multinomial | Dirichlet | MCMC
|Gaussian |  |




[^cs.princeton]:http://www.cs.princeton.edu/~bee/courses/lec/lec_feb12.pdf

[^\.cmu]:http://www.stat.cmu.edu/~cshalizi/350/lectures/29/lecture-29.pdf

[^bishop]:http://www.rmki.kfki.hu/~banmi/elte/bishop_em.pdf



<br>
Resourses:

 1. http://home.iitk.ac.in/~apanda/cs300/5.pdf

 5. https://www.mrc-bsu.cam.ac.uk/wp-content/uploads/EM_slides.pdf
 6. https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
 7. file:///C:/Users/dell/Desktop/10.1007%252F978-0-387-35768-3.pdf
 

  [1]: file:///C:/Users/dell/Documents/Tencent%20Files/1151664945/FileRecv/MobileFile/IMG_8955.JPG
  [2]: http://www.stat.cmu.edu/~cshalizi/350/lectures/29/lecture-29.pdf
  [3]: file:///C:/Users/dell/Desktop/10.1007%252F978-0-387-35768-3.pdf
  [4]: https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_1_Distributions.ipynb
  [5]: https://www.rdocumentation.org/packages/mixtools/versions/1.0.4/topics/multmixEM
  [6]: Image:%20http://www.cse.psu.edu/~rtc12/CSE586/lectures/cse586samplingPreMCMC.pdf