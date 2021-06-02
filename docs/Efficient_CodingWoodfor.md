# Efficient Coding Models -- VAE approach

## General loss function to consider

$$
\mathcal{L} = E_{\pi(x)} [- \int dr p_\theta(r|x) \log(q_\phi(x|r)) + \beta \int dr p_\theta(r|x)\log\frac{q_\phi(x|r)}{q_\psi(r)}]
$$
Where: $x$ stimulus --> (1) dimensional
$r$ neural response --> N dimensional
$\phi$ encoder parameters --> $p_\phi(r|x)$
$\theta$ decoder parameters --> $q_\theta(x|r)$
$\psi$ prior parameters --> $q(r)$ 

Several interpretations of this loss function. Most appealing, two bounds on the MI between r and x, to be explored further (see Rava's talk).

## Woodford choice

Choice:
Prior:  $q(r) \sim Cat(N)$ parametrized by $N$ discrete numbers, such that $q(r_j=1)\equiv q(j) = q_j $ with $j=1,...,N$  -> $\int dr -> \sum_j$
Decoder: $q(x|j) \sim \mathcal{N}(\mu_j,\sigma_j) $ 
Encoder: functional derivative of loss, obtaining $p_\theta(j|x) = \frac{q(j)q(x|j)^{1/\beta}}{Z_x}$ where $Z_x = \sum_j q(j)p(x|j)^{1/\beta}$  .

Indeed, substituing the expression in the loss function, we obtain that we want to minimize
$$
\mathcal{L} = -E_{x\sim \pi(x)}[\log Z_x] = -E_{x\sim \pi(x)}[\log (\sum_j q_jq(x|j)^{1/\beta})],
$$
which is, a part from the exponent, nothing else than the likelihood of points under a Gaussian mixture model. This can be fitted with EM.
Todo: 

* try to recover similar results with SGD. It may be worth considering already Eq.(2), as it simplify some terms.

* parametrize $p_\theta(j|x) = \frac{\exp(a_jx^2+b_jx +c_j)}{\tilde{Z_x}}$ ...similar results?

## Going beyond

Choose a more general space for neural activities.
Bernoulli encoding model
$$
p(r|x) = \Pi_j p(r_j|x) \\
p(r_j=1|x) \sim Bernoulli(\frac{1}{1 + u_j(x)^{-1}}) \\
u_j(x) = A_j \exp(-\frac{(x-c_j)^2}{2\sigma_j^2})
$$
Using the exponential family notation, this can be written as $p(r_j|x) = \exp\left(\eta_j(x)r - \log(1+e^{\eta_j(x)})\right)$ with $\eta_j(x) = -(x-c_j)^2/2\sigma_j - log(A_j)$ (quadratic function)

Interpretation: bell-shaped function outputing the probability of spiking. It has the following advantage: encoder probability can be written as
$$
p(r|x) =\exp(\eta(x)r - \sum_j\log(1+e^{\eta_j(x)}) )
$$
Approximation $\sum_j \log(1 + e^{\eta_j(x)})$ is independent from x --> to verify, usualli it is assumed $\sum_ju_j(x) $ indep from x.

Inverting the relationship, one realize
$$
p(x|r) \propto p(r|x)p(x) \propto \exp((x,x^2)^T\Theta r - \sum_j(...) + \log(\pi(x)))
$$
IF the sum is independent from x, and $log(\pi(x)) \propto \exp(ax^2 + bx) $  depends only on the first and second moment, we can recognize the form of a Gaussian, with *natural* parameters $\Theta_1*r + a, \Theta_2*r + b$ . From the natural parameters, it is possible to obtain the mean and variance of the gaussian function.

Choice of the prior:

E.g. (1): Ising model prior
$$
q_{Ising}(r)  \propto \exp(hr + r^TJr)
$$

* hard to compute the Dkl in a closed form, as the normalization is not straightforward
* hard to sample from
* ...

E.g. (2): mixture of posterior distribution.  From the minimization of Eq. (1), given a decoder, the optimal prior is the marginal posterior
$$
q^*(r) = \int dx p(r|x)
$$
Several studies (insert refs, look for VAMP prior) considered to approximate this optimal prior as a finite mixture:
$$
q_{VAMP}(r) = \sum_k \frac{1}{K} p(r|x_k)
$$
and suggest to find also $x_k$ with Gradient descent. Note that within this formulation, the Dkl term is somehow computable as 
$$
\sum_r p(r|x) \log\frac{p(r|x)}{q(r)} \approx \log(\sum_k \frac{1}{K} \exp(-Dkl(p(r|x)||p(r|x_k)))
$$
with the advantage that the Dkl in the exponents are Dkl between bernoulli probabilities, so they have an analytical form.  In this way, we  eliminate beta, as  we may simly vary the number of mixtures K.		