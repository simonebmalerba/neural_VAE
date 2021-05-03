# Efficient Coding

## Woodford model


VAE Loss (to be MINIMIZED)
$$
\mathcal{L} = E_{\pi(x)} [- \int dr p_\theta(r|x) \log(q_\phi(x|r)) + \beta \int dr p_\theta(r|x)\log\frac{q_\phi(x|r)}{q_\psi(r)}]
$$
Where: $x$ stimulus --> (1) dimensional
$r$ neural response --> N dimensional
$\phi$ encoder parameters --> $p_\phi(r|x)$
$\theta$ decoder parameters --> $q_\theta(x|r)$
$\psi$ prior parameters --> $q(r)$ 

Choice:
Prior:  $q(r) \sim Cat(N)$ parametrized by $N$ discrete numbers, such that $q(r_j=1)\equiv q(j) = q_j $ with $j=1,...,N$ .
Decoder: $q(x|j) \sim \mathcal{N}(\mu_j,\sigma_j) $ 
Encoder: functional derivative of loss, obtaining $p_\theta(j|x) = \frac{q(j)q(x|j)^{1/\beta}}{Z_x}$ where $Z_x = \sum_j q(j)p(x|j)^{1/\beta}$  , such that we can fit everything with EM.
Propose: parametrize $p_\theta(j|x) = \frac{\exp(a_jx^2+b_jx +c_j)}{\tilde{Z_j}}$ ...similar results?