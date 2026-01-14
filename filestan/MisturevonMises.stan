data {
  int<lower=1> N;                      // Number of observations
  array[N] real y;                         // Observed angles (in radians)
  int<lower=1> K;                      // Number of mixture components
  real mu0;                            // Prior mean direction for mu[k]
  real<lower=0> kappa0;                // Prior concentration for mu[k]
  real<lower=0> alpha;                 // Shape parameter for Gamma prior on kappa[k]
  real<lower=0> beta;                  // Rate parameter for Gamma prior on kappa[k]
  vector<lower=0>[K] dirichlet_alpha; // Dirichlet prior parameters for mixture weights
}

parameters {
  simplex[K] theta;                   // Mixture weights
  vector<lower=-pi(), upper=pi()>[K] mu;    // Mean directions for each component
  vector<lower=0>[K] kappa;           // Concentrations for each component
}

model {
  // Priors
  theta ~ dirichlet(dirichlet_alpha);
  mu ~ von_mises(mu0, kappa0);
  kappa ~ gamma(alpha, beta);

  // Likelihood
  for (n in 1:N) {
    vector[K] log_lik;
    for (k in 1:K) {
      log_lik[k] = log(theta[k]) + von_mises_lpdf(y[n] | mu[k], kappa[k]);
    }
    target += log_sum_exp(log_lik);
  }
}
