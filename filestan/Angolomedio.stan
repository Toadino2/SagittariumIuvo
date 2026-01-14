data {
  int<lower=1> N;             // Number of observations
  array[N] real y;                // Observed angles in radians (between -pi and pi)
  real mu0;                   // Prior mean direction (in radians)
  real<lower=0> kappa0;       // Prior concentration for mu
  real<lower=0> alpha;        // Shape parameter for Gamma prior on kappa
  real<lower=0> beta;         // Rate parameter for Gamma prior on kappa
}

parameters {
  real<lower=-pi(), upper=pi()> mu;    // Latent mean direction
  real<lower=0> kappa;                 // Concentration parameter
}

model {
  // Prior on mu: von Mises with mean mu0 and concentration kappa0
  mu ~ von_mises(mu0, kappa0);

  // Prior on kappa: Gamma distribution
  kappa ~ gamma(alpha, beta);

  // Likelihood: von Mises distribution for the observations
  y ~ von_mises(mu, kappa);
}
