data {
  int<lower=0> N;              // number of observations
  array[N] real y;                   // observed data
  real<lower=0> alpha;         // Beta prior parameter
  real<lower=0> beta;
  real<lower=0> alpha2;        // Gamma prior shape (for precision)
  real<lower=0> beta2;         // Gamma prior rate (for precision)
}

parameters {
  real<lower=0, upper=1> theta;    // latent beta variable
  real<lower=0> sigma;               // precision = 1 / sigma^2
}

transformed parameters {
  real mu;
  mu = 10 * theta;
}

model {
  theta ~ beta(alpha, beta);
  sigma ~ gamma(alpha2, beta2);
  y ~ normal(mu, sigma);
}