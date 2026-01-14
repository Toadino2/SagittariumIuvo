data {
    int<lower=1> N;
    array[N] vector[2] dati;
    vector[2] mu0;
    cov_matrix[2] Sigma0;
    real<lower=0> a;
    real<lower=0> b;
    real eta;
}

parameters {
  vector[2] mu;
  vector<lower=0>[2] sigma;
  corr_matrix[2] L;
}

transformed parameters {
  matrix[2,2] L_Sigma;
  L_Sigma = quad_form_diag(L, sigma);  // Cholesky of full covariance
}

model {
  mu ~ multi_normal(mu0, Sigma0);
  sigma ~ gamma(a, b);       // weakly informative prior
  L ~ lkj_corr(eta);     // flat prior on correlation

  for (n in 1:N){
    dati[n] ~ multi_normal(mu, L_Sigma);
  }
}