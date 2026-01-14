data {
  int<lower=1> G;                            // Number of groups
  array[G] int<lower=1> Npriori;                  // Observations per group
  array[sum(Npriori)] vector[2] datipriori;             // All data, flattened
  array[sum(Npriori)] int<lower=1> group_id;      // Group index for each obs
  int<lower=1> Ncorrente;
  array[Ncorrente] vector[2] daticorrenti;
}

transformed data {
  vector[2] mu0 = rep_vector(0.0, 2);
  vector[2] sigma0 = rep_vector(1.0, 2);
  cov_matrix[2] Sigma0 = diag_matrix(sigma0);
}

parameters {
  // Hyperparameters
  vector<lower=0>[2] a;
  vector<lower=0>[2] b;
  real<lower=0> eta;

  // Group-level parameters
  array[G] vector[2] mu;
  array[G] real<lower=0> sigma1;
  array[G] real<lower=0> sigma2;
  array[G] corr_matrix[2] L;

  vector[2] mucorrente;
  real<lower=0> sigmacorrente1;
  real<lower=0> sigmacorrente2;
  corr_matrix[2] Lcorrente;
}

transformed parameters {
  array[G] vector<lower=0>[2] sigma;
  for (g in 1:G){
    sigma[g][1] = sigma1[g];
    sigma[g][2] = sigma2[g];
  }
  array[G] cov_matrix[2] Sigma;
  for (g in 1:G){
    Sigma[g] = quad_form_diag(L[g], sigma[g]);
  }
  vector<lower=0>[2] sigmacorrente;
  sigmacorrente[1] = sigmacorrente1;
  sigmacorrente[2] = sigmacorrente2;
  cov_matrix[2] Sigmacorrente;
  Sigmacorrente = quad_form_diag(Lcorrente, sigmacorrente);
}

model {
  // Hyperpriors (can be adjusted or fixed if preferred)
  a ~ gamma(0.3, 1.0);
  b ~ gamma(1.0, 1.0);
  eta ~ gamma(0.25, 0.25);

  // Group-level priors
  for (g in 1:G) {
    mu[g] ~ multi_normal(mu0, Sigma0);
    sigma1[g] ~ gamma(a[1], b[1]);
    sigma2[g] ~ gamma(a[2], b[2]);
    L[g] ~ lkj_corr(eta);
  }

  // Likelihood
  for (i in 1:sum(Npriori)) {
    int g = group_id[i];
    datipriori[i] ~ multi_normal(mu[g], Sigma[g]);
  }

  mucorrente ~ multi_normal(mu0, Sigma0);
  sigmacorrente1 ~ gamma(a[1], b[1]);
  sigmacorrente2 ~ gamma(a[2], b[2]);
  Lcorrente ~ lkj_corr(eta);
  daticorrenti ~ multi_normal(mucorrente, Sigmacorrente);
}
