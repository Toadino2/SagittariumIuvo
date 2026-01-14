data {
  int<lower=1> G;                           // Number of prior groups
  array[G] int<lower=1> N;                 // Observations per prior group
  array[sum(N)] vector[2] y_prior;         // Flattened prior observations
  array[sum(N)] int<lower=1> prior_group;  // Group index per prior observation

  int<lower=1> N_current;                  // Current data size
  array[N_current] vector[2] y;            // Current observations
}

parameters {
  // Hyperparameters for prior means
  vector[2] mu_hyper;
  cov_matrix[2] lambda_hyper;

  // Group-level prior means and covariances
  array[G] vector[2] mu_prior;
  array[G] vector<lower=0>[2] sigma_prior;
  array[G] corr_matrix[2] R_prior;

  // Parameters for current dataset
  vector[2] mu;
  vector<lower=0>[2] sigma;
  corr_matrix[2] R;
}

transformed parameters {
  array[G] cov_matrix[2] Sigma_prior;
  cov_matrix[2] Sigma;

  for (g in 1:G) {
    Sigma_prior[g] = quad_form_diag(R_prior[g], sigma_prior[g]);
  }

  Sigma = quad_form_diag(R, sigma);
}

model {
  // Hyperpriors
  mu_hyper ~ multi_normal(rep_vector(0, 2), diag_matrix(rep_vector(1, 2)));
  lambda_hyper ~ wishart(3, diag_matrix(rep_vector(1, 2)));  // Can be adjusted

  // Prior group-level parameters
  for (g in 1:G) {
    mu_prior[g] ~ multi_normal(mu_hyper, lambda_hyper);
    sigma_prior[g] ~ gamma(4, 3);
    R_prior[g] ~ lkj_corr(1);
  }

  // Likelihood for prior data
  for (n in 1:sum(N)) {
    int g = prior_group[n];
    y_prior[n] ~ multi_normal(mu_prior[g], Sigma_prior[g]);
  }

  // Current-level priors
  mu ~ multi_normal(mu_hyper, lambda_hyper);
  sigma ~ gamma(4, 3);
  R ~ lkj_corr(1);

  // Likelihood for current data
  for (n in 1:N_current) {
    y[n] ~ multi_normal(mu, Sigma);
  }
}
