data {
    int<lower=1> N;
    array[N] vector[2] y;
    vector[2] mu0;
    cov_matrix[2] lambda0;
    real a0;
    real b0;
    real<lower=0> eta;
}

parameters {
    vector[2] mu;
    vector<lower=0>[2] sigma;
    corr_matrix[2] R;
}

transformed parameters {
    cov_matrix[2] Sigma;
    Sigma = quad_form_diag(R, sigma);
}

model {
    mu ~ multi_normal(mu0, lambda0);
    sigma ~ gamma(a0, b0);
    R ~ lkj_corr(eta);
    for (n in 1:N){
        y[n] ~ multi_normal(mu, Sigma);
    }
}