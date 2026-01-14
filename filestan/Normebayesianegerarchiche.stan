data {
  int<lower=1> G;                       // number of groups (prior + current)
  array[G] int<lower=1> N;             // observations per group
  array[sum(N)] real<upper=10> y;                // all observations (flattened)
  array[sum(N)] int<lower=1> group_id; // group index per observation
  int<lower=1> Ncorrente;
  array[Ncorrente] real<upper=10> ycorrente;
}

parameters {
  real<lower=0> alfatheta;
  real<lower=0> betatheta;
  real<lower=0> alfasigma;
  real<lower=0> betasigma;
  array[G] real<lower=0, upper=1> theta;
  array[G] real<lower=0> sigma;
  real<lower=0, upper=1> thetacorrente;
  real<lower=0> sigmacorrente;
}

transformed parameters {
  array[G] real mu;
  for (g in 1:G) {
    mu[g] = 10 * theta[g];
  }
  real mucorrente;
  mucorrente = 10*thetacorrente;
}

model {
  alfatheta ~ gamma(0.5, 0.5);
  betatheta ~ gamma(0.5, 0.5);
  alfasigma ~ gamma(0.5, 0.5);
  betasigma ~ gamma(0.5, 0.5);
  for (g in 1:G){
    theta[g] ~ beta(alfatheta, betatheta);
    sigma[g] ~ gamma(alfasigma, betasigma);
  }
  for (i in 1:sum(N)){
    int g = group_id[i];
    y[i] ~ normal(mu[g], sigma[g]);
  }
  thetacorrente ~ beta(alfatheta, betatheta);
  sigmacorrente ~ gamma(alfasigma, betasigma);
  ycorrente ~ normal(mucorrente, sigmacorrente);
}
