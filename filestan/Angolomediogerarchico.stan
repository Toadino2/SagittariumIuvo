data {
    int<lower=1> Ncorrente;
    array[Ncorrente] real angolicorrenti;
    int<lower=0> priori;
    array[priori] int<lower=1> Npriori;
    array[sum(Npriori)] int<lower=1> gruppipriori;
    array[sum(Npriori)] real angolipriori;
    real<lower=0> c;
    real<lower=0> d;
    real<lower=0> xi;
    real<lower=0> zeta;
}

parameters {
    real<lower=-pi(), upper=pi()> gamma;
    real<lower=-pi(), upper=pi()> m;
    real<lower=0> eta;
    real<lower=0> a;
    real<lower=0> b;
    real<lower=0> beta;
    array[priori] real<lower=-pi(), upper=pi()> mupriori;
    array[priori] real<lower=0> kappapriori;
    real<lower=-pi(), upper=pi()> mu;
    real<lower=0> kappa;
}

model {
    gamma ~ uniform(-pi(), pi());
    eta ~ gamma(c, d);
    m ~ von_mises(gamma, eta);
    a ~ gamma(xi, zeta);
    b ~ gamma(xi, zeta);
    beta ~ gamma(1, 1);
    for (g in 1:priori){
        kappapriori[g] ~ gamma(a, b);
    }
    for (g in 1:priori){
        mupriori[g] ~ von_mises(m, beta*kappapriori[g]);
    }
    for (i in 1:sum(Npriori)){
        int gruppo = gruppipriori[i];
        angolipriori[i] ~ von_mises(mupriori[gruppo], kappapriori[gruppo]);
    }
    kappa ~ gamma(a, b);
    mu ~ von_mises(m, beta*kappa);
    for (i in 1:Ncorrente){
        angolicorrenti[i] ~ von_mises(mu, kappa);
    }
}