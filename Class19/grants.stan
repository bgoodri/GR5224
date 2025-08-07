data {
  int<lower = 0> N; // number of observations
  int<lower = 1> J; // number of disciplines
  array[N] int<lower = 1, upper = J> discipline;
  vector<lower = 0, upper = 1>[N] female;
  array[N] int<lower = 0> applications;
  array[N] int<lower = 0, upper = max(applications)> awards;
  
  int<lower = 0, upper = 1> prior_only;
  vector[2] m;            // prior me{di}ans
  vector<lower = 0>[2] s; // prior standard deviations
  real<lower = 0> r;      // prior rate
}
transformed data {
  real x_bar = mean(female);
  vector[N] x = female - x_bar;
}
parameters {
  real gamma; // intercept relative to centered predictor
  real beta;  // coefficient on centered predictors
  real<lower = 0> sigma; // standard deviation in intercepts
  
  vector[J] a; // deviations in intercept
}

model {
  
  target += normal_lpdf(gamma | m[1], s[1]);
  target += normal_lpdf(beta  | m[2], s[2]);
  target += exponential_lpdf(sigma | r);
  
  target += normal_lpdf(a | 0, sigma);
  
  if (!prior_only) {
    vector[N] eta = gamma + beta * x + a[discipline];
    target += binomial_logit_lpmf(awards | applications, eta);
  }
}
generated quantities {
  real alpha = gamma -beta * x_bar;
  vector[N] log_lik;
  for (n in 1:N) {
    real eta = gamma+ beta * x[n] + a[discipline[n]];
    log_lik[n] = binomial_logit_lpmf(awards[n] | applications[n], eta);
  }
}
