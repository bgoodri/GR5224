data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  int<lower = 2> J; // number of outcome categories
  array[N] int<lower = 1, upper = J> y;   // outcomes
  int<lower = 0, upper = 1> prior_only;   // ignore data?
} // data block can include stuff for priors
parameters {
  vector[J - 1] alpha_free;
  matrix[K, J - 1] beta_free;
}
transformed parameters {
  vector[J] alpha   = append_row(0, alpha_free);
  matrix[K, J] beta = append_col(rep_vector(0, K), beta_free);
}
model {
  target += std_normal_lpdf(alpha_free);
  target += std_normal_lpdf(to_vector(beta_free));
  if (!prior_only) {
    target += categorical_logit_glm_lpmf(y | X, alpha, beta);
  } 
}
generated quantities {
  vector[N] log_lik; // for use with loo() etc.
  for (n in 1:N) {
    log_lik[n] = categorical_logit_glm_lpmf(y[n] | X[n, ], alpha, beta);
  }
}
