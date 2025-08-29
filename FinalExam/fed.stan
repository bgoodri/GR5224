data {
  int<lower = 1> n;
  vector<lower = 0, upper = 1>[n] y;
  vector[n] days; // until Fed meeting in September 2025
  real alpha_m; real<lower = 0> alpha_s;
  real<lower = 0, upper = 1> beta_m; real<lower = 0> beta_s;
  real omega_m; real<lower = 0> omega_s;
  real<lower = 0> theta_m; real<lower = 0> theta_s;
}
transformed data {
  vector[n] log_odds = log(y ./ (1 - y));
  vector[n] lag = append_row(0, head(log_odds, n - 1));
}
parameters {
  real alpha;
  real<lower = 0, upper = 1> beta;
  real omega;
  real<lower = 0> theta;
}
model {
  target += normal_lpdf(alpha | alpha_m, alpha_s);
  target += normal_lpdf(beta  | beta_m, beta_s);
  target += normal_lpdf(omega | omega_m, omega_s);
  target += normal_lpdf(theta | theta_m, theta_s);
  vector[n] gamma = exp(omega + theta * days);
  target += logistic_lpdf(log_odds | alpha + beta * lag, gamma);
}
generated quantities {
  vector[n] log_lik;
  vector[n] y_rep;
  for (i in 1:n) {
    real gamma = exp(omega + theta * days[i]);
    real mu = alpha + beta * lag[i];
    log_lik[i] = logistic_lpdf(log_odds[i] | mu, gamma);
    y_rep[i] = logistic_rng(mu, gamma);
  }
  vector[22] prob;
  {
    real past = log_odds[n];
    for (i in 1:22) {
      real mu = alpha + beta * past;
      real gamma = exp(omega + theta * (22 - i));
      real Y = logistic_rng(mu, gamma);
      prob[i] = inv_logit(Y);
      past = Y;
    }
  }
}
