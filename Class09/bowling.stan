functions {
  real log_Pr(int x, int n, real theta) {
    real inv_theta = inv(theta);
    if (x > n || x < 0) return negative_infinity();
    if (theta > 0) {
      return log(log1p(inv(n + inv_theta - x))) - 
             log(log1p(theta * (n + 1)));
    } else if (theta < 0) {
      return log(log1p(inv(x - inv_theta))) - 
             log(log1p(-theta * (n + 1)));
    } else return -log(n + 1);
  }
}
data {
  int<lower = 0> N; // frames
  array[N, 2] int<lower = 0, upper = 10> pins;
  real theta_m; real<lower = 0> theta_s;
}
parameters {
  real theta;
}
model {
  target += normal_lpdf(theta | theta_m, theta_s);
  for (i in 1:N) {
    int x_1 = pins[i, 1];
    target += log_Pr(x_1, 10, theta);
    target += log_Pr(pins[i, 2], 10 - x_1, theta);
  }
}
