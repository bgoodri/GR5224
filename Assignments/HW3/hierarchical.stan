functions {
  real golf_lpmf(int strokes, int par, real good, real bad) {
    if (strokes < 2) reject("strokes must be at least 2");
    if (par < 3) reject("par must be at least 3");
    if (good <= 0 || good >= 0.5) reject("good must be on (0,0.5)");
    if (bad  <= 0 || bad  >= 0.5) reject("bad must be on (0,0.5)");   
    int k = (par + 1) %/% 2;
    if (strokes < k) return negative_infinity();
    vector[par - k + 1] terms;
    real log_good = log(good);
    real log_bad  = log(bad);
    real log_mediocre = log1m(good + bad);
    int i = 0;
    for (j in k:par) {
      i += 1;
      int strokes_j = strokes - j;
      if (strokes_j < 0) {
        terms[i] = negative_infinity();
        continue;
      }
      int par_j = par - j;
      int twoj_par = 2 * j - par;
      int twoj_par_1 = twoj_par - 1;
      terms[i] = lchoose(strokes - 1, strokes_j)
               + strokes_j * log_bad
               + log_sum_exp(lchoose(j - 1, twoj_par_1) + 
                             (par_j + 1) * log_good +
                             twoj_par_1 * log_mediocre,
                             lchoose(j, twoj_par) + 
                             par_j * log_good + 
                             twoj_par * log_mediocre);
    }
    return log_sum_exp(terms);
  }
}
data {
  int<lower = 0> J; // number of golfers
  array[18 * J * 2] int<lower = 2> strokes;
  array[18] int<lower = 3, upper = 5> par_values;
  array[2] real hole_m;
  array[2] real<lower = 0> hole_s;
  array[2] real golfer_m;
  array[2] real<lower = 0> golfer_s;
  array[2] real<lower = 0> hole_r;
  array[2] real<lower = 0> golfer_r;
}
parameters {
  array[2] real hole_mu;
  array[2] real golfer_mu;
  array[2] real<lower = 0> hole_sigma;
  array[2] real<lower = 0> golfer_sigma;
  real<lower = -1, upper = 1> hole_rho;
  real<lower = -1, upper = 1> golfer_rho;
  array[18, 2] real hole_std;
  array[J, 2]  real golfer_std;
}
transformed parameters {
  real hole_beta = hole_sigma[2] / hole_sigma[1] * hole_rho;
  real golfer_beta = golfer_sigma[2] / golfer_sigma[1] * golfer_rho; 
  real hole_sigma_cond = hole_sigma[2] * sqrt(1 - square(hole_rho));
  real golfer_sigma_cond = golfer_sigma[2] * sqrt(1 - square(golfer_rho));  
}
model {
  target += normal_lpdf(hole_mu | hole_m, hole_s);
  target += normal_lpdf(golfer_mu | golfer_m, golfer_s);
  target += exponential_lpdf(hole_sigma | hole_r);
  target += exponential_lpdf(golfer_sigma | golfer_r);
  // implicit: hole_rho and golfer_rho are uniform under the prior
  target += std_normal_lpdf(to_array_1d(hole_std));
  target += std_normal_lpdf(to_array_1d(golfer_std));
  
  int i = 0;
  for (h in 1:18) {
    int par_h = par_values[h];
    real good_h = hole_mu[1] + hole_sigma[1] * hole_std[h, 1];
    real bad_h = hole_mu[2] + hole_beta * (good_h - hole_mu[1])
               + hole_sigma_cond * hole_std[h, 2];
    for (j in 1:J) {
      real good_j = golfer_mu[1] + golfer_sigma[1] * golfer_std[j, 1];
      real bad_j  = golfer_mu[2] + golfer_beta * (good_j - golfer_mu[1])
                  + golfer_sigma_cond * golfer_std[j, 2];
      real good = inv_logit(good_h + good_j) / 2;
      real bad  = inv_logit(bad_h + bad_j) / 2;
      for (r in 1:2) {
        i += 1;
        target += golf_lpmf(strokes[i] | par_h, good, bad);
      }
    }
  }
}
generated quantities {
  vector[size(strokes)] log_lik;
  { // local block means i does not get stored
    int i = 0;
    for (h in 1:18) {
      int par_h = par_values[h];
      real good_h = hole_mu[1] + hole_sigma[1] * hole_std[h, 1];
      real bad_h = hole_mu[2] + hole_beta * (good_h - hole_mu[1])
                 + hole_sigma_cond * hole_std[h, 2];
      for (j in 1:J) {
        real good_j = golfer_mu[1] + golfer_sigma[1] * golfer_std[j, 1];
        real bad_j  = golfer_mu[2] + golfer_beta * (good_j - golfer_mu[1])
                    + golfer_sigma_cond * golfer_std[j, 2];
        real good = inv_logit(good_h + good_j) / 2;
        real bad  = inv_logit(bad_h + bad_j) / 2;
        for (r in 1:2) {
          i += 1;
          log_lik[i] += golf_lpmf(strokes[i] | par_h, good, bad);
        }
      }
    }
  }
}
