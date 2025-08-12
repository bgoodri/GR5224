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
  real<lower = 0, upper = 0.5> good_m;
  real<lower = 0> good_s;
  real<lower = 0, upper = 0.5> bad_m;
  real<lower = 0> bad_s;
}
parameters {
  array[18] real<lower = 0, upper = 0.5> good;
  array[18] real<lower = 0, upper = 0.5> bad;
}
model {
  target += normal_lpdf(good | good_m, good_s);
  target += normal_lpdf(bad  |  bad_m,  bad_s);
  
  int i = 0;
  for (h in 1:18) {
    real good_h = good[h];
    real bad_h = bad[h];
    int par_h = par_values[h];
    for (j in 1:J) for (r in 1:2) {
      i += 1;
      target += golf_lpmf(strokes[i] | par_h, good_h, bad_h);
    }
  }
}
generated quantities {
  vector[size(strokes)] log_lik;
  { // local block means i does not get stored
    int i = 0;
    for (h in 1:18) {
      real good_h = good[h];
      real bad_h = bad[h];
      int par_h = par_values[h];
      for (j in 1:J) for (r in 1:2) {
        i += 1;
        log_lik[i] = golf_lpmf(strokes[i] | par_h, good_h, bad_h);
      }
    }
  }
}
