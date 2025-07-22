Pr <- Vectorize(function(x, n = 10, theta = 1) {
  inv_theta <- 1 / theta
  if (x > n | x < 0) {
    0
  } else if (theta > 0) {
    log1p(1 / (n + inv_theta - x)) / log1p(theta * (n + 1))
  } else if (theta < 0) {
    log1p(1 / (x - inv_theta)) / log1p(-theta * (n + 1))
  } else 1 / (n + 1)
})

Omega <- 0:10
names(Omega) <- as.character(Omega)
