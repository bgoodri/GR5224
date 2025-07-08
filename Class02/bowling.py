from sympy import *
from random import choices

Omega = list(range(11)) # 0, 1, ..., 10 but EXCLUDES 11
def Pr(x, n = 10):
  assert isinstance(x, int)
  assert isinstance(n, int)
  assert n >= 0 & n <= 10
  if (x > n | x < 0):
    return 0
  return log(1 + 1 / (n + 1 - x), n + 2).evalf()

p = list(map(Pr, Omega))

x_1 = choices(Omega, weights = p)[0]

x_1 = symbols("x_1", integer = True)
x_2 = symbols("x_2", integer = True)
joint_Pr = zeros(11, 11)
for x_1 in Omega:
  for x_2 in range(0, 11 - x_1):
    joint_Pr[x_1, x_2] = Pr(x_1) * Pr(x_2, n = 10 - x_1)

