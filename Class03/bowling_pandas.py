import numpy as np
import pandas as pd

def Pr(x, n=10):
    if np.any(x > n):
        return 0
    return np.log(1 + 1 / (n + 1 - x)) / np.log(n + 2)

Omega = np.arange(11)  # 0, 1, ..., 10
Omega_probs = Pr(Omega)

R = 10**7
frames = pd.DataFrame({'X_1': np.random.choice(Omega, size=R, p=Omega_probs)})

def sample_second_roll(group):
    remaining_pins = 10 - group.name
    valid_Omega = Omega[Omega <= remaining_pins]
    probs = Pr(valid_Omega, n=remaining_pins)
    samples = np.random.choice(valid_Omega, size=len(group), p=probs)
    return pd.Series(samples, index=group.index)

frames['X_2'] = frames.groupby('X_1', group_keys=False).apply(sample_second_roll)

joint_Pr = np.zeros((len(Omega), len(Omega)))
for x_1 in Omega:
    Pr_x_1 = Pr(x_1, n=10)
    for x_2 in range(0, 11 - x_1):
        joint_Pr[x_1, x_2] = Pr_x_1 * Pr(x_2, n=10 - x_1)
