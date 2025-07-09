import numpy as np
from siuba import _, group_by, mutate, ungroup, summarize
from siuba.dply.vector import sample

def Pr(x, n=10):
    if np.any(x > n):
        return 0
    return np.log(1 + 1 / (n + 1 - x)) / np.log(n + 2)

Omega = np.arange(11)  # 0, 1, ..., 10
Omega_probs = Pr(Omega)  # probabilities for each pin count

R = 10**7

# Create the first rolls of R frames
frames = pd.DataFrame({
    'X_1': sample(Omega, size=R, replace=True, prob=Omega_probs)
})

# Create the second rolls via a group_by on X_1 to sample X_2 conditionally
frames = (
    frames
    >> group_by(_.X_1)
    >> mutate(
        X_2 = sample(
            Omega,
            size=len(_),
            replace=True,
            prob=Pr(Omega, n=10 - _.X_1.iloc[0])
        )
    )
    >> ungroup()
)

joint_Pr = np.zeros((len(Omega), len(Omega)))
for x_1 in Omega:
    Pr_x_1 = Pr(x_1, n=10)
    for x_2 in range(0, 11 - x_1):
        joint_Pr[x_1, x_2] = Pr_x_1 * Pr(x_2, n=10 - x_1)
        

