import numpy as np
import pandas as pd
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

print(frames.head())

joint_Pr = np.zeros((len(Omega), len(Omega)))
for x_1 in Omega:
    Pr_x_1 = Pr(x_1, n=10)
    for x_2 in range(0, 11 - x_1):
        joint_Pr[x_1, x_2] = Pr_x_1 * Pr(x_2, n=10 - x_1)
        

summarize(frames, joint_prob = (_.X_1 == 8) & (_.X_2 == 2).mean()) # modern
Pr(x = 8, n = 10) * Pr(x = 2, n = 10 - 8) # ancient

(frames 
 >> summarize(
     prob = ((_.X_1 == 8) | (_.X_2 == 2)).mean(),
     wrong = (_.X_1 == 8).mean() + (_.X_2 == 2).mean(),
     right = (_.X_1 == 8).mean() + (_.X_2 == 2).mean() - 
             ((_.X_1 == 8) & (_.X_2 == 2)).mean()
 )
)
# modern above vs. ancient below
joint_Pr_df = pd.DataFrame(
    np.random.rand(11, 11),  # random numbers for example
    index=[str(i) for i in range(11)],
    columns=[str(i) for i in range(11)]
)
result = (joint_Pr_df.loc["8"].sum() + 
          joint_Pr_df["2"].sum() - 
          joint_Pr_df.loc["8", "2"])
print(result)

(frames
 >> summarize(
     spare_prob = ((_.X_1 != 10) & (_.X_1 + _.X_2 == 10)).mean()
 )
)
# modern above vs. ancient below
spare_prob = 0
for x_1 in range(0, 10): 
  spare_prob = spare_prob + joint_Pr[x_1, 10 - x_1]
  
spare_prob
spare_prob
