import numpy as np
import pandas as pd

def Pr(x, n=10, theta=1):
    """
    Probability Mass Function for bowling with ability theta
    
    Parameters:
    x: scalar or array-like
        Input value(s) to evaluate the function at
    n: int, optional
        Parameter n (default 10)
    theta: float, optional
        Parameter theta (default 1)
    
    Returns:
    scalar or array-like
        Result of the function applied to each x
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    inv_theta = 1 / theta
    
    # Conditions
    cond1 = (x > n) | (x < 0)
    cond2 = (~cond1) & (theta > 0)
    cond3 = (~cond1) & (theta < 0)
    cond4 = (~cond1) & (theta == 0)
    
    # Calculations for each condition
    result[cond1] = 0
    result[cond2] = np.log1p(1 / (n + inv_theta - x[cond2])) / np.log(theta * (n + 1 + inv_theta))
    result[cond3] = np.log1p(1 / (x[cond3] - inv_theta)) / np.log1p(-theta * (n + 1))
    result[cond4] = 1 / n
    
    return result.item() if np.isscalar(x) else result
  
Omega = np.arange(11)
