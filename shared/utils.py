import numpy as np
from scipy.stats import norm

def float2str_format(x: np.ndarray, digits=3) -> np.ndarray:
    fmt = f"%.{digits}f"
    x_str = np.char.mod(fmt, x)
    return x_str

def print_se_parentheses(x: np.ndarray, x_ses: np.ndarray, digits=3, stars=False, alpha=0.05) -> np.ndarray:
    x = x.copy().reshape((-1,))
    x_ses = x_ses.copy().reshape((-1,))
    x_str = float2str_format(x, digits)
    x_se_str = float2str_format(x_ses, digits)
    x_se_parentheses = np.char.add(x_str, np.char.add(" (", np.char.add(x_se_str, ")")))
    if stars:
        z_star = norm.ppf(1. - alpha/2.)
        stars_chars = [("*" if abs(x[i]/x_ses[i]) >= z_star else "") for i in range(x.size)]
        x_se_parentheses = np.char.add(x_se_parentheses, np.array(stars_chars))
    return x_se_parentheses

def rank(arr, axis=-1):
    order = np.argsort(arr, axis=axis)
    ranks = np.argsort(order, axis=axis)
    return ranks