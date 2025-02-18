import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm, uniform
from shared.utils import rank
import warnings


class DiscretizedDist():
    def __init__(self, base_dist, vals, rng):
        self.base_dist = base_dist
        self.vals = vals.copy()
        self.rng = rng
        raw_probs = base_dist.pdf(vals)
        probs = raw_probs / raw_probs.sum()
        self.probs = probs
        self.cumprobs = np.cumsum(probs)
        self.pdf_lookup = pd.Series(probs, index=vals)
        
    def pdf(self, x):
        return self.pdf_lookup[x]
    
    def rv(self):
        u = self.rng.random()
        if u == self.cumprobs[-1]:  # Edge case
            idx = self.cumprobs.size - 1
        else:
            idx = np.argmax(u < self.cumprobs)
        return self.vals[idx]
    
    def rvs(self, size=1):
        if size > 1:
            x = np.array([self.rv() for _ in range(size)])
        else:
            x = self.rv()
        return x

    
class Analyzer(ABC):
    name: str
    
    def __init__(self, name, alpha, prior_dist):
        self.name = name
        self.alpha = alpha
        self.lower_quantile = alpha/2.
        self.upper_quantile = 1. - self.lower_quantile
        self.ci_quantiles = np.array([self.lower_quantile, self.upper_quantile])
        self.z_star = norm.ppf(1. - alpha/2.)
        self.prior_dist = prior_dist
        
    def get_ys(self, y, a, thetas):
        thetas_repped = np.repeat(thetas.copy()[:, np.newaxis], y.size, axis=1)
        y0 = y - a*thetas_repped
        y1 = y0 + thetas_repped
        return y0, y1
    
    @abstractmethod
    def analyze(self, y, a, thetas) -> (float, float, float, float):
        """Return estimate, lower bound, upper bound, and nominal coverage rate"""
        pass

    
class ProbAnalyzer(Analyzer):   
    def __init__(self, name, alpha, prior_dist):
        super().__init__(name, alpha, prior_dist)
        
    def normalize_probs(self, raw_probs):
        raw_probs = np.asarray(raw_probs)
        raw_probs_sum = raw_probs.sum()
        probs = raw_probs / raw_probs_sum
        return probs
        
    def process_probs(self, raw_probs, thetas):
        probs = self.normalize_probs(raw_probs)
        posterior_mean = (probs * thetas).sum()
        cum_probs = np.cumsum(probs)
        lower_idx = int(np.argmax(self.lower_quantile < cum_probs))
        if lower_idx > 0:
            lower_idx -= 1
        upper_idx = int(np.argmax(self.upper_quantile < cum_probs))
        lower_bound = thetas[lower_idx]
        upper_bound = thetas[upper_idx]
        nominal_coverage_rate = cum_probs[upper_idx] - cum_probs[lower_idx] + probs[lower_idx]
        return posterior_mean, lower_bound, upper_bound, nominal_coverage_rate


class PriorAnalyzer(ProbAnalyzer):
    def __init__(self, name: str, alpha: float, prior_dist):
        super().__init__(name, alpha, prior_dist)
    
    def analyze(self, y, a, thetas):
        raw_probs = self.prior_dist.pdf(thetas)
        return self.process_probs(raw_probs, thetas)

class BayesAnalyzer(ProbAnalyzer):
    @abstractmethod
    def get_likelihoods(self, y, a, thetas):
        pass
    
    def analyze(self, y, a, thetas):
        posterior_probs = self.get_posterior_probs(y, a, thetas)
        return self.summarize_posterior(posterior_probs, thetas)
    
    def get_posterior_probs(self, y, a, thetas):
        prior_probs = self.prior_dist.pdf(thetas)
        posterior_probs_raw = prior_probs * self.get_likelihoods(y, a, thetas)
        posterior_probs = self.normalize_probs(posterior_probs_raw)
        return posterior_probs
    
    def summarize_posterior(self, posterior_probs, thetas):
        return self.process_probs(posterior_probs, thetas)
    
    
class CalculatesDiffMeans():
    
    def get_n_each(self, a):
        return int(a.sum())
    
    def get_treatment_mean(self, y, a):
        return (a*y).sum() / self.get_n_each(a)
        
    def get_control_mean(self, y, a):
        return ((1.-a)*y).sum() / self.get_n_each(a)
    
    def get_diff_means(self, y, a):
        treatment_mean = self.get_treatment_mean(y, a)
        control_mean = self.get_control_mean(y, a)
        return treatment_mean - control_mean
    

class BRIAnalyzer(BayesAnalyzer, CalculatesDiffMeans):
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray):
        super().__init__(name, alpha, prior_dist)
        self.n_each = n_each
        self.n = 2*n_each
        self.n_theta_vals = n_theta_vals
        self.a_vals_3d = np.repeat(a_vals.copy()[np.newaxis, :, :], n_theta_vals, axis=0)
        self.not_a_vals_3d = ~self.a_vals_3d
        self.n_combs = a_vals.shape[0]
    
    def get_ys_3d(self, y, a, thetas):
        y0, y1 = self.get_ys(y, a, thetas)
        y0_3d = np.repeat(y0[:, np.newaxis, :], self.n_combs, axis=1)
        y1_3d = np.repeat(y1[:, np.newaxis, :], self.n_combs, axis=1)
        return y0_3d, y1_3d
    
    def get_stats(self, y, a, thetas):
        diff_means_observed = self.get_diff_means(y, a)
        y0_3d, y1_3d = self.get_ys_3d(y, a, thetas)
        control_means = (self.not_a_vals_3d * y0_3d).sum(axis=2) / self.n_each
        treated_means = (self.a_vals_3d * y1_3d).sum(axis=2) / self.n_each
        diff_means = treated_means - control_means        
        return diff_means_observed, diff_means
   
    
class RankSumAnalyzer(BRIAnalyzer):
    def get_likelihoods(self, y, a, thetas):
        ranks_observed = rank(y)
        rank_sum_observed = (ranks_observed * a).sum()
        
        y0_3d, y1_3d = self.get_ys_3d(y, a, thetas)
        y_3d = self.a_vals_3d*y1_3d + self.not_a_vals_3d*y0_3d
        ranks = rank(y_3d)  # Ranks last axis by default
        rank_sums = (ranks * self.a_vals_3d).sum(axis=2)
        
        likelihoods = (rank_sums == rank_sum_observed).mean(axis=1)
        return likelihoods

    
class BRIOneSidedAnalyzer(BRIAnalyzer, CalculatesDiffMeans):
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray, nu:float = 0.1):
        super().__init__(name, alpha, prior_dist, n_each, n_theta_vals, a_vals)
        self.nu = nu
    
    def get_likelihoods(self, y, a, thetas, tol=1e-16):
        n_each = self.get_n_each(a)
        treatment_mean_observed = self.get_treatment_mean(y, a)
        y0_3d, y1_3d = self.get_ys_3d(y, a, thetas)
        y_3d = self.a_vals_3d*y1_3d + self.not_a_vals_3d*y0_3d
        E_treatment_means = (self.a_vals_3d*y1_3d).sum(axis=2) / n_each
        Var_treatment_means = ((self.nu / n_each)**2) * (self.a_vals_3d * a).sum(axis=2)
        Sd_treatment_means = np.sqrt(Var_treatment_means)
        zero_sd = Sd_treatment_means < tol
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        likelihood_components = norm.pdf(treatment_mean_observed, loc=E_treatment_means, scale=Sd_treatment_means)
        likelihood_components = np.nan_to_num(likelihood_components)
        equals_observed = np.abs(E_treatment_means - treatment_mean_observed) < tol
        likelihood_components = (~zero_sd) * likelihood_components + zero_sd * equals_observed
        warnings.simplefilter("default")
        
        likelihoods = likelihood_components.mean(axis=1)
        return likelihoods


class RoundedAnalyzer(BRIAnalyzer, CalculatesDiffMeans):
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray, digits:int=1):
        super().__init__(name, alpha, prior_dist, n_each, n_theta_vals, a_vals)
        self.digits = digits
    
    def get_likelihoods(self, y, a, thetas):
        diff_means_observed, diff_means = self.get_stats(y, a, thetas)
        diff_means_observed_rounded = np.round(diff_means_observed, self.digits)
        diff_means_rounded = np.round(diff_means, self.digits)
        likelihoods = (diff_means_rounded == diff_means_observed_rounded).mean(axis=1)
        return likelihoods

    
class NeighborhoodAnalyzer(BRIAnalyzer, CalculatesDiffMeans):
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray, eps: float):
        super().__init__(name, alpha, prior_dist, n_each, n_theta_vals, a_vals)
        self.eps = eps
    
    def get_likelihoods(self, y, a, thetas):
        diff_means_observed, diff_means = self.get_stats(y, a, thetas)
        below = diff_means < (diff_means_observed + self.eps)
        above = (diff_means_observed - self.eps) < diff_means
        in_neighborhood = below & above
        likelihoods = in_neighborhood.mean(axis=1)
        return likelihoods
    
    
class BRIAsympAnalyzer(BayesAnalyzer, CalculatesDiffMeans):
    def get_likelihoods(self, y, a, thetas):
        n_each = int(a.sum())
        n = 2*n_each
        diff_means_obs = self.get_diff_means(y, a)
        y0, y1 = self.get_ys(y, a, thetas)
        avg_diff_means = (y1 - y0).mean(axis=1)
        s2_0 = y0.var(axis=1, ddof=1)
        s2_1 = y1.var(axis=1, ddof=1)
        s2_01 = (y1-y0).var(axis=1, ddof=1)
        var_diff_means = s2_0/n_each + s2_1/n_each - s2_01/n
        sd_diff_means = np.sqrt(var_diff_means)
        likelihoods = norm.pdf(diff_means_obs, loc=avg_diff_means, scale=sd_diff_means)
        return likelihoods


class FreqAnalyzer():
    def get_est_var(self, y, a, thetas):
        y1 = y[a]
        y0 = y[~a]
        est = y1.mean() - y0.mean()
        var = y1.var()/y1.size + y0.var()/y0.size
        return est, var
    
    
class DiffMeansAnalyzer(FreqAnalyzer, Analyzer):
    def analyze(self, y, a, thetas):
        est, var = self.get_est_var(y, a, thetas)
        se = np.sqrt(var)
        lb = est - self.z_star * se
        ub = est + self.z_star * se
        cr = 1. - self.alpha
        return est, lb, ub, cr

    
class LIBDiffMeansAnalyzer(FreqAnalyzer, BayesAnalyzer):
    def get_likelihoods(self, y, a, thetas):
        est, var = self.get_est_var(y, a, thetas)
        sd = np.sqrt(var)
        likelihoods = norm.pdf(thetas, est, sd)
        return likelihoods