import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm, uniform
from shared.utils import rank, safe_log
import warnings


class DiscretizedDist():
    def __init__(self, base_dist, vals):
        self.base_dist = base_dist
        self.vals = vals.copy()
        raw_probs = base_dist.pdf(vals)
        normalizing_constant = raw_probs.sum()
        log_normalizing_constant = np.log(normalizing_constant)
        self.log_probs = base_dist.logpdf(vals) - log_normalizing_constant
        self.probs = raw_probs / normalizing_constant
        self.cumprobs = np.cumsum(self.probs)
        self.logpdf_lookup = pd.Series(self.log_probs, index=vals)
        self.pdf_lookup = pd.Series(self.probs, index=vals)

    def logpdf(self, x):
        return self.logpdf_lookup[x]
    
    def pdf(self, x):
        return self.pdf_lookup[x]
    
    def rvs(self, rng, size=1):
        us = rng.random(size)
        idxs = np.searchsorted(self.cumprobs, us, side="left")
        x = self.vals[idxs]
        if size == 1:
            x = x[0]
        return x

    
class Analyzer(ABC):
    name: str
    
    def __init__(self, name, alpha, prior_dist, uses_s=False):
        self.name = name
        self.alpha = alpha
        self.lower_quantile = alpha/2.
        self.upper_quantile = 1. - self.lower_quantile
        self.ci_quantiles = np.array([self.lower_quantile, self.upper_quantile])
        self.z_star = norm.ppf(1. - alpha/2.)
        self.prior_dist = prior_dist
        self.uses_s = uses_s
        
    def get_ys(self, y, a, thetas):
        thetas_repped = np.repeat(thetas.copy()[:, np.newaxis], y.size, axis=1)
        y0 = y - a*thetas_repped
        y1 = y0 + thetas_repped
        return y0, y1

    @abstractmethod
    def analyze(self, y, a, thetas, y_s=None, a_s=None) -> (float, float, float, float):
        """Return estimate, lower bound, upper bound, and nominal coverage rate"""
        pass

    def analyze_s(self, y, a, thetas, y_s=None, a_s=None) -> (float, float, float, float):
        if self.uses_s:
            return self.analyze(y, a, thetas, y_s=y_s, a_s=a_s)
        else:
            return self.analyze(y, a, thetas)

    
class ProbAnalyzer(Analyzer):   
    def __init__(self, name, alpha, prior_dist, uses_s=False):
        super().__init__(name, alpha, prior_dist, uses_s=uses_s)
        
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
    def __init__(self, name: str, alpha: float, prior_dist, uses_s=False):
        super().__init__(name, alpha, prior_dist, uses_s=uses_s)
    
    def analyze(self, y, a, thetas, y_s=None, a_s=None):
        raw_probs = self.prior_dist.pdf(thetas)
        return self.process_probs(raw_probs, thetas)

class BayesAnalyzer(ProbAnalyzer):
    @abstractmethod
    def get_log_likelihoods(self, y, a, thetas, y_s=None, a_s=None):
        pass
    
    def analyze(self, y, a, thetas, y_s=None, a_s=None):
        posterior_probs = self.get_posterior_probs(y, a, thetas, y_s=y_s, a_s=a_s)
        return self.process_probs(posterior_probs, thetas)
    
    def get_posterior_probs(self, y, a, thetas, y_s=None, a_s=None):
        prior_log_probs = self.prior_dist.logpdf(thetas)
        log_likelihoods = self.get_log_likelihoods(y, a, thetas, y_s=y_s, a_s=a_s)
        posterior_log_probs_raw = prior_log_probs + log_likelihoods
        posterior_probs_raw = np.exp(posterior_log_probs_raw)
        posterior_probs = self.normalize_probs(posterior_probs_raw)
        return posterior_probs
    
    
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
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray, uses_s=False):
        super().__init__(name, alpha, prior_dist, uses_s=uses_s)
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
    
    def get_simulated_stats(self, y, a, thetas):
        y0_3d, y1_3d = self.get_ys_3d(y, a, thetas)
        control_means = (self.not_a_vals_3d * y0_3d).sum(axis=2) / self.n_each
        treated_means = (self.a_vals_3d * y1_3d).sum(axis=2) / self.n_each
        diff_means = treated_means - control_means        
        return diff_means
   
    
class RankSumAnalyzer(BRIAnalyzer):
    def get_log_likelihoods(self, y, a, thetas, y_s=None, a_s=None):
        if (y_s is None) and (a_s is None):
            rank_sum_observed = (rank(y) * a).sum()
        else:
            rank_sum_observed = (rank(y_s) * a_s).sum()
        y0_3d, y1_3d = self.get_ys_3d(y, a, thetas)
        y_3d = self.a_vals_3d*y1_3d + self.not_a_vals_3d*y0_3d
        ranks = rank(y_3d)  # Ranks last axis by default
        rank_sums = (ranks * self.a_vals_3d).sum(axis=2)
        
        likelihoods = (rank_sums == rank_sum_observed).mean(axis=1)
        return safe_log(likelihoods)

    
class BRIOneSidedAnalyzer(BRIAnalyzer, CalculatesDiffMeans):
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray, nu:float = 0.1, uses_s=False):
        super().__init__(name, alpha, prior_dist, n_each, n_theta_vals, a_vals, uses_s=uses_s)
        self.nu = nu
    
    def get_log_likelihoods(self, y, a, thetas, y_s=None, a_s=None, tol=1e-16):
        n_each = self.get_n_each(a)
        if (y_s is None) and (a_s is None):
            treatment_mean_observed = self.get_treatment_mean(y, a)
        else:
            treatment_mean_observed = self.get_treatment_mean(y_s, a_s)
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
        return safe_log(likelihoods)


class RoundedAnalyzer(BRIAnalyzer, CalculatesDiffMeans):
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray, digits:int=1, uses_s=False):
        super().__init__(name, alpha, prior_dist, n_each, n_theta_vals, a_vals, uses_s=uses_s)
        self.digits = digits
    
    def get_log_likelihoods(self, y, a, thetas, y_s=None, a_s=None):
        if (y_s is None) and (a_s is None):
            diff_means_observed = self.get_diff_means(y, a)
        else:
            diff_means_observed = self.get_diff_means(y_s, a_s)
        diff_means = self.get_simulated_stats(y, a, thetas)
        diff_means_observed_rounded = np.round(diff_means_observed, self.digits)
        diff_means_rounded = np.round(diff_means, self.digits)
        likelihoods = (diff_means_rounded == diff_means_observed_rounded).mean(axis=1)
        return safe_log(likelihoods)

    
class NeighborhoodAnalyzer(BRIAnalyzer, CalculatesDiffMeans):
    def __init__(self, name: str, alpha: float, prior_dist, n_each: int, n_theta_vals: int, a_vals: np.ndarray, eps: float, uses_s=False):
        super().__init__(name, alpha, prior_dist, n_each, n_theta_vals, a_vals, uses_s=uses_s)
        self.eps = eps
    
    def get_log_likelihoods(self, y, a, thetas, y_s=None, a_s=None):
        if (y_s is None) and (a_s is None):
            diff_means_observed = self.get_diff_means(y, a)
        else:
            diff_means_observed = self.get_diff_means(y_s, a_s)
        diff_means = self.get_simulated_stats(y, a, thetas)
        below = diff_means < (diff_means_observed + self.eps)
        above = (diff_means_observed - self.eps) < diff_means
        in_neighborhood = below & above
        likelihoods = in_neighborhood.mean(axis=1)
        return safe_log(likelihoods)
    
    
class BRIAsympAnalyzer(BayesAnalyzer, CalculatesDiffMeans):
    def get_log_likelihoods(self, y, a, thetas, y_s=None, a_s=None):
        n_each = int(a.sum())
        n = 2*n_each
        if (y_s is None) and (a_s is None):
            diff_means_obs = self.get_diff_means(y, a)
        else:
            diff_means_obs = self.get_diff_means(y_s, a_s)
        y0, y1 = self.get_ys(y, a, thetas)
        avg_diff_means = (y1 - y0).mean(axis=1)
        s2_0 = y0.var(axis=1, ddof=1)
        s2_1 = y1.var(axis=1, ddof=1)
        s2_01 = (y1-y0).var(axis=1, ddof=1)
        var_diff_means = s2_0/n_each + s2_1/n_each - s2_01/n
        sd_diff_means = np.sqrt(var_diff_means)
        log_likelihoods = norm.logpdf(diff_means_obs, loc=avg_diff_means, scale=sd_diff_means)
        return log_likelihoods


class FreqAnalyzer():
    def get_est_var(self, y, a, thetas):
        y1 = y[a]
        y0 = y[~a]
        est = y1.mean() - y0.mean()
        var = y1.var()/y1.size + y0.var()/y0.size
        return est, var
    
    
class DiffMeansAnalyzer(FreqAnalyzer, Analyzer):
    def analyze(self, y, a, thetas, y_s=None, a_s=None):
        est, var = self.get_est_var(y, a, thetas)
        se = np.sqrt(var)
        lb = est - self.z_star * se
        ub = est + self.z_star * se
        cr = 1. - self.alpha
        return est, lb, ub, cr

    
class LIBDiffMeansAnalyzer(FreqAnalyzer, BayesAnalyzer):
    def get_log_likelihoods(self, y, a, thetas, y_s=None, a_s=None):
        est, var = self.get_est_var(y, a, thetas)
        sd = np.sqrt(var)
        log_likelihoods = norm.logpdf(thetas, est, sd)
        return log_likelihoods