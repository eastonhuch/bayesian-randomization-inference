import numpy as np
from scipy.stats import norm
from itertools import combinations
from shared.univariate import DiscretizedDist, PriorAnalyzer, RankSumAnalyzer, RoundedAnalyzer, NeighborhoodAnalyzer, BRIOneSidedAnalyzer, BRIAsympAnalyzer, DiffMeansAnalyzer, LIBDiffMeansAnalyzer


# Simulation function
def exact_simulate_one(i):
    sim_rng = np.random.default_rng(seed=i)
    theta_rng, y_rng, a_rng, s_rng = sim_rng.spawn(4)
    
    theta_abs_lim = 50.
    theta_min = -theta_abs_lim
    theta_max = theta_abs_lim
    theta_inc = 0.01
    theta_vals = np.arange(theta_min, theta_max+0.0001, theta_inc)
    n_theta_vals = theta_vals.size
    alpha = 0.05
    n_each = 5
    n = 2*n_each
    n_arange = np.arange(n)
    exact = n_each <= 10
    if exact:
        combs = list(combinations(range(n), n_each))
        n_combs = len(combs)
        n_combs_arange = np.arange(n_combs)
        a_vals = np.full((n_combs, n), False, dtype=bool)
        for i, c in enumerate(combs):
            a_vals[i, c] = True
            
    def sample_a(rng):
        if exact:
            a_idx = rng.choice(n_combs_arange)
            a_sample = a_vals[a_idx]
        else:
            a_idx = rng.choice(n_arange, n_each, replace=False)
            a_sample = np.isin(n_arange, a_idx)
        return a_sample
    
    prior_dist = DiscretizedDist(norm(loc=0., scale=10.), theta_vals)
    
    # Non-BRI methods
    prior_analyzer = PriorAnalyzer("Prior", alpha, prior_dist)
    diff_mean_analyzer = DiffMeansAnalyzer("DIM", alpha, prior_dist)
    lib_analyzer = LIBDiffMeansAnalyzer("LIB", alpha, prior_dist)
    
    # BRI methods
    rounded_analyzer = RoundedAnalyzer("BRI-R", alpha, prior_dist, n_each, n_theta_vals, a_vals, 0, uses_s=False)
    rank_sum_analyzer = RankSumAnalyzer("BRI-RS", alpha, prior_dist, n_each, n_theta_vals, a_vals, uses_s=False)
    neighborhood_analyzer = NeighborhoodAnalyzer("BRI-N", alpha, prior_dist, n_each, n_theta_vals, a_vals, 0.5, uses_s=False)
    bri_onesided_analyzer = BRIOneSidedAnalyzer("BRI-O", alpha, prior_dist, n_each, n_theta_vals, a_vals, 1., uses_s=False)
    bri_asymp_analyzer = BRIAsympAnalyzer("BRI-A", alpha, prior_dist, uses_s=False)
    
    # BRI methods with newly sampled statistic (should be exact)
    rounded_analyzer_s = RoundedAnalyzer("BRI-R*", alpha, prior_dist, n_each, n_theta_vals, a_vals, 0, uses_s=True)
    rank_sum_analyzer_s = RankSumAnalyzer("BRI-RS*", alpha, prior_dist, n_each, n_theta_vals, a_vals, uses_s=True)
    neighborhood_analyzer_s = NeighborhoodAnalyzer("BRI-N*", alpha, prior_dist, n_each, n_theta_vals, a_vals, 0.5, uses_s=True)
    bri_onesided_analyzer_s = BRIOneSidedAnalyzer("BRI-O*", alpha, prior_dist, n_each, n_theta_vals, a_vals, 1., uses_s=True)
    bri_asymp_analyzer_s = BRIAsympAnalyzer("BRI-A*", alpha, prior_dist, uses_s=True)
    
    methods = [
        prior_analyzer,
        diff_mean_analyzer,
        lib_analyzer,
        bri_onesided_analyzer,
        bri_asymp_analyzer,
        rounded_analyzer,
        neighborhood_analyzer,
        rank_sum_analyzer,
        bri_onesided_analyzer_s,
        bri_asymp_analyzer_s,
        rounded_analyzer_s,
        neighborhood_analyzer_s,
        rank_sum_analyzer_s,
    ]
    method_names = [m.name for m in methods]
    n_methods = len(methods)
    
    a = sample_a(a_rng)
    not_a = ~a
    y0 = y_rng.normal(size=n, scale=10.) + y_rng.gamma(20./5., scale=5./2., size=n)
    true_theta_i = prior_dist.rvs(theta_rng)
    y1 = y0 + true_theta_i
    y = not_a*y0 + a*y1
    a_s = sample_a(s_rng)
    not_a_s = ~a_s
    y_s = a_s*y1 + not_a_s*y0

    # Arrays to store results
    posterior_means = np.zeros(n_methods)
    lower_bounds = np.zeros(n_methods)
    upper_bounds = np.zeros(n_methods)
    nominal_coverage_rates = np.zeros(n_methods)
    for j, method in enumerate(methods):
        # Get posterior probs
        pm, lb, ub, cr = method.analyze_s(y, a, theta_vals, y_s=y_s, a_s=a_s)
        posterior_means[j] = pm
        lower_bounds[j] = lb
        upper_bounds[j] = ub
        nominal_coverage_rates[j] = cr
    return posterior_means, lower_bounds, upper_bounds, nominal_coverage_rates, true_theta_i, method_names