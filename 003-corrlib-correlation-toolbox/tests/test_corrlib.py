#-------- tests for corrlib ---------------------------------------------------
#-----------------------------------------------------------------------------
# Every test builds data where the true answer is known, then checks the
# library recovers it. Run from the project folder with:
#
#       python -m pytest tests

import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corrlib import Correlator
from corrlib import correlation_measures as cm
from corrlib import sampling_estimators as se
from corrlib import factor_removal as fr
from corrlib import random_matrix_cleaning as rmt
from corrlib import array_stacking as stack


#-------- Synthetic data with a known answer ---------------------------------
#-----------------------------------------------------------------------------
# Block structure: a market factor everyone shares, plus sector factors shared
# within blocks, plus independent noise. The true correlation matrix is known
# exactly from the loadings.
def block_model(n_blocks=5, per_block=8, market=0.4, sector=0.5, seed=0):
    N = n_blocks * per_block
    B = np.zeros((N, 1 + n_blocks))
    B[:, 0] = market
    for b in range(n_blocks):
        B[b * per_block:(b + 1) * per_block, 1 + b] = sector
    noise_var = 1 - (B ** 2).sum(axis=1)
    true_corr = B @ B.T + np.diag(noise_var)
    return true_corr, B, noise_var


def sample(true_corr, T, seed=0, df=None):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(true_corr)
    Z = rng.standard_normal((true_corr.shape[0], T))
    if df is not None:                                  # multivariate t: fat tails, same shape
        Z = Z / np.sqrt(rng.chisquare(df, T) / df)
    return L @ Z


def frob(A, B):
    return np.linalg.norm(A - B)


#-------- correlation_measures ------------------------------------------------
def test_pearson_matches_numpy():
    X = np.random.default_rng(1).standard_normal((6, 300))
    assert np.allclose(cm.pearson(X), np.corrcoef(X))


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 200))
    X[:, :50] = 0.0                                     # lots of ties, like solar at night
    rho, _ = stats.spearmanr(X.T)
    assert np.allclose(cm.spearman(X), rho)


def test_kendall_matrix_matches_scipy():
    X = np.random.default_rng(3).standard_normal((3, 150))
    K = cm.kendall_matrix(X)
    tau, _ = stats.kendalltau(X[0], X[2])
    assert np.isclose(K[0, 2], tau) and np.allclose(K, K.T)


def test_gaussian_rank_recovers_correlation_through_a_monotone_distortion():
    true_corr, _, _ = block_model()
    X = sample(true_corr, 4000, seed=4)
    distorted = np.exp(2 * X)                           # skewed, but the same ordering
    err_gr = frob(cm.gaussian_rank(distorted), true_corr)
    err_p = frob(cm.pearson(distorted), true_corr)
    assert err_gr < 0.5 * err_p                         # Pearson is fooled by the skew
    assert err_gr / true_corr.shape[0] < 0.03


def test_partial_correlation_recovers_known_precision():
    #a chain: 0 - 1 - 2. 0 and 2 are correlated, but only through 1
    P = np.array([[2.0, -0.8, 0.0], [-0.8, 2.0, -0.8], [0.0, -0.8, 2.0]])
    X = sample(np.linalg.inv(P), 50000, seed=5)
    pc = cm.partial(X)
    assert abs(cm.pearson(X)[0, 2]) > 0.1               # looks correlated
    assert abs(pc[0, 2]) < 0.02                         # but not directly
    assert np.isclose(pc[0, 1], 0.4, atol=0.02)         # -P01/sqrt(P00 P11)


#-------- random_matrix_cleaning ----------------------------------------------
def test_noise_band_and_removed_modes():
    lo, hi = rmt.noise_band(100, 400)
    assert np.isclose(lo, 0.25) and np.isclose(hi, 2.25)
    _, hi_removed = rmt.noise_band(100, 400, n_removed=19)
    assert hi_removed < hi


def test_pure_noise_eigenvalues_sit_inside_the_band():
    X = np.random.default_rng(6).standard_normal((100, 500))
    vals, _ = rmt.eigen_spectrum(cm.pearson(X))
    lo, hi = rmt.noise_band(100, 500)
    assert vals.max() < hi * 1.1 and vals.min() > lo * 0.8


def test_effective_rank_limits():
    assert np.isclose(rmt.effective_rank(np.eye(10)), 10)
    assert np.isclose(rmt.effective_rank(np.ones((10, 10))), 1)


@pytest.mark.parametrize("method", ["clip", "linear_shrinkage", "nonlinear_shrinkage"])
@pytest.mark.parametrize("T", [20, 40, 80, 400])          # q from 2 down to 0.1
def test_every_cleaner_beats_raw(method, T):
    true_corr, _, _ = block_model()
    errors_raw, errors_clean = [], []
    for seed in range(8):
        X = sample(true_corr, T, seed=seed)
        C = cm.pearson(X)
        errors_raw.append(frob(C, true_corr))
        errors_clean.append(frob(rmt.clean(C, method, T, X=X), true_corr))
    #short data: a clear improvement. plenty of data: little noise to remove, so just not worse
    margin = 0.97 if T <= 80 else 1.0
    assert np.mean(errors_clean) < margin * np.mean(errors_raw)


def test_nonlinear_shrinkage_is_best_on_structured_data():
    true_corr, _, _ = block_model()
    for T in [40, 80, 200]:
        errs = {m: [] for m in ["raw", "clip", "linear_shrinkage", "nonlinear_shrinkage"]}
        for seed in range(8):
            X = sample(true_corr, T, seed=seed)
            C = cm.pearson(X)
            for m in errs:
                errs[m].append(frob(rmt.clean(C, m, T, X=X), true_corr))
        means = {m: np.mean(v) for m, v in errs.items()}
        assert min(means, key=means.get) == "nonlinear_shrinkage", (T, means)


@pytest.mark.parametrize("method", ["raw", "clip", "linear_shrinkage", "nonlinear_shrinkage"])
def test_cleaned_matrix_is_a_valid_correlation_matrix(method):
    true_corr, _, _ = block_model()
    X = sample(true_corr, 60, seed=7)
    M = rmt.clean(cm.pearson(X), method, 60, X=X)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > -1e-8


@pytest.mark.parametrize("method", ["clip", "nonlinear_shrinkage"])
def test_cleaners_are_scale_equivariant(method):
    #cleaning a covariance must give the same answer as cleaning the correlation
    #and putting the variances back - the old clip failed this
    true_corr, _, _ = block_model()
    X = sample(true_corr, 100, seed=8)
    scales = np.linspace(0.5, 20, X.shape[0])
    cov = np.cov(X * scales[:, None])
    R, sd = rmt._to_correlation(cov)
    a = rmt.clean(cov, method, 100)
    b = rmt.clean(R, method, 100) * np.outer(sd, sd)
    assert np.allclose(a, b)


def test_ledoit_wolf_intensity_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(9)
    X = rng.standard_normal((30, 120)) + 0.5 * rng.standard_normal((1, 120))
    Xc = X - X.mean(axis=1, keepdims=True)
    C = Xc @ Xc.T / X.shape[1]
    ours = rmt.ledoit_wolf_intensity(C, X.shape[1], X=X)
    assert np.isclose(ours, sklearn.ledoit_wolf_shrinkage(X.T), rtol=1e-6)


def test_tyler_beats_sample_correlation_on_fat_tails():
    true_corr, _, _ = block_model(n_blocks=3, per_block=5)
    errs_t, errs_p = [], []
    for seed in range(4):
        X = sample(true_corr, 400, seed=seed, df=2.5)
        errs_t.append(frob(rmt.tyler_correlation(X), true_corr))
        errs_p.append(frob(cm.pearson(X), true_corr))
    assert np.mean(errs_t) < np.mean(errs_p)


#-------- factor_removal -------------------------------------------------------
def test_regress_out_known_market_reveals_sectors():
    true_corr, B, _ = block_model(market=0.6, sector=0.4)
    rng = np.random.default_rng(10)
    T = 5000
    factors = rng.standard_normal((B.shape[1], T))
    X = B @ factors + np.sqrt(1 - (B ** 2).sum(axis=1))[:, None] * rng.standard_normal((B.shape[0], T))

    residuals, loadings = fr.regress_out(X, factors[0])
    assert np.allclose(loadings[:, 0], 0.6, atol=0.03)  # recovers each variable's market loading
    R = cm.pearson(residuals)
    #what is left: sector covariance 0.16 over residual variance 1 - 0.36
    assert np.isclose(R[0, 1], 0.16 / 0.64, atol=0.04)
    assert abs(R[0, -1]) < 0.05


def test_remove_modes_leaves_rank_n_minus_k():
    true_corr, _, _ = block_model()
    C = cm.pearson(sample(true_corr, 2000, seed=11))
    residual, U, lam = fr.remove_modes_matrix(C, 2)
    vals = np.linalg.eigvalsh(residual)
    assert np.sum(vals < 1e-8) == 2
    assert np.allclose(np.diag(residual), 1.0)


def test_cleaners_do_not_put_removed_modes_back():
    true_corr, _, _ = block_model()
    X = sample(true_corr, 100, seed=12)
    residual, _, _ = fr.remove_modes_matrix(cm.pearson(X), 1)
    for method in ["clip", "linear_shrinkage", "nonlinear_shrinkage"]:
        M = rmt.clean(residual, method, 100, n_removed=1)
        assert np.sum(np.linalg.eigvalsh(M) < 1e-6) >= 1, method


#-------- sampling_estimators --------------------------------------------------
def test_effective_n_for_autocorrelated_pairs():
    #two independent AR(1) series with coefficient phi: n_eff = n (1 - phi^2) / (1 + phi^2)
    rng = np.random.default_rng(13)
    phi, T = 0.7, 20000
    X = np.zeros((2, T))
    e = rng.standard_normal((2, T))
    for t in range(1, T):
        X[:, t] = phi * X[:, t - 1] + e[:, t]
    expected = T * (1 - phi ** 2) / (1 + phi ** 2)
    assert np.isclose(se.effective_n_pair(X[0], X[1], max_lag=40), expected, rtol=0.1)
    assert np.isclose(se.effective_n(X, max_lag=40), expected, rtol=0.1)


def test_hayashi_yoshida_equals_realised_on_a_common_clock():
    rng = np.random.default_rng(14)
    p = 100 * np.exp(np.cumsum(0.001 * rng.standard_normal((2, 500)), axis=1))
    t = np.arange(500.0)
    C_hy = se.hayashi_yoshida_matrix([t, t], [p[0], p[1]])
    assert np.allclose(C_hy, se.realised(p))


#-------- array_stacking -------------------------------------------------------
def synthetic_array(n_sensors=10, T=3000, seed=15):
    rng = np.random.default_rng(seed)
    time = np.arange(T)
    wave = np.sin(2 * np.pi * (time - 2000) / 60) * np.exp(-((time - 2000) / 150.0) ** 2)
    wave[time < 2000] = 0
    shared = rng.standard_normal(T)
    exposure = np.linspace(0.2, 2.0, n_sensors)         # how close each sensor is to the road
    X = wave + exposure[:, None] * shared + 0.5 * rng.standard_normal((n_sensors, T))
    return X, wave


def test_mvdr_beats_averaging_on_correlated_noise():
    X, wave = synthetic_array()
    window = (0, 1800)
    s_simple, _ = stack.simple(X)
    s_mvdr, w = stack.mvdr(X, clean=None, noise_window=window)
    assert np.isclose(w.sum(), 1.0)                     # unit gain on the signal
    assert stack.gain(s_mvdr, X[0], window) > stack.gain(s_simple, X[0], window) + 6


def test_min_variance_with_diagonal_covariance():
    var = np.array([1.0, 2.0, 4.0])
    w = stack.min_variance(np.diag(var))
    assert np.allclose(w, (1 / var) / (1 / var).sum())


def test_align_shifts_without_wrapping():
    rng = np.random.default_rng(16)
    base = np.zeros(400)
    base[100:140] = rng.standard_normal(40)
    X = np.vstack([base, stack._shift(base, 25), stack._shift(base, -10)])
    aligned, shifts = stack.align(X, max_lag=50)
    assert list(shifts) == [0, 25, -10]
    assert np.allclose(aligned[1][:375], base[:375])
    assert np.all(aligned[1][375:] == 0)                # zero-padded, not wrapped


def test_explain_recovers_loadings_on_two_drivers():
    rng = np.random.default_rng(17)
    D = rng.standard_normal((2, 3000))
    true = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 2.0]])
    X = true @ D + 0.1 * rng.standard_normal((3, 3000))
    loadings, residuals, shared = stack.explain(X, D)
    assert np.allclose(loadings, true, atol=0.02)
    assert np.all(shared > 0.95)


def test_common_mode_returns_stack_and_weights():
    X, wave = synthetic_array()
    stacked, w = stack.common_mode(X)
    assert stacked.shape == (X.shape[1],) and np.isclose(w.sum(), 1.0)


#-------- Correlator -----------------------------------------------------------
@pytest.mark.parametrize("measure", ["pearson", "spearman", "gaussian_rank"])
@pytest.mark.parametrize("clean", ["raw", "clip", "ledoit_wolf", "nonlinear_shrinkage"])
def test_correlator_runs_every_configuration(measure, clean):
    true_corr, _, _ = block_model()
    X = sample(true_corr, 200, seed=18)
    c = Correlator(measure=measure, remove=1, clean=clean, effective_n="bartlett")
    M = c.fit(X)
    assert M.shape == true_corr.shape
    assert np.allclose(np.diag(M), 1.0)
    assert c.n_removed_ == 1 and c.q_ > 0


def test_correlator_dataframe_input_and_known_factor():
    pd = pytest.importorskip("pandas")
    true_corr, B, _ = block_model(market=0.6, sector=0.4)
    rng = np.random.default_rng(19)
    T = 3000
    factors = rng.standard_normal((B.shape[1], T))
    X = B @ factors + np.sqrt(1 - (B ** 2).sum(axis=1))[:, None] * rng.standard_normal((B.shape[0], T))
    df = pd.DataFrame(X.T, columns=[f"a{i}" for i in range(X.shape[0])])

    c = Correlator(remove=[factors[0]], clean="ledoit_wolf")
    M = c.fit(df)
    assert c.labels_[0] == "a0"
    assert c.n_effective_ == T - 1                      # one factor regressed, one observation spent
    assert np.isclose(M[0, 1], 0.16 / 0.64, atol=0.05) and abs(M[0, -1]) < 0.06


def test_correlator_partial_output():
    P = np.array([[2.0, -0.8, 0.0], [-0.8, 2.0, -0.8], [0.0, -0.8, 2.0]])
    X = sample(np.linalg.inv(P), 20000, seed=20)
    M = Correlator(output="partial").fit(X)
    assert abs(M[0, 2]) < 0.03


def test_correlator_plot_runs():
    import matplotlib
    matplotlib.use("Agg")
    true_corr, _, _ = block_model()
    c = Correlator(clean="nonlinear_shrinkage", remove=1)
    c.fit(sample(true_corr, 300, seed=21))
    fig = c.plot()
    assert len(fig.axes) >= 4


def test_bulk_variance_ignores_the_structure():
    #block model: the noise eigenvalues are 1 - market^2 - sector^2 = 0.59; the spikes must not count
    true_corr, _, noise_var = block_model()
    X = sample(true_corr, 4000, seed=22)
    vals, _ = rmt.eigen_spectrum(cm.pearson(X))
    assert np.isclose(rmt.bulk_variance(vals, 4000), noise_var[0], rtol=0.05)
    assert rmt.bulk_variance(vals, 4000) < 0.8 * vals.mean()   # the naive average is far too high


@pytest.mark.parametrize("q", [0.1, 0.5, 1.0, 1.5])
def test_noise_median_matches_simulation(q):
    rng = np.random.default_rng(23)
    N = 400
    vals = np.linalg.eigvalsh(np.cov(rng.standard_normal((N, int(N / q))), bias=True))
    assert np.isclose(rmt._noise_median(q), np.median(vals), rtol=0.03)
