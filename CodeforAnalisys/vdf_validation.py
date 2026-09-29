"""Held-out particle validation of density fits in the explicitly declared u coordinate."""
import numpy as np
from analysis_contract import effective_sample_size


def predictive_check(u, weights, rng, fitter, sigma_multiplier=6., bins=160):
    u, weights = np.asarray(u), np.asarray(weights)
    train = rng.random(u.size) < .7
    if min(train.sum(), (~train).sum()) < 100:
        return {'model_validation_status': 'UNVERIFIED_insufficient_particles'}
    mean = np.average(u[train], weights=weights[train])
    sigma = np.sqrt(np.average((u[train]-mean)**2, weights=weights[train]))
    if not sigma > 0:
        return {'model_validation_status': 'UNVERIFIED_zero_width'}
    edges=np.linspace(-sigma_multiplier*sigma,sigma_multiplier*sigma,bins+1)
    x=.5*(edges[1:]+edges[:-1]); width=np.diff(edges)
    def histogram(mask):
        counts,_=np.histogram(u[mask]-mean,bins=edges,weights=weights[mask])
        raw,_=np.histogram(u[mask]-mean,bins=edges)
        return counts/(weights[mask].sum()*width),raw
    y, count=histogram(train); test, test_count=histogram(~train)
    keep=count>=8
    if keep.sum()<12:
        return {'model_validation_status': 'UNVERIFIED_insufficient_bins'}
    m,k=fitter(x[keep],y[keep],sigma)
    ym=m[0]*np.exp(-.5*(x/m[1])**2)
    yk=k[0]*(1+x*x/((2*k[2]-3)*k[1]**2))**(-k[2])
    good=keep & (test_count>=8)
    if not good.any():
        return {'model_validation_status': 'UNVERIFIED_no_supported_test_bins'}
    em=float(np.mean((test[good]-ym[good])**2));ek=float(np.mean((test[good]-yk[good])**2))
    return {'model_validation_status':'predictive_scores_only_not_physical_identification',
            'heldout_mse_maxwellian':em,'heldout_mse_kappa':ek,
            'heldout_mse_improvement':em-ek,'heldout_supported_bins':int(good.sum()),
            'heldout_n_effective':effective_sample_size(weights[~train]),
            'mixture_ambiguity':'A global fit cannot exclude a mixture of local populations'}


def mixture_tail_budget(t_par, t_perp, counts, kappa_fit, t_par_all, t_perp_all, kappa_upper=None):
    """How much of a measured kappa tail a spatial temperature mixture explains.

    A kappa distribution is exactly a superposition of Maxwellians whose
    inverse temperature is Gamma-distributed, so a window that averages
    regions of different temperature can show a kappa-like tail with no
    suprathermal particles anywhere. For local Maxwellians with whitened
    temperature tau = (T_par/T_par,all + 2 T_perp/T_perp,all)/3 the whitened
    kurtosis of the mixture is K = (5/3)(1 + CV^2(tau)), the value of a kappa
    distribution with kappa_mix = 5/2 + 1/CV^2(tau) -- the relation of the
    kappa_eff estimator. The tail strength 1/(kappa - 5/2) is therefore
    additive: fraction = CV^2 / (1/(kappa_fit - 5/2)) of the measured tail is
    what the resolved temperature spread alone produces.

    The Gaussian sampling variance 2/(3N) of each block is removed; a heavier
    local tail has more, so the resolved spread (and the mixture fraction) is
    if anything overestimated, which keeps an "intrinsic" verdict conservative.
    Temperature variation below the block size stays inside the local VDFs.
    `kappa_upper` (e.g. kappa + 2 sigma) that is infinite marks a fit
    consistent with a Maxwellian: there is then no tail to apportion.
    """
    t_par, t_perp, n = (np.asarray(x, float).ravel() for x in (t_par, t_perp, counts))
    ok = np.isfinite(t_par) & np.isfinite(t_perp) & (n > 0)
    out = {'mixture_blocks': int(ok.sum()), 'mixture_scope': 'macro-cell scale; finer variation stays in local VDFs'}
    if ok.sum() < 4 or not (t_par_all > 0 and t_perp_all > 0):
        return {**out, 'mixture_verdict': 'undetermined_too_few_blocks'}
    tau = (t_par[ok] / t_par_all + 2 * t_perp[ok] / t_perp_all) / 3
    w = n[ok] / n[ok].sum()
    mean = float(np.sum(w * tau))
    cv2 = float(np.sum(w * (tau - mean) ** 2)) / mean ** 2
    sampling = float(np.sum(w * tau ** 2 * 2 / (3 * n[ok]))) / mean ** 2
    resolved = max(cv2 - sampling, 0.0)
    out.update(mixture_cv2_observed=cv2, mixture_cv2_sampling=sampling,
               kappa_mixture=2.5 + 1 / resolved if resolved > 0 else float('inf'))
    if kappa_fit is None or np.isnan(kappa_fit) or (np.isfinite(kappa_fit) and kappa_fit <= 2.5):
        return {**out, 'mixture_verdict': 'undetermined_kappa_fit'}
    if not np.isfinite(kappa_fit) or (kappa_upper is not None and not np.isfinite(kappa_upper)):
        return {**out, 'mixture_fraction_of_tail': float('nan'), 'mixture_verdict': 'no_significant_tail'}
    tail = 1 / (kappa_fit - 2.5)
    fraction = resolved / tail
    # CV^2 from a few hundred blocks carries ~10 % sampling error, so "most of
    # the tail" rather than "all of it" is the resolvable statement.
    verdict = ('mixture_explains_tail' if fraction >= 0.75 else
               'intrinsic_at_macrocell_scale' if fraction <= 0.25 else 'partial_mixture')
    return {**out, 'mixture_fraction_of_tail': fraction, 'mixture_verdict': verdict}
