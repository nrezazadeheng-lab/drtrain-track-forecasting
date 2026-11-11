#!/usr/bin/env python3
"""Post hoc checks for the second revision of the DR-Train track-monitoring study (Reviewer 8, second round).

This script reuses railtrack_revision_analyses.py (version R1.0, unchanged; its SHA-256 is checked) and, through it,
the frozen study pipeline (v1.2.0, unchanged; its SHA-256 is checked against FROZEN_PROTOCOL.json). The frozen
protocol, the foundation files and all earlier results are only read. Nothing is re-fitted on holdout data: with
--allow-holdout-diagnostics the digest-checked holdout features are read again only to describe the frozen models.
All outputs go to <study-dir>/revision_R2/.

Checks:
  N  observation variance of the daily median in Eq. (2): interval calibration by the number of passages N,
     squared error against 1/N, and the efficiency of the median under the observed passage noise      [R8.3]
  T  day-level tests that allow for temporal dependence (Newey-West Diebold-Mariano and moving-block
     bootstrap over observation days), and skill within the seasonal blocks                             [R8.4]
  P  inspection effort: share of the route that each ranking needs to find the same high-response bin-days  [R8.6]
"""
import argparse
import hashlib
import importlib.util
import math
import platform
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

MODULE_VERSION = 'R2.0'
R1_SHA256 = '48d40d8403500ac8762b90c1e211f3b4b64a43e259278344ed23bd140c1dac04'
OUT_NAME = 'revision_R2'
DEFAULTS = {
    'r1_module': '/data2/nr1/V1_RailTrack/railtrack_revision_analyses.py',
    'pipeline': '/data2/nr1/V1_RailTrack/railtrack_study_pipeline.py',
    'foundation_dir': '/data2/nr1/V1_RailTrack/FOUNDATION_for_new_chat',
    'study_dir': '/data2/nr1/V1_RailTrack/STUDY_v1',
}
R1 = None   # railtrack_revision_analyses module
P = None    # frozen pipeline module (loaded by R1.setup)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_r1(path, allow_mismatch):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Revision R1 module not found: {path}')
    sha = sha256_file(path)
    if sha != R1_SHA256:
        msg = f'railtrack_revision_analyses.py differs from the archived R1.0 version ({sha[:12]} against {R1_SHA256[:12]}).'
        if not allow_mismatch:
            raise RuntimeError(msg + ' Restore the archived file (it must not be edited).')
        print('WARNING ' + msg, flush=True)
    spec = importlib.util.spec_from_file_location('railtrack_revision_analyses', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['railtrack_revision_analyses'] = mod
    spec.loader.exec_module(mod)
    return mod, sha


def log(msg):
    R1.log(msg)


def finite(x):
    return float(x) if x is not None and np.isfinite(x) else float('nan')


# ----------------------------------------------------------------------------
# Forecast record sets (I1, primary model)
# ----------------------------------------------------------------------------
def record_sets(ctx, with_fixed=False, with_30=True):
    key = ctx.prim
    st = R1.dev_store(ctx, key)
    out = [('development rolling origin', 'next observation day', st['recs']['next observation day'])]
    if with_30:
        out.append(('development rolling origin', 'at least 30 days ahead', st['recs']['at least 30 days ahead']))
    hs = R1.holdout_store(ctx, key)
    if hs is not None:
        out.append(('holdout sequential', 'next observation day', hs['recs']['holdout sequential']))
        if with_fixed:
            out.append(('holdout fixed origin', 'fixed origin', hs['recs']['holdout fixed origin']))
    return out


def episode_index(clusters):
    keys, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    return keys, [np.where(inv == k)[0] for k in range(len(keys))]


# ----------------------------------------------------------------------------
# N: observation variance of the daily median (Eq. 2)
# ----------------------------------------------------------------------------
N_GROUPS = [('1', lambda n: n == 1), ('2', lambda n: n == 2), ('3', lambda n: n == 3), ('4 or more', lambda n: n >= 4),
            ('3 or more', lambda n: n >= 3), ('all', lambda n: n >= 1)]


def ols_two(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def passage_residuals(model, table):
    """Within-day passage residuals of the anomaly, for bin-days with at least two QC passages.

    For the k passages of one bin-day, r = (e - mean e) * sqrt(k / (k - 1)) has the passage-noise variance; the
    day-level component and the underlying level cancel. Returns pooled residuals, the pooled within-day variance
    and the number of bin-days by k.
    """
    res, ss, df = [], 0.0, 0
    kcount = {}
    for d in P.DIRECTIONS:
        A = P.apply_indicator_model(model, table, d)
        E, qc, pord = A['E'], A['qc'], A['pord']
        for o in np.unique(pord[qc]).tolist():
            M = E[qc & (pord == o)]
            if M.shape[0] < 2:
                continue
            ok = np.isfinite(M)
            k = ok.sum(axis=0)
            use = k >= 2
            if not use.any():
                continue
            Mz = np.where(ok, M, 0.0)
            mean = Mz.sum(axis=0) / np.maximum(k, 1)
            dev = np.where(ok, M - mean[None, :], np.nan)[:, use]
            kk = k[use]
            ss += float(np.nansum(dev ** 2))
            df += int(np.sum(kk - 1))
            scale = np.sqrt(kk / (kk - 1.0))
            r = (dev * scale[None, :])[np.isfinite(dev)]
            res.append(r)
            for v in kk.tolist():
                kcount[int(v)] = kcount.get(int(v), 0) + 1
    r = np.concatenate(res) if res else np.zeros(0)
    return r, (ss / df if df else float('nan')), kcount


def median_factors(r, rng, kmax=6, draws=200000):
    """k Var(median of k) / Var(single) for draws from the pooled residuals and from a Gaussian; 1 means the 1/N law."""
    out = []
    r = r / np.std(r)
    for k in range(1, kmax + 1):
        X = rng.choice(r, size=(draws, k), replace=True)
        G = rng.standard_normal((draws, k))
        f_obs = k * float(np.var(np.median(X, axis=1)))
        f_gau = k * float(np.var(np.median(G, axis=1)))
        e_obs = float(np.var(np.median(X, axis=1)) / np.var(np.mean(X, axis=1)))
        out.append((k, f_obs, f_gau, e_obs))
    return out


def block_N(ctx):
    log('Check N: observation variance of the daily median (Eq. 2)')
    pm, B = ctx.pm, ctx.B
    z80, z95 = stats.norm.ppf(0.9), stats.norm.ppf(0.975)
    cal_rows, reg_rows, eff_rows = [], [], []
    summ = OrderedDict()
    for scope, hl, R in record_sets(ctx):
        y, mu = R['y'], R[pm]
        var = np.maximum(R['var_' + pm], 1e-12)
        sd = np.sqrt(var)
        z = (y - mu) / sd
        e2 = (y - mu) ** 2
        N = np.asarray(R['N'], float)
        rp = float(np.median(R['vc_passage'] * np.maximum(N, 1.0)))
        rd = float(np.median(R['vc_day']))
        ep = np.asarray(R['episode']).astype(str)
        for gname, gf in N_GROUPS:
            g = gf(N)
            n = int(g.sum())
            if n == 0:
                continue
            c80 = (np.abs(z[g]) <= z80).astype(float)
            c80m, c80lo, c80hi, C = P.cluster_bootstrap_mean(c80, ep[g], B, ctx.rng)
            zz = z[g]
            cal_rows.append([scope, ctx.prim, hl, gname, n, n / len(y), float(np.mean(N[g])), C, c80m, c80lo, c80hi,
                             float(np.mean(np.abs(zz) <= z95)), float(np.mean(zz ** 2)),
                             float(1.4826 * np.median(np.abs(zz - np.median(zz)))), float(np.mean(np.abs(zz) > 3)),
                             float(np.mean(np.abs(y[g] - mu[g]))), float(np.mean(e2[g])), float(np.mean(var[g])),
                             float(np.mean(e2[g]) / np.mean(var[g])), float(np.mean(R['vc_passage'][g] / var[g]))])
            summ[f'{scope}|{hl}|N {gname}'] = {'bin_days': n, 'share': n / len(y), 'cov80': c80m,
                                                'cov80_95': [c80lo, c80hi], 'mean_z2': float(np.mean(zz ** 2)),
                                                'robust_sd_z': cal_rows[-1][13]}
        # Squared error against 1/N: the slope estimates the passage-noise variance actually present in the median.
        x = 1.0 / np.maximum(N, 1.0)
        keys, idx = episode_index(ep)
        for variant, keep in (('all bin-days', np.ones(len(y), bool)),
                              ('largest 0.5% of squared errors removed', e2 <= np.quantile(e2, 0.995))):
            a, b = ols_two(x[keep], e2[keep])
            boots = []
            for _ in range(B):
                ii = np.concatenate([idx[j] for j in ctx.rng.integers(0, len(keys), len(keys))])
                ii = ii[keep[ii]]
                if len(np.unique(x[ii])) > 1:
                    boots.append(ols_two(x[ii], e2[ii]))
            boots = np.array(boots) if boots else np.full((1, 2), np.nan)
            alo, ahi = np.nanpercentile(boots[:, 0], [2.5, 97.5])
            blo, bhi = np.nanpercentile(boots[:, 1], [2.5, 97.5])
            reg_rows.append([scope, ctx.prim, hl, variant, int(keep.sum()), len(keys), a, alo, ahi, b, blo, bhi, rp,
                             b / rp, blo / rp, bhi / rp, float(np.mean((var - R['vc_passage'])[keep])), rd,
                             rd / (rd + rp)])
            summ[f'{scope}|{hl}|slope on 1/N ({variant})'] = {'slope': b, 'slope_95': [blo, bhi], 'fitted_passage_var': rp,
                                                             'ratio': b / rp, 'ratio_95': [blo / rp, bhi / rp],
                                                             'implied_intraday_correlation': rd / (rd + rp)}
    # Efficiency of the median under the observed passage noise.
    st = R1.dev_store(ctx, ctx.prim)
    sources = [('development (all QC passages)', st['model'], ctx.found.table,
                float(st['fits'][pm]['params'][2]))]
    hs = R1.holdout_store(ctx, ctx.prim)
    if hs is not None:
        sources.append(('holdout (all QC passages)', hs['model'], ctx.hold, float(hs['fits'][pm]['params'][2])))
    for name, model, table, rp_fit in sources:
        r, wvar, kcount = passage_residuals(model, table)
        if len(r) < 100:
            log(f'  {name}: too few within-day residuals ({len(r)})')
            continue
        sdr = float(np.std(r))
        kurt = float(stats.kurtosis(r, fisher=True))
        tail = float(np.mean(np.abs(r) > 3 * sdr))
        kc = '; '.join(f'{k}: {v}' for k, v in sorted(kcount.items()))
        for k, f_obs, f_gau, e_obs in median_factors(r, ctx.rng):
            eff_rows.append([name, ctx.prim, len(r), math.sqrt(wvar), math.sqrt(rp_fit), kurt, tail, kc, k, f_obs, f_gau,
                             e_obs])
        summ[f'median efficiency|{name}'] = {
            'residuals': int(len(r)), 'within_day_passage_sd': math.sqrt(wvar), 'fitted_passage_sd': math.sqrt(rp_fit),
            'excess_kurtosis': kurt, 'share_beyond_3sd': tail, 'bin_days_by_k': kcount,
            'factor_k_var_median_over_var_single': {int(row[8]): row[9] for row in eff_rows if row[0] == name},
            'gaussian_factor': {int(row[8]): row[10] for row in eff_rows if row[0] == name}}
    R1.wcsv(ctx, 'N1_calibration_by_passages.csv',
            ['scope', 'indicator', 'horizon', 'passages_N', 'bin_days', 'share_of_bin_days', 'mean_N', 'episodes',
             'coverage80', 'coverage80_lo95', 'coverage80_hi95', 'coverage95', 'mean_z_squared', 'robust_sd_z',
             'share_abs_z_above_3', 'mae', 'mean_squared_error', 'mean_predictive_variance',
             'squared_error_over_predictive_variance', 'mean_passage_share_of_predictive_variance'], cal_rows)
    R1.wcsv(ctx, 'N2_squared_error_against_inverse_N.csv',
            ['scope', 'indicator', 'horizon', 'variant', 'bin_days', 'episodes', 'intercept', 'intercept_lo95',
             'intercept_hi95', 'slope_on_1_over_N', 'slope_lo95', 'slope_hi95', 'fitted_passage_variance',
             'slope_over_fitted', 'slope_over_fitted_lo95', 'slope_over_fitted_hi95',
             'model_mean_variance_excluding_passage_term', 'fitted_day_variance', 'implied_intraday_correlation'],
            reg_rows)
    R1.wcsv(ctx, 'N3_median_efficiency.csv',
            ['source', 'indicator', 'within_day_residuals', 'within_day_passage_sd', 'fitted_passage_sd',
             'excess_kurtosis', 'share_beyond_3sd', 'bin_days_by_passages', 'k',
             'k_var_median_over_var_single_observed_noise', 'k_var_median_over_var_single_gaussian',
             'var_median_over_var_mean_observed_noise'], eff_rows)
    ctx.summary['N'] = summ


# ----------------------------------------------------------------------------
# T: day-level tests allowing for temporal dependence
# ----------------------------------------------------------------------------
def nw_lrv(d, L):
    T = len(d)
    x = d - d.mean()
    s = float(np.dot(x, x) / T)
    for lag in range(1, min(L, T - 1) + 1):
        s += 2.0 * (1.0 - lag / (L + 1.0)) * float(np.dot(x[lag:], x[:-lag]) / T)
    return max(s, 1e-300)


def dm_test(d, L):
    """Diebold-Mariano statistic with Newey-West variance and the Harvey-Leybourne-Newbold correction (h = 1)."""
    T = len(d)
    stat = float(d.mean() / math.sqrt(nw_lrv(d, L) / T))
    hln = stat * math.sqrt((T - 1.0) / T)
    return stat, hln, float(stats.t.sf(hln, T - 1))


def mbb(d, b, B, rng):
    """Moving-block bootstrap over consecutive observation days: one-sided p for a positive mean and 95% interval."""
    T = len(d)
    b = max(1, min(b, T))
    nb = int(math.ceil(T / b))
    starts = rng.integers(0, T - b + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(B, -1)[:, :T]
    means = d[idx].mean(axis=1)
    p = float(np.mean(means - d.mean() >= d.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return p, float(lo), float(hi)


def block_T(ctx):
    log('Check T: day-level tests allowing for temporal dependence')
    pm, B = ctx.pm, ctx.B
    rows, blk_rows = [], []
    summ = OrderedDict()
    for scope, hl, R in record_sets(ctx, with_fixed=True, with_30=False):
        days, inv = np.unique(R['day'], return_inverse=True)
        T = len(days)
        cnt = np.bincount(inv).astype(float)
        loss = {m: np.bincount(inv, weights=np.abs(R['y'] - R[m])) / cnt
                for m in (pm, 'persistence', 'fingerprint', 'moving_mean')}
        auto = int(math.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
        for base in ('persistence', 'fingerprint'):
            d = loss[base] - loss[pm]
            r1 = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if T > 3 else float('nan')
            k = int(np.sum(d > 0))
            p_sign = float(stats.binomtest(k, T, 0.5, alternative='greater').pvalue)
            skill_days = float(1.0 - loss[pm].mean() / loss[base].mean())
            for L in sorted({0, 1, 2, 3, auto}):
                if L >= T:
                    continue
                dm, hln, p = dm_test(d, L)
                rows.append([scope, ctx.prim, hl, base, T, float(d.mean()), skill_days, r1, k, p_sign, 'Newey-West',
                             L, dm, hln, p, '', '', ''])
            for bl in (2, 3):
                if bl < T:
                    p, lo, hi = mbb(d, bl, max(B, 2000) * 5, ctx.rng)
                    rows.append([scope, ctx.prim, hl, base, T, float(d.mean()), skill_days, r1, k, p_sign,
                                 'moving-block bootstrap', bl, '', '', p, lo, hi, ''])
            summ[f'{scope}|{base}'] = {'days': T, 'mean_daily_mae_difference': float(d.mean()),
                                       'lag1_autocorrelation': r1, 'days_better': f'{k} of {T}', 'auto_lag': auto,
                                       'dm_hln_p_auto_lag': dm_test(d, min(auto, T - 2))[2],
                                       'mbb_p_block3': mbb(d, min(3, T - 1), max(B, 2000) * 5, ctx.rng)[0]}
        # Seasonal blocks: episodes formed with a 14-day gap.
        R14 = R1.regroup(R, ctx.FC, 14)
        for base in ('persistence', 'fingerprint'):
            am, ab = np.abs(R14['y'] - R14[pm]), np.abs(R14['y'] - R14[base])
            keys, sk = P.episode_skills(am, ab, R14['episode'])
            for kk, s in zip(keys.tolist(), sk.tolist()):
                m = np.asarray(R14['episode']).astype(str) == kk
                blk_rows.append([scope, ctx.prim, hl, base, kk, int(m.sum()), int(len(np.unique(R14['day'][m]))),
                                 float(s)])
    R1.wcsv(ctx, 'T1_day_level_tests.csv',
            ['scope', 'indicator', 'horizon', 'baseline', 'observation_days', 'mean_daily_mae_difference',
             'skill_from_daily_means', 'lag1_autocorrelation_of_difference', 'days_model_better', 'day_sign_test_p',
             'method', 'lag_or_block_length', 'dm_statistic', 'dm_hln_statistic', 'one_sided_p', 'mean_lo95',
             'mean_hi95', 'note'], rows)
    R1.wcsv(ctx, 'T2_seasonal_blocks.csv',
            ['scope', 'indicator', 'horizon', 'baseline', 'block_first_day', 'bin_days', 'observation_days', 'skill'],
            blk_rows)
    ctx.summary['T'] = summ


# ----------------------------------------------------------------------------
# P: inspection effort
# ----------------------------------------------------------------------------
GRID = [round(0.01 * i, 2) for i in range(1, 61)]


def block_P(ctx):
    log('Check P: inspection effort for the same share of high-response bin-days')
    pm = ctx.pm
    lim = P.priority_limit(ctx.cfg)
    curve_rows, eff_rows = [], []
    summ = OrderedDict()
    for scope, hl, R in record_sets(ctx, with_fixed=True, with_30=True):
        keys = R1.instance_keys(R)
        sd = R1.pred_sd(R, pm)
        prob = 1 - stats.norm.cdf((lim - R[pm]) / sd)
        scores = OrderedDict([('local level, exceedance probability', prob), ('persistence', R['persistence']),
                              ('moving mean', R['moving_mean']), ('historical mean', R['historical_mean'])])
        hits = {(s, b): 0 for s in scores for b in GRID}
        sel = {b: 0 for b in GRID}
        n_ev, n_inst = 0, 0
        for kk in np.unique(keys):
            s = keys == kk
            n = int(s.sum())
            if n < ctx.FC['min_bins_ranking']:
                continue
            ev = R['y'][s] > lim
            if not ev.any():
                continue
            n_inst += 1
            n_ev += int(ev.sum())
            for b in GRID:
                sel[b] += max(1, int(round(b * n)))
            for sname, sc in scores.items():
                cum = np.cumsum(ev[np.argsort(-sc[s], kind='stable')])
                for b in GRID:
                    hits[(sname, b)] += int(cum[max(1, int(round(b * n))) - 1])
        if n_ev == 0:
            continue
        rec = {sname: np.array([hits[(sname, b)] / n_ev for b in GRID]) for sname in scores}
        for sname in scores:
            for j, b in enumerate(GRID):
                curve_rows.append([scope, ctx.prim, hl, sname, b, n_inst, n_ev, rec[sname][j],
                                   hits[(sname, b)] / sel[b]])
        ll = rec['local level, exceedance probability']
        for target in (0.10, 0.20, 0.30, 0.50):
            j = GRID.index(target)
            for sname in scores:
                if sname.startswith('local level'):
                    continue
                need = next((GRID[i] for i in range(len(GRID)) if rec[sname][i] >= ll[j] - 1e-12), float('nan'))
                eff_rows.append([scope, ctx.prim, hl, target, ll[j], sname, rec[sname][j],
                                 ll[j] / rec[sname][j] - 1 if rec[sname][j] > 0 else float('nan'), need,
                                 need / target if np.isfinite(need) else float('nan'), target])
                if abs(target - 0.20) < 1e-9:
                    summ[f'{scope}|{hl}|{sname}'] = {'recall_local_level_20pct': float(ll[j]),
                                                     'recall_baseline_20pct': float(rec[sname][j]),
                                                     'relative_gain': eff_rows[-1][7],
                                                     'share_needed_by_baseline': need}
    R1.wcsv(ctx, 'P1_inspection_curves.csv',
            ['scope', 'indicator', 'horizon', 'ranking_score', 'share_of_route_inspected', 'decision_instances',
             'events', 'recall', 'precision'], curve_rows)
    R1.wcsv(ctx, 'P2_inspection_effort.csv',
            ['scope', 'indicator', 'horizon', 'share_inspected', 'recall_local_level', 'baseline', 'recall_baseline',
             'relative_gain_of_local_level', 'share_baseline_needs_for_same_recall', 'effort_ratio_baseline',
             'random_selection_recall'], eff_rows)
    ctx.summary['P'] = summ


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def finish(ctx, r1_sha, my_sha):
    P.write_json(ctx.out / 'REVISION2_SUMMARY.json', OrderedDict([('checks', ctx.checks), ('summary', ctx.summary)]))
    ctx.files.append('REVISION2_SUMMARY.json')
    P.write_text(ctx.out / 'revision2_log.txt', '\n'.join(P.LOG_LINES) + '\n')
    files = OrderedDict((f, sha256_file(ctx.out / f)) for f in sorted(set(ctx.files)) if (ctx.out / f).is_file())
    man = OrderedDict([
        ('created_utc', datetime.now(timezone.utc).isoformat(timespec='seconds')),
        ('module', 'railtrack_revision2_checks.py'), ('module_version', MODULE_VERSION), ('module_sha256', my_sha),
        ('purpose', 'Post hoc checks for the second revision; the confirmatory protocol and its results are unchanged.'),
        ('revision_R1_module_sha256', r1_sha), ('frozen_pipeline_sha256', ctx.pipeline_sha), ('protocol_sha256', ctx.psha),
        ('foundation_sha256', ctx.found.sha256), ('holdout_features_read', ctx.hold is not None),
        ('holdout_features_sha256', ctx.fsha), ('bootstrap', ctx.B), ('python', sys.version.split()[0]),
        ('numpy', np.__version__), ('scipy', __import__('scipy').__version__), ('host', platform.node()),
        ('run_minutes', round((time.monotonic() - ctx.t_start) / 60, 1)), ('checks', ctx.checks), ('files', files)])
    P.write_json(ctx.out / 'REVISION2_MANIFEST.json', man)
    return man


def main(argv=None):
    global R1, P
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--r1-module', default=DEFAULTS['r1_module'])
    ap.add_argument('--pipeline', default=DEFAULTS['pipeline'])
    ap.add_argument('--foundation-dir', default=DEFAULTS['foundation_dir'])
    ap.add_argument('--study-dir', default=DEFAULTS['study_dir'])
    ap.add_argument('--allow-holdout-diagnostics', action='store_true',
                    help='read the frozen holdout features again to describe the frozen models (post hoc)')
    ap.add_argument('--checks', default='NTP')
    ap.add_argument('--bootstrap', type=int, default=None)
    ap.add_argument('--no-ledger', action='store_true', help='do not record the run in the protocol ledger')
    ap.add_argument('--allow-r1-mismatch', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--allow-pipeline-mismatch', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    my_sha = sha256_file(Path(__file__).resolve())
    R1, r1_sha = load_r1(args.r1_module, args.allow_r1_mismatch)
    R1.OUT_NAME = OUT_NAME   # outputs to revision_R2; revision_R1 is not touched
    ns = argparse.Namespace(pipeline=args.pipeline, foundation_dir=args.foundation_dir, study_dir=args.study_dir,
                            allow_holdout_diagnostics=args.allow_holdout_diagnostics, temperature_csv=None,
                            bootstrap=args.bootstrap, allow_pipeline_mismatch=args.allow_pipeline_mismatch, blocks='')
    ctx = R1.setup(ns)
    P = R1.P
    log(f'Revision 2 checks {MODULE_VERSION} ({my_sha[:12]}); R1 module {r1_sha[:12]}; outputs {ctx.out}')
    if not args.no_ledger:
        P.ledger_append(ctx.study, 'post-hoc revision 2 checks started', module_version=MODULE_VERSION,
                        module_sha256=my_sha, revision_R1_module_sha256=r1_sha, protocol_sha256=ctx.psha,
                        holdout_features_read=ctx.hold is not None, holdout_features_sha256=ctx.fsha,
                        note='Post hoc, for the second journal revision; confirmatory results unchanged.')
    R1.reproduction_checks(ctx)
    funcs = {'N': block_N, 'T': block_T, 'P': block_P}
    failed = []
    for c in 'NTP':
        if c not in args.checks.upper():
            continue
        t0 = time.monotonic()
        ctx.rng = np.random.default_rng(ctx.cfg['seed'] + 202 + ord(c))
        try:
            funcs[c](ctx)
        except Exception as exc:
            import traceback
            failed.append(c)
            log(f'CHECK {c} FAILED: {exc}')
            log(traceback.format_exc())
        log(f'Check {c} finished in {time.monotonic() - t0:.0f} s')
    ctx.checks['failed_checks'] = failed
    man = finish(ctx, r1_sha, my_sha)
    if not args.no_ledger:
        P.ledger_append(ctx.study, 'post-hoc revision 2 checks completed', module_sha256=my_sha,
                        manifest_sha256=sha256_file(ctx.out / 'REVISION2_MANIFEST.json'), failed_checks=failed)
    log(f'Revision 2 checks complete in {man["run_minutes"]} min: {ctx.out}' + (f'; FAILED {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
