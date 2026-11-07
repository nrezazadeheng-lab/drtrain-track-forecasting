#!/usr/bin/env python3
"""Post hoc analyses for the first revision of the DR-Train track-monitoring study.

The frozen study pipeline (railtrack_study_pipeline.py, v1.2.0) is imported read-only and is never modified;
its SHA-256 must match the one recorded in FROZEN_PROTOCOL.json. The frozen protocol, the foundation files and the
existing development and holdout results are only read. With --allow-holdout-diagnostics the frozen holdout feature
file is read again, after its digest has been checked against the extraction record, and is used only to describe the
frozen models (calibration, residuals, operating conditions, inspection decisions). Nothing here is re-fitted on
holdout data and nothing changes the confirmatory results. All outputs go to <study-dir>/revision_R1/.

Blocks (reviewer comments in brackets):
  A  calibration and sharpness of the predictive distributions        [R3.6, R8.5]
  B  model selection, residual diagnostics, parameter stability        [R3.3]
  C  absolute versus relative indicator, route-wide level              [R3.2, R8.3]
  D  bin size, data-quality thresholds, positional misregistration     [R3.4, R3.10]
  E  operating conditions, season, temperature, sensor degradation     [R3.10, R8.2]
  F  regime confounding and further placebos for classification        [R3.7]
  G  sensitivity of detection limits                                   [R3.8]
  H  episode definition, leave-one-episode-out, inspection decisions   [R3.5, R8.6, R8.9]
  I  audit of the confirmatory statistics                              [R8.8]
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

MODULE_VERSION = 'R1.0'
DEFAULTS = {
    'pipeline': '/data2/nr1/V1_RailTrack/railtrack_study_pipeline.py',
    'foundation_dir': '/data2/nr1/V1_RailTrack/FOUNDATION_for_new_chat',
    'study_dir': '/data2/nr1/V1_RailTrack/STUDY_v1',
}
OUT_NAME = 'revision_R1'
ALL_BLOCKS = 'ABCDEFGHI'
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
LEAD_BUCKETS = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 120), (121, 10000)]
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
P = None  # the frozen pipeline module


# ----------------------------------------------------------------------------
# Set-up
# ----------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location('railtrack_study_pipeline', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['railtrack_study_pipeline'] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg):
    P.log(msg)


class Ctx:
    pass


def setup(args):
    global P
    t_start = time.monotonic()
    pipe = Path(args.pipeline)
    if not pipe.is_file():
        raise FileNotFoundError(f'Frozen pipeline not found: {pipe}')
    P = load_pipeline(pipe)
    P.quiet_numpy_warnings()
    ctx = Ctx()
    ctx.args = args
    ctx.t_start = t_start
    ctx.study = Path(args.study_dir)
    log(f'Revision analyses {MODULE_VERSION}; frozen pipeline {pipe}')
    ctx.protocol, ctx.psha = P.load_protocol(ctx.study)
    ctx.pipeline_sha = P.this_file_sha256()
    if ctx.pipeline_sha != ctx.protocol['pipeline_sha256']:
        msg = 'The pipeline file differs from the frozen one recorded in the protocol.'
        if not args.allow_pipeline_mismatch:
            raise RuntimeError(msg + ' Restore the frozen file (it must not be edited).')
        log('WARNING ' + msg)
    ctx.found = P.Foundation(args.foundation_dir)
    if ctx.found.sha256 != ctx.protocol['inputs_sha256']:
        raise RuntimeError('Foundation files differ from those recorded in the frozen protocol.')
    ctx.cfg = copy.deepcopy(ctx.protocol['config'])
    if P.config_hash(ctx.cfg) != ctx.protocol.get('config_sha256', P.config_hash(ctx.cfg)):
        log('WARNING the configuration read from the protocol does not reproduce its recorded digest.')
    ctx.FC = ctx.cfg['forecast']
    ctx.DT = ctx.cfg['detection']
    ctx.B = args.bootstrap or ctx.FC['bootstrap']
    ctx.cal_end = ctx.cfg['calibration_end']
    ctx.cal_ord = P.d2o(ctx.cal_end)
    ctx.prim = P.ind_key(*ctx.cfg['primary_indicator'])
    ctx.keys = [ctx.prim] + [P.ind_key(*s) for s in ctx.cfg['secondary_indicators']]
    ctx.pm = ctx.cfg['primary_model']
    ctx.rng = np.random.default_rng(ctx.cfg['seed'] + 101)
    ctx.hold = None
    ctx.fsha = None
    if args.allow_holdout_diagnostics:
        ctx.xlog, ctx.fsha = P.check_extraction_record(ctx.study, ctx.psha)
        hp = P.holdout_paths(ctx.study)
        ctx.hold = P.FeatureTable.read(hp['features'])
        if sorted(ctx.hold.passages()) != sorted(ctx.xlog.get('accepted_passage_ids', [])):
            raise RuntimeError('Passages in the holdout file differ from those accepted at extraction.')
        log(f'Holdout feature file verified and read for post hoc diagnostics ({ctx.hold.n} windows).')
    else:
        log('Holdout features not read (no --allow-holdout-diagnostics): holdout parts use saved results only.')
    ctx.out = ctx.study / OUT_NAME
    if ctx.out.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        old = ctx.study / f'{OUT_NAME}_superseded_{stamp}'
        shutil.move(str(ctx.out), str(old))
        log(f'Earlier revision outputs moved to {old}')
    (ctx.out / 'figures').mkdir(parents=True)
    (ctx.out / 'work').mkdir()
    ctx.files = []
    ctx.summary = OrderedDict()
    ctx.checks = OrderedDict()
    ctx.temp = read_temperature(args.temperature_csv) if args.temperature_csv else None
    ctx.audit = P.run_audit(ctx.found, ctx.out / 'work', ctx.cfg, {})
    ctx.dev = OrderedDict()
    ctx.hst = OrderedDict()
    return ctx


def wcsv(ctx, name, header, rows):
    P.write_csv(ctx.out / name, header, rows)
    ctx.files.append(name)
    log(f'  wrote {name} ({len(rows)} rows)')


def read_temperature(path):
    """Daily mean temperature by date from a NOAA GHCN-Daily style CSV (DATE and TAVG, or TMAX and TMIN)."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path}: no rows')
    cols = {c.upper().strip(): c for c in rows[0].keys()}
    dcol = cols.get('DATE')
    if dcol is None:
        raise ValueError(f'{path}: no DATE column')
    out = {}
    for r in rows:
        try:
            o = date.fromisoformat(str(r[dcol]).strip()[:10]).toordinal()
        except ValueError:
            continue

        def num(k):
            c = cols.get(k)
            if c is None:
                return float('nan')
            try:
                return float(r[c])
            except (TypeError, ValueError):
                return float('nan')
        t = num('TAVG')
        if not np.isfinite(t):
            t = 0.5 * (num('TMAX') + num('TMIN'))
        if np.isfinite(t):
            out[o] = t
    if not out:
        raise ValueError(f'{path}: no usable temperatures')
    return out


# ----------------------------------------------------------------------------
# Series and forecast records (with bin identity)
# ----------------------------------------------------------------------------
def add_extras(D, A, table):
    """Bin-day medians of speed and GPS residual over the quality-controlled passages of each day."""
    DM, qc, pord = A['DM'], A['qc'], A['pord']
    mats = {'speed': DM.matrix(table.x['estimated_speed_mps']),
            'gps': DM.matrix(table.x['gps_median_track_residual_m'])}
    for name, M in mats.items():
        out = np.full((len(D['ord']), DM.B), np.nan)
        for k, o in enumerate(D['ord'].tolist()):
            rows = qc & (pord == o)
            if rows.any():
                out[k] = P.nanmed(M[rows], axis=0)
        D[name] = out
    return D


def series_ext(table, model, FC, transform=None):
    series = OrderedDict()
    for d in P.DIRECTIONS:
        A = P.apply_indicator_model(model, table, d)
        if transform is not None:
            A = transform(A, d)
        D = P.day_aggregate(A)
        add_extras(D, A, table)
        D['direction'] = d
        series[d] = D
    P.attach_breaks(series, model['feature'], FC)
    return series


def cal_slices(series, last_ord, direction=None):
    return [P.series_slice(D, D['ord'] <= last_ord) for d, D in series.items() if direction is None or d == direction]


def origin_next(D, t):
    return t - 1


def origin_days(h):
    def f(D, t):
        return int(np.searchsorted(D['ord'], D['ord'][t] - h, side='right') - 1)
    return f


def origin_fixed(D, t):
    return int(np.sum(~D['is_hold'])) - 1


def collect(series, fits, FC, target_fn, origin_fn, gap=None):
    """Forecast records like the frozen collect_forecasts, keeping direction, bin, day and variance components."""
    cols = defaultdict(list)
    targets = {d: (target_fn(D) if D['E'].shape[0] else np.zeros(0, int)) for d, D in series.items()}
    tdays = [int(series[d]['ord'][t]) for d in series for t in targets[d]]
    for d, D in series.items():
        E, N, td = D['E'], D['N'], D['ord']
        tgt = targets[d]
        if E.shape[0] == 0 or len(tgt) == 0:
            continue
        states = P.compute_states(D, fits, FC)
        LV, MEAN, MOV = P.running_baselines(E, td, FC['moving_mean_days'])
        X, PP, LAST = states['local_level']
        q, rd, rp = [float(v) for v in fits['local_level']['params']]
        cor = np.asarray(D['corridor'], bool)
        for t in tgt:
            o = origin_fn(D, int(t))
            if o is None or o < 0 or o >= t:
                continue
            y = E[t]
            v = np.isfinite(y) & np.isfinite(LAST[o]) & np.isfinite(LV[o])
            if not v.any():
                continue
            n = int(v.sum())
            idx = np.where(v)[0]
            cols['direction'].append(np.full(n, d, dtype=object))
            cols['day'].append(np.full(n, int(td[t])))
            cols['origin'].append(np.full(n, int(td[o])))
            cols['bin'].append(idx)
            cols['chainage'].append(np.asarray(D['chainage'], float)[idx])
            cols['corridor'].append(cor[idx])
            cols['y'].append(y[v])
            cols['N'].append(N[t][v])
            cols['speed'].append(D['speed'][t][v] if 'speed' in D else np.full(n, np.nan))
            cols['gps'].append(D['gps'][t][v] if 'gps' in D else np.full(n, np.nan))
            cols['fingerprint'].append(np.zeros(n))
            cols['persistence'].append(LV[o][v])
            cols['historical_mean'].append(MEAN[o][v])
            cols['moving_mean'].append(MOV[o][v])
            for m in P.KALMAN_MODELS:
                mu, var = P.state_forecast(m, states[m], o, td[t], N[t], fits[m]['params'], P.extra_q(fits, m))
                cols[m].append(mu[v])
                cols['var_' + m].append(var[v])
            dt = td[t] - LAST[o][v]
            cols['vc_state'].append(PP[o][v])
            cols['vc_drift'].append(q * dt)
            cols['vc_day'].append(np.full(n, rd))
            cols['vc_passage'].append(rp / np.maximum(N[t][v], 1.0))
    if not cols:
        return {}
    R = {k: np.concatenate(v) for k, v in cols.items()}
    R['lead'] = R['day'] - R['origin']
    ep = P.calendar_episodes(tdays, FC['episode_gap_days'] if gap is None else gap)
    R['episode'] = np.array([ep[int(o)] for o in R['day'].tolist()], dtype=object)
    return R


def subset(R, mask):
    return {k: v[mask] for k, v in R.items()}


def regroup(R, FC, gap):
    """Same records with calendar episodes formed with another gap."""
    R2 = dict(R)
    ep = P.calendar_episodes(np.unique(R['day']).tolist(), gap)
    R2['episode'] = np.array([ep[int(o)] for o in R['day'].tolist()], dtype=object)
    return R2


def instance_keys(R):
    return np.array([f'{a}|{b}' for a, b in zip(R['direction'].tolist(), R['day'].tolist())], dtype=object)


def skill(R, m, base, ctx):
    am = np.abs(R['y'] - R[m])
    ab = np.abs(R['y'] - R[base])
    sk, lo, hi, C = P.cluster_bootstrap_skill(am, ab, R['episode'], ctx.B, ctx.rng)
    k, n, p = P.sign_summary(P.episode_skills(am, ab, R['episode'])[1])
    return sk, lo, hi, C, k, n, p


def rank_stats(R, m, min_bins):
    keys = instance_keys(R)
    rhos, eps = [], []
    for k in np.unique(keys):
        s = keys == k
        if s.sum() >= min_bins:
            rhos.append(P.spearman(R[m][s], R['y'][s]))
            eps.append(R['episode'][s][0])
    if not rhos:
        return float('nan'), 0, 0, float('nan')
    rhos = np.array(rhos, float)
    ek, em = P.episode_means(rhos, np.array(eps, dtype=object))
    k, n, p = P.sign_summary(em)
    return float(np.nanmean(rhos)), k, n, p


def pred_sd(R, m):
    return np.sqrt(np.maximum(R['var_' + m], 1e-12))


def amp_pct(delta, scale):
    return 100.0 * (math.exp(delta / scale) - 1.0)


# ----------------------------------------------------------------------------
# Development and holdout stores
# ----------------------------------------------------------------------------
def dev_store(ctx, key):
    if key in ctx.dev:
        return ctx.dev[key]
    c, f = P.ind_channel(key), P.ind_feature(key)
    t0 = time.monotonic()
    model = P.fit_indicator_model(ctx.found.table, c, f, ctx.cal_end, ctx.cfg)
    series = series_ext(ctx.found.table, model, ctx.FC)
    cal = cal_slices(series, ctx.cal_ord)
    fits = P.add_break_hazard(P.fit_all_kalman(cal, ctx.FC), cal, ctx.FC)
    st = {'model': model, 'series': series, 'fits': fits, 'recs': OrderedDict()}
    tgt = lambda D: np.where(D['ord'] > ctx.cal_ord)[0]
    st['recs']['next observation day'] = collect(series, fits, ctx.FC, tgt, origin_next)
    for h in (30, 90):
        st['recs'][f'at least {h} days ahead'] = collect(series, fits, ctx.FC, tgt, origin_days(h))
    first = min(int(D['ord'][0]) for D in series.values() if len(D['ord']))
    ref = lambda D: np.where((D['ord'] <= ctx.cal_ord) & (D['ord'] >= first + ctx.FC['burn_in_days']))[0]
    st['ref'] = {'next observation day': collect(series, fits, ctx.FC, ref, origin_next)}
    for h in (30, 90):
        st['ref'][f'at least {h} days ahead'] = collect(series, fits, ctx.FC, ref, origin_days(h))
    ctx.dev[key] = st
    log(f'  development store {key}: {time.monotonic() - t0:.0f} s')
    return st


def holdout_store(ctx, key):
    if ctx.hold is None:
        return None
    if key in ctx.hst:
        return ctx.hst[key]
    spec = ctx.protocol['indicators'][key]
    model, fits = spec['model'], spec['fits']
    series = OrderedDict()
    qcinfo = []
    for d in P.DIRECTIONS:
        Adev = P.apply_indicator_model(model, ctx.found.table, d)
        Ah = P.apply_indicator_model(model, ctx.hold, d)
        Ddev = add_extras(P.day_aggregate(Adev), Adev, ctx.found.table)
        Dh = add_extras(P.day_aggregate(Ah), Ah, ctx.hold)
        C = P.combine_series(Ddev, Dh)
        for k in ('speed', 'gps'):
            C[k] = np.vstack([Ddev[k], Dh[k]]) if Dh[k].shape[0] else Ddev[k].copy()
        C['direction'] = d
        series[d] = C
        for o in np.unique(Ah['pord']).tolist():
            s = Ah['pord'] == o
            qcinfo.append([key, d, P.o2d(o), int(s.sum()), int((Ah['qc'] & s).sum()),
                           float(np.nanmedian(Ah['rho'][s])) if np.isfinite(Ah['rho'][s]).any() else float('nan')])
    P.attach_breaks(series, model['feature'], ctx.FC)
    hold_t = lambda D: np.where(D['is_hold'])[0]
    st = {'model': model, 'fits': fits, 'series': series, 'qcinfo': qcinfo, 'recs': OrderedDict()}
    st['recs']['holdout sequential'] = collect(series, fits, ctx.FC, hold_t, origin_next)
    st['recs']['holdout fixed origin'] = collect(series, fits, ctx.FC, hold_t, origin_fixed)
    first = min(int(D['ord'][0]) for D in series.values() if len(D['ord']))
    ref = lambda D: np.where((~D['is_hold']) & (D['ord'] >= first + ctx.FC['burn_in_days']))[0]
    st['ref'] = {'holdout sequential': collect(series, fits, ctx.FC, ref, origin_next),
                 'holdout fixed origin': collect(series, fits, ctx.FC, ref, origin_days(30))}
    ctx.hst[key] = st
    return st


def read_saved_metrics(path, scope, key, horizon, model):
    if not Path(path).is_file():
        return None
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['scope'] == scope and r['indicator'] == key and r['horizon'] == horizon and r['model'] == model:
                return r
    return None


def reproduction_checks(ctx):
    """The re-implemented record collection must reproduce the frozen numbers exactly."""
    st = dev_store(ctx, ctx.prim)
    R = st['recs']['next observation day']
    mine = float(np.mean(np.abs(R['y'] - R[ctx.pm])))
    saved = read_saved_metrics(ctx.study / 'develop' / '03_forecast_metrics.csv', 'development', ctx.prim,
                               'next observation day', ctx.pm)
    ok = saved is not None and abs(float(saved['mae']) - mine) < 1e-5
    ctx.checks['development_mae_reproduced'] = {'recomputed': mine, 'saved': float(saved['mae']) if saved else None,
                                                'pass': ok if saved else 'saved file absent'}
    log(f'Reproduction check, development MAE: recomputed {mine:.6f}, saved {saved["mae"] if saved else "absent"}')
    hs = holdout_store(ctx, ctx.prim)
    if hs is not None:
        for scope in ('holdout sequential', 'holdout fixed origin'):
            Rh = hs['recs'][scope]
            mine = float(np.mean(np.abs(Rh['y'] - Rh[ctx.pm])))
            hz = 'next observation day' if scope == 'holdout sequential' else 'fixed origin'
            saved = read_saved_metrics(ctx.study / 'holdout_results' / 'holdout_forecast_metrics.csv', scope, ctx.prim, hz,
                                       ctx.pm)
            ok = saved is not None and abs(float(saved['mae']) - mine) < 1e-5
            ctx.checks[f'{scope} mae reproduced'] = {'recomputed': mine, 'saved': float(saved['mae']) if saved else None,
                                                     'pass': ok if saved else 'saved file absent'}
            log(f'Reproduction check, {scope} MAE: recomputed {mine:.6f}, saved {saved["mae"] if saved else "absent"}')


def scopes(ctx, key, include_hold=True):
    """(scope, horizon label, records, reference records) for development and, if read, holdout."""
    out = []
    st = dev_store(ctx, key)
    for hl, R in st['recs'].items():
        out.append(('development rolling origin', hl, R, st['ref'][hl]))
    hs = holdout_store(ctx, key) if include_hold else None
    if hs is not None:
        for sc, R in hs['recs'].items():
            out.append((sc, 'next observation day' if sc == 'holdout sequential' else 'fixed origin', R, hs['ref'][sc]))
    return out


# ----------------------------------------------------------------------------
# A: calibration and sharpness
# ----------------------------------------------------------------------------
def gauss_logscore(mu, sd, y):
    return 0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * ((y - mu) / sd) ** 2


def interval_score(mu, sd, y, alpha):
    zq = stats.norm.ppf(1 - alpha / 2)
    lo, hi = mu - zq * sd, mu + zq * sd
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def block_A(ctx):
    log('Block A: calibration and sharpness')
    rows, pit_rows, rel_rows, ep_rows, lead_rows, vc_rows = [], [], [], [], [], []
    z80, z95 = stats.norm.ppf(0.9), stats.norm.ppf(0.975)
    for key in ctx.keys:
        scale = P.quantity_scale(P.ind_feature(key))
        for scope, hl, R, Rref in scopes(ctx, key):
            if not R:
                continue
            y = R['y']
            ref_sd = {b: float(np.sqrt(np.mean((Rref['y'] - Rref[b]) ** 2))) for b in ('persistence', 'fingerprint',
                                                                                        'moving_mean')}
            crps_base = {b: P.gauss_crps(R[b], np.full(len(y), ref_sd[b]), y) for b in ref_sd}
            cand = [(m, R[m], pred_sd(R, m)) for m in P.KALMAN_MODELS]
            cand += [(b + '_gaussian', R[b], np.full(len(y), ref_sd[b])) for b in ref_sd]
            for m, mu, sd in cand:
                z = (y - mu) / sd
                crps = P.gauss_crps(mu, sd, y)
                w80, w95 = 2 * z80 * sd, 2 * z95 * sd
                rows.append([scope, key, hl, m, len(y), len(np.unique(R['episode'])), float(np.median(R['lead'])),
                             float(np.mean(np.abs(z) <= stats.norm.ppf(0.75))), float(np.mean(np.abs(z) <= z80)),
                             float(np.mean(np.abs(z) <= z95)), float(np.mean(w80)), float(np.mean(w95)),
                             amp_pct(float(np.mean(w80)) / 2, scale), float(np.mean(sd)),
                             float(np.sqrt(np.mean((y - mu) ** 2))),
                             float(np.sqrt(np.mean((y - mu) ** 2)) / np.sqrt(np.mean(sd ** 2))),
                             float(np.mean(z)), float(np.std(z)), float(np.mean(crps)),
                             1 - float(np.mean(crps)) / float(np.mean(crps_base['persistence'])),
                             1 - float(np.mean(crps)) / float(np.mean(crps_base['fingerprint'])),
                             float(np.mean(gauss_logscore(mu, sd, y))), float(np.mean(interval_score(mu, sd, y, 0.2))),
                             float(np.mean(R['N']))])
                u = stats.norm.cdf(z)
                cnt = np.histogram(u, bins=np.linspace(0, 1, 11))[0]
                for i, c in enumerate(cnt):
                    pit_rows.append([scope, key, hl, m, f'{i / 10:.1f}-{(i + 1) / 10:.1f}', int(c), float(c) / len(u)])
                for lv in LEVELS:
                    rel_rows.append([scope, key, hl, m, lv, float(np.mean(np.abs(z) <= stats.norm.ppf(0.5 + lv / 2)))])
                if m in (ctx.pm, 'persistence_gaussian'):
                    for e in np.unique(R['episode']):
                        s = R['episode'] == e
                        ep_rows.append([scope, key, hl, m, e, int(s.sum()), float(np.mean(np.abs(z[s]) <= z80)),
                                        float(np.mean(np.abs(z[s]) <= z95)), float(np.mean(w80[s])),
                                        float(np.mean(np.abs(y[s] - mu[s]))), float(np.mean(R['N'][s]))])
                    for a, b in LEAD_BUCKETS:
                        s = (R['lead'] >= a) & (R['lead'] <= b)
                        if s.sum() >= 20:
                            lead_rows.append([scope, key, hl, m, f'{a}-{b}' if b < 10000 else f'>{a - 1}', int(s.sum()),
                                              float(np.mean(np.abs(z[s]) <= z80)), float(np.mean(np.abs(z[s]) <= z95)),
                                              float(np.mean(w80[s])), float(np.mean(np.abs(y[s] - mu[s])))])
            comp = {k: R[k] for k in ('vc_state', 'vc_drift', 'vc_day', 'vc_passage')}
            tot = sum(comp.values())
            vc_rows.append([scope, key, hl, len(y)] + [float(np.mean(comp[k])) for k in comp] + [float(np.mean(tot))]
                           + [float(np.mean(comp[k]) / np.mean(tot)) for k in comp]
                           + [float(np.mean((y - R['local_level']) ** 2)), float(np.mean(R['N']))])
    head = ['scope', 'indicator', 'horizon', 'model', 'n_bin_days', 'n_episodes', 'median_lead_days', 'coverage50',
            'coverage80', 'coverage95', 'mean_width80_log', 'mean_width95_log', 'half_width80_amplitude_percent',
            'mean_predictive_sd', 'rmse', 'rmse_over_rms_predictive_sd', 'mean_z', 'sd_z', 'crps',
            'crps_skill_vs_persistence_gaussian', 'crps_skill_vs_fingerprint_gaussian', 'log_score', 'interval_score80',
            'mean_passages_per_bin_day']
    wcsv(ctx, 'A1_calibration_sharpness.csv', head, rows)
    wcsv(ctx, 'A2_pit_histogram.csv', ['scope', 'indicator', 'horizon', 'model', 'pit_bin', 'count', 'fraction'], pit_rows)
    wcsv(ctx, 'A3_reliability.csv', ['scope', 'indicator', 'horizon', 'model', 'nominal', 'empirical'], rel_rows)
    wcsv(ctx, 'A4_coverage_by_episode.csv', ['scope', 'indicator', 'horizon', 'model', 'episode', 'n_bin_days',
                                             'coverage80', 'coverage95', 'mean_width80_log', 'mae',
                                             'mean_passages_per_bin_day'], ep_rows)
    wcsv(ctx, 'A5_coverage_by_lead.csv', ['scope', 'indicator', 'horizon', 'model', 'lead_days', 'n_bin_days',
                                          'coverage80', 'coverage95', 'mean_width80_log', 'mae'], lead_rows)
    wcsv(ctx, 'A6_variance_components_local_level.csv',
         ['scope', 'indicator', 'horizon', 'n_bin_days', 'state_variance', 'drift_variance', 'day_noise_variance',
          'passage_noise_variance', 'total_predictive_variance', 'share_state', 'share_drift', 'share_day_noise',
          'share_passage_noise', 'mean_squared_error', 'mean_passages_per_bin_day'], vc_rows)
    ctx.summary['A'] = {f'{r[0]}|{r[2]}|{r[1]}|{r[3]}': {'coverage80': r[8], 'coverage95': r[9], 'width80': r[10],
                                                        'half_width80_amp_pct': r[12], 'sd_z': r[17], 'crps': r[18],
                                                        'crps_skill_vs_persistence': r[19],
                                                        'crps_skill_vs_fingerprint': r[20]}
                        for r in rows if r[1] == ctx.prim and r[3] in (ctx.pm, 'persistence_gaussian', 'fingerprint_gaussian')}
    fig_A(ctx, rel_rows, pit_rows)


# ----------------------------------------------------------------------------
# B: model selection, residual diagnostics, parameter stability
# ----------------------------------------------------------------------------
N_PARAMS = {'local_level': 3, 'local_level_regime': 3, 'mean_reverting': 4, 'local_trend': 4}


def autocorr_pairs(R, z, lags=(1, 2, 3, 4, 5)):
    """Pooled correlation of standardised errors between successive observation days of the same bin."""
    keys = np.array([f'{a}|{b}' for a, b in zip(R['direction'].tolist(), R['bin'].tolist())], dtype=object)
    order = np.lexsort((R['day'], keys))
    k_sorted, z_sorted = keys[order], z[order]
    res = {}
    lb_p = []
    for lag in lags:
        a = z_sorted[:-lag]
        b = z_sorted[lag:]
        same = k_sorted[:-lag] == k_sorted[lag:]
        ok = same & np.isfinite(a) & np.isfinite(b)
        res[lag] = (float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else float('nan'), int(ok.sum()))
    for k in np.unique(k_sorted):
        s = z_sorted[k_sorted == k]
        s = s[np.isfinite(s)]
        n = len(s)
        if n < 20:
            continue
        s = s - s.mean()
        den = float(np.dot(s, s))
        if den <= 0:
            continue
        Q = 0.0
        for lag in lags:
            r = float(np.dot(s[:-lag], s[lag:])) / den
            Q += r * r / (n - lag)
        Q *= n * (n + 2)
        lb_p.append(float(stats.chi2.sf(Q, len(lags))))
    return res, np.array(lb_p)


def spatial_pairs(R, z, lags=(1, 2, 3)):
    keys = instance_keys(R)
    order = np.lexsort((R['bin'], keys))
    k_s, b_s, z_s = keys[order], R['bin'][order], z[order]
    res = {}
    for lag in lags:
        out_a, out_b = [], []
        for sh in range(1, lag + 1):
            a, b = z_s[:-sh], z_s[sh:]
            ok = (k_s[:-sh] == k_s[sh:]) & ((b_s[sh:] - b_s[:-sh]) == lag)
            out_a.append(a[ok])
            out_b.append(b[ok])
        a, b = np.concatenate(out_a), np.concatenate(out_b)
        res[lag] = (float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else float('nan'), int(len(a)))
    return res


def block_B(ctx):
    log('Block B: model selection, residual diagnostics and parameter stability')
    sel_rows, oos_rows, res_rows, qq_rows, stab_rows = [], [], [], [], []
    for key in ctx.keys:
        st = dev_store(ctx, key)
        fit_sets = [('calibration (to ' + ctx.cal_end + ')', st['fits'])]
        fit_sets.append(('all development (frozen protocol)', ctx.protocol['indicators'][key]['fits']))
        for name, fits in fit_sets:
            vals = []
            for m in P.KALMAN_MODELS:
                ll, n = float(fits[m]['loglik']), int(fits[m]['n_obs'])
                k = N_PARAMS[m]
                vals.append([name, key, m, k, ll, n, 2 * k - 2 * ll, k * math.log(n) - 2 * ll,
                             bool(fits[m].get('converged', True))])
            amin = min(v[6] for v in vals)
            bmin = min(v[7] for v in vals)
            for v in vals:
                sel_rows.append(v + [v[6] - amin, v[7] - bmin])
        for scope, hl, R, Rref in scopes(ctx, key):
            if not R:
                continue
            y = R['y']
            for m in ['fingerprint', 'persistence', 'historical_mean', 'moving_mean'] + list(P.KALMAN_MODELS):
                mae = float(np.mean(np.abs(y - R[m])))
                crps = ls = cov = float('nan')
                if m in P.KALMAN_MODELS:
                    sd = pred_sd(R, m)
                    crps = float(np.mean(P.gauss_crps(R[m], sd, y)))
                    ls = float(np.mean(gauss_logscore(R[m], sd, y)))
                    cov = float(np.mean(np.abs(y - R[m]) / sd <= stats.norm.ppf(0.9)))
                oos_rows.append([scope, key, hl, m, len(y), mae, crps, ls, cov])
            if hl not in ('next observation day',):
                continue
            for m in P.KALMAN_MODELS:
                z = (y - R[m]) / pred_sd(R, m)
                ac, lb = autocorr_pairs(R, z)
                sp = spatial_pairs(R, z)
                res_rows.append([scope, key, m, len(z), float(np.mean(z)), float(np.std(z)), float(stats.skew(z)),
                                 float(stats.kurtosis(z)), float(np.mean(np.abs(z) > 3)), 2 * stats.norm.sf(3)]
                                + [ac[l][0] for l in (1, 2, 3, 4, 5)] + [ac[1][1]]
                                + [sp[l][0] for l in (1, 2, 3)] + [sp[1][1]]
                                + [len(lb), float(np.mean(lb < 0.05)) if len(lb) else float('nan'),
                                   float(np.median(lb)) if len(lb) else float('nan')])
                if m == ctx.pm:
                    for pq in np.round(np.arange(0.005, 1.0, 0.01), 3):
                        qq_rows.append([scope, key, m, float(pq), float(stats.norm.ppf(pq)), float(np.quantile(z, pq))])
                    if key == ctx.prim:
                        ctx.summary.setdefault('B_residual_correlation', {})[scope] = {
                            'temporal_lag1': ac[1][0], 'spatial_lag1': sp[1][0], 'sd_z': float(np.std(z)),
                            'ljung_box_share_p_below_0_05': float(np.mean(lb < 0.05)) if len(lb) else None}
    # Parameter stability: indicator and filters refitted with data up to each date, and per direction.
    sets = [('calibration', ctx.cal_end, None)] + [('cut ' + c, c, None) for c in ctx.cfg['pseudo_prospective']['cuts']]
    sets += [('calibration, inbound only', ctx.cal_end, 'inbound'), ('calibration, outbound only', ctx.cal_end, 'outbound')]
    for key in ctx.keys:
        c, f = P.ind_channel(key), P.ind_feature(key)
        F_ref = None
        models = list(P.KALMAN_MODELS) if key == ctx.prim else [ctx.pm]
        for name, last, direc in sets:
            t0 = time.monotonic()
            if name == 'calibration':
                st = dev_store(ctx, key)
                model, series, fits = st['model'], st['series'], st['fits']
            else:
                model = P.fit_indicator_model(ctx.found.table, c, f, last, ctx.cfg) if direc is None else dev_store(ctx, key)['model']
                series = series_ext(ctx.found.table, model, ctx.FC) if direc is None else dev_store(ctx, key)['series']
                sl = cal_slices(series, P.d2o(last), direc)
                fits = OrderedDict((m, P.fit_kalman(sl, P.KALMAN_KIND[m], ctx.FC)) for m in models)
            Fs = {d: np.array([np.nan if v is None else v for v in model['directions'][d]['F']], float) for d in P.DIRECTIONS}
            if F_ref is None:
                F_ref = Fs
            frho = float(np.mean([P.spearman(Fs[d][np.isfinite(Fs[d]) & np.isfinite(F_ref[d])],
                                             F_ref[d][np.isfinite(Fs[d]) & np.isfinite(F_ref[d])]) for d in P.DIRECTIONS]))
            for m in models:
                p = list(fits[m]['params'])
                q = qs = tau = s2 = float('nan')
                if m == 'local_trend':
                    q, qs, rd, rp = p
                    ch90 = math.sqrt(90 * q)
                elif m == 'mean_reverting':
                    tau, s2, rd, rp = p
                    ch90 = math.sqrt(2.0 * s2 * (1.0 - math.exp(-90.0 / tau)))
                else:
                    q, rd, rp = p
                    ch90 = math.sqrt(90 * q)
                noise = math.sqrt(rd + rp)
                stab_rows.append([name, last, direc or 'both', key, m, int(fits[m]['n_obs']), float(fits[m]['loglik']),
                                  q, qs, tau, s2, rd, rp, ch90, noise, ch90 / noise if noise > 0 else float('nan'),
                                  model['beta_speed'], frho])
            log(f'  stability {key} {name}: {time.monotonic() - t0:.0f} s')
    wcsv(ctx, 'B1_model_selection_likelihood.csv',
         ['fit_set', 'indicator', 'model', 'n_parameters', 'loglik', 'n_obs', 'aic', 'bic', 'converged', 'delta_aic',
          'delta_bic'], sel_rows)
    wcsv(ctx, 'B2_out_of_sample_scores.csv', ['scope', 'indicator', 'horizon', 'model', 'n_bin_days', 'mae', 'crps',
                                              'log_score', 'coverage80'], oos_rows)
    wcsv(ctx, 'B3_residual_diagnostics.csv',
         ['scope', 'indicator', 'model', 'n', 'mean_z', 'sd_z', 'skewness', 'excess_kurtosis', 'share_abs_z_above_3',
          'gaussian_share_abs_z_above_3', 'temporal_acf_lag1', 'temporal_acf_lag2', 'temporal_acf_lag3',
          'temporal_acf_lag4', 'temporal_acf_lag5', 'temporal_pairs_lag1', 'spatial_acf_lag1_bin', 'spatial_acf_lag2_bins',
          'spatial_acf_lag3_bins', 'spatial_pairs_lag1', 'bins_ljung_box', 'share_bins_ljung_box_p_below_0_05',
          'median_ljung_box_p'], res_rows)
    wcsv(ctx, 'B4_qq_standardised_errors.csv', ['scope', 'indicator', 'model', 'probability', 'gaussian_quantile',
                                                'empirical_quantile'], qq_rows)
    wcsv(ctx, 'B5_parameter_stability.csv',
         ['fit_set', 'data_to', 'directions', 'indicator', 'model', 'n_obs', 'loglik', 'level_drift_variance_per_day',
          'slope_variance_per_day', 'reversion_time_days', 'stationary_level_variance', 'day_noise_variance',
          'passage_noise_variance', 'level_change_sd_90_days', 'single_passage_noise_sd', 'change_to_noise_ratio_90_days',
          'speed_slope', 'fingerprint_spearman_vs_calibration'], stab_rows)
    ctx.summary['B_selection'] = {f'{r[0]}|{r[1]}|{r[2]}': {'delta_aic': r[9], 'delta_bic': r[10]} for r in sel_rows
                                  if r[1] == ctx.prim}
    ctx.summary['B_stability_primary_local_level'] = [
        {'fit_set': r[0], 'drift_sd_90': r[13], 'noise_sd': r[14], 'ratio': r[15], 'speed_slope': r[16],
         'fingerprint_rho': r[17]} for r in stab_rows if r[3] == ctx.prim and r[4] == ctx.pm]
    fig_B(ctx, res_rows)


# ----------------------------------------------------------------------------
# C: absolute versus relative indicator
# ----------------------------------------------------------------------------
def regime_ids(ctx, channels, ords):
    segs = [ctx.audit['segments'].get(c) or [] for c in channels]
    return np.array(['|'.join(str(P.segment_of(s, o)) if s else '0' for s in segs) for o in np.asarray(ords).tolist()],
                    dtype=object)


def channels_of(key):
    c = P.ind_channel(key)
    return (4, 5) if c == 45 else (c,)


def boundary_mask(ctx, key, R):
    m = np.zeros(len(R['day']), bool)
    for ch in channels_of(key):
        for _, first_after, _ in ctx.audit['boundaries'].get(ch, []):
            m |= (R['origin'] < first_after) & (first_after <= R['day'])
    return m


def block_C(ctx):
    log('Block C: absolute versus relative indicator')
    rows, lvl_rows = [], []
    for key in ctx.keys:
        st = dev_store(ctx, key)
        model = st['model']

        def absolute(A, d):
            ref = P.nanmed(A['offset'][A['qc'] & (A['pord'] <= ctx.cal_ord)])
            A2 = dict(A)
            A2['E'] = A['E'] + (A['offset'] - ref)[:, None]
            return A2
        aser = series_ext(ctx.found.table, model, ctx.FC, transform=absolute)
        acal = cal_slices(aser, ctx.cal_ord)
        afits = P.add_break_hazard(P.fit_all_kalman(acal, ctx.FC), acal, ctx.FC)
        tgt = lambda D: np.where(D['ord'] > ctx.cal_ord)[0]
        for hl, ofn in (('next observation day', origin_next), ('at least 30 days ahead', origin_days(30)),
                        ('at least 90 days ahead', origin_days(90))):
            Rabs = collect(aser, afits, ctx.FC, tgt, ofn)
            Rrel = st['recs'][hl]
            for variant, R in (('relative (frozen)', Rrel), ('absolute (route-wide level kept)', Rabs)):
                bm = boundary_mask(ctx, key, R)
                for subset_name, mask in (('all evaluation bin-days', np.ones(len(R['y']), bool)),
                                          ('no instrument-regime change between origin and target', ~bm)):
                    S = subset(R, mask)
                    if len(S['y']) < 50:
                        continue
                    for m in (ctx.pm, 'moving_mean'):
                        sp = skill(S, m, 'persistence', ctx)
                        sf = skill(S, m, 'fingerprint', ctx)
                        sd = pred_sd(S, ctx.pm) if m == ctx.pm else None
                        rows.append([key, variant, hl, subset_name, m, len(S['y']), len(np.unique(S['episode'])),
                                     float(np.std(S['y'])), float(np.mean(np.abs(S['y'] - S[m]))),
                                     float(np.mean(np.abs(S['y'] - S['persistence']))),
                                     float(np.mean(np.abs(S['y']))),
                                     sp[0], sp[1], sp[2], f'{sp[4]} of {sp[5]}', sf[0], sf[1], sf[2], f'{sf[4]} of {sf[5]}',
                                     float(np.mean(np.abs(S['y'] - S[m]) / sd <= stats.norm.ppf(0.9))) if sd is not None
                                     else float('nan')])
        # Route-wide daily level (passage offsets): how much is instrument regime, how much trend within regimes.
        days, offs = [], []
        pooled = defaultdict(list)
        for D in st['series'].values():
            for o, v in zip(D['ord'].tolist(), D['offset'].tolist()):
                if np.isfinite(v):
                    pooled[o].append(v)
        days = np.array(sorted(pooled), int)
        offs = np.array([float(np.median(pooled[o])) for o in days])
        reg = regime_ids(ctx, channels_of(key), days)
        ureg = np.unique(reg)
        means = {r: float(offs[reg == r].mean()) for r in ureg}
        fitted = np.array([means[r] for r in reg])
        sst = float(np.sum((offs - offs.mean()) ** 2))
        r2 = 1 - float(np.sum((offs - fitted) ** 2)) / sst if sst > 0 else float('nan')
        tyr = (days - days.min()) / 365.25
        tw = tyr - np.array([tyr[reg == r].mean() for r in reg])
        yw = offs - fitted
        slope = float(np.dot(tw, yw) / np.dot(tw, tw)) if np.dot(tw, tw) > 0 else float('nan')
        ep = P.calendar_episodes(days.tolist(), ctx.FC['episode_gap_days'])
        epi = np.array([ep[int(o)] for o in days])
        ue = np.unique(epi)
        boot = []
        for _ in range(ctx.B):
            pick = ctx.rng.choice(ue, len(ue), replace=True)
            idx = np.concatenate([np.where(epi == e)[0] for e in pick])
            a, b = tw[idx], yw[idx]
            if np.dot(a, a) > 0:
                boot.append(float(np.dot(a, b) / np.dot(a, a)))
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float('nan'), float('nan')))
        scale = P.quantity_scale(P.ind_feature(key))
        within_sd = float(np.std(yw))
        between_sd = float(np.std([means[r] for r in ureg])) if len(ureg) > 1 else 0.0
        lvl_rows.append([key, len(days), len(ureg), float(np.std(offs)), r2, between_sd, within_sd, slope, float(lo),
                         float(hi), amp_pct(slope, scale), amp_pct(float(lo), scale), amp_pct(float(hi), scale), len(ue)])
    wcsv(ctx, 'C1_absolute_vs_relative_forecasting.csv',
         ['indicator', 'indicator_variant', 'horizon', 'subset', 'model', 'n_bin_days', 'n_episodes', 'sd_of_target', 'mae',
          'mae_persistence', 'mae_fingerprint', 'skill_vs_persistence', 'skill_vs_persistence_lo95',
          'skill_vs_persistence_hi95', 'episodes_better_than_persistence', 'skill_vs_fingerprint',
          'skill_vs_fingerprint_lo95', 'skill_vs_fingerprint_hi95', 'episodes_better_than_fingerprint', 'coverage80'], rows)
    wcsv(ctx, 'C2_route_wide_level_decomposition.csv',
         ['indicator', 'n_days', 'n_instrument_regimes', 'sd_daily_route_level', 'r2_explained_by_regimes',
          'sd_between_regime_means', 'sd_within_regimes', 'within_regime_trend_per_year_log',
          'trend_lo95', 'trend_hi95', 'trend_per_year_amplitude_percent', 'trend_lo95_amplitude_percent',
          'trend_hi95_amplitude_percent', 'n_episodes'], lvl_rows)
    ctx.summary['C_route_level'] = {r[0]: {'r2_regimes': r[4], 'between_sd': r[5], 'within_sd': r[6],
                                           'trend_amp_pct_per_year': [r[10], r[11], r[12]]} for r in lvl_rows}
    ctx.summary['C_forecast_primary'] = [{'variant': r[1], 'horizon': r[2], 'subset': r[3], 'skill_vs_persistence': r[11],
                                          'skill_vs_fingerprint': r[15], 'episodes_fp': r[18]}
                                         for r in rows if r[0] == ctx.prim and r[4] == ctx.pm]


# ----------------------------------------------------------------------------
# D: bin size, data quality and positional misregistration
# ----------------------------------------------------------------------------
def derived_table(table, s_updates=None, x_updates=None):
    s = dict(table.s)
    x = {k: v for k, v in table.x.items() if not k.startswith('s45_')}
    if s_updates:
        s.update(s_updates)
    if x_updates:
        x.update(x_updates)
    return P.FeatureTable(s, x, list(s) + list(x), 'revision-derived')


def coarse_table(table, size, power_cols):
    """Aggregate complete runs of speed-screened 100 m windows into coarser bins (duration-weighted power)."""
    need = int(round(size / 100.0))
    idx = np.where(table.screened)[0]
    d = table.s['direction'][idx]
    pid = table.s['passage_id'][idx]
    start = (np.floor(table.x['chainage_start_m'][idx] / size) * size).astype(int)
    keys = np.array([f'{a}|{b}|{c}' for a, b, c in zip(pid.tolist(), d.tolist(), start.tolist())], dtype=object)
    uk, first, inv, cnt = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    dur = table.x['estimated_duration_s'][idx]
    w = np.where(np.isfinite(dur) & (dur > 0), dur, np.nan)
    wsum = np.bincount(inv, weights=np.nan_to_num(w), minlength=len(uk))
    nw = np.bincount(inv, weights=np.isfinite(w).astype(float), minlength=len(uk))
    keep = (cnt == need) & (nw == need) & (wsum > 0)
    x = {}
    for col in power_cols:
        v = table.x[col][idx]
        good = np.bincount(inv, weights=(np.isfinite(v) & (v > 0)).astype(float), minlength=len(uk))
        x[col] = np.where(good == need, np.bincount(inv, weights=np.nan_to_num(v) * np.nan_to_num(w), minlength=len(uk))
                          / np.where(wsum > 0, wsum, 1), np.nan)
    x['estimated_duration_s'] = wsum
    x['estimated_speed_mps'] = 100.0 * cnt / np.where(wsum > 0, wsum, np.nan)
    for g in ('gps_median_track_residual_m', 'gps_p95_track_residual_m'):
        m = np.full(len(uk), -np.inf)
        np.maximum.at(m, inv, np.nan_to_num(table.x[g][idx], nan=np.inf))
        x[g] = m
    cor = np.bincount(inv, weights=(table.s['corridor_candidate_bin'][idx] == 'yes').astype(float), minlength=len(uk)) / cnt
    rows0 = idx[first]
    x['chainage_start_m'] = start[first].astype(float)
    x['chainage_end_m'] = x['chainage_start_m'] + size
    s = {c: table.s[c][rows0] for c in P.STRING_COLUMNS}
    s['track_bin_id'] = np.array([f'{a}_{size}m_{b}' for a, b in zip(d[first].tolist(), start[first].tolist())], dtype=object)
    s['corridor_candidate_bin'] = np.where(cor >= 0.5, 'yes', 'no').astype(object)
    s['analysis_screening_flag'] = np.full(len(uk), 'speed_screen_pass', dtype=object)
    s = {k: v[keep] for k, v in s.items()}
    x = {k: v[keep] for k, v in x.items()}
    return P.FeatureTable(s, x, list(s) + list(x), f'revision-derived {size} m bins')


def jitter_table(table, max_shift, cols, rng):
    """Emulate a passage-wide positional error u (uniform within +/- max_shift metres): each 100 m value becomes the
    length-weighted mix of its own bin and the neighbouring bin in the direction of the shift."""
    pid = table.s['passage_id']
    d = table.s['direction']
    ch = table.x['chainage_start_m']
    lookup = {(a, b, float(c)): i for i, (a, b, c) in enumerate(zip(pid.tolist(), d.tolist(), ch.tolist()))}
    shifts = {p: float(rng.uniform(-max_shift, max_shift)) for p in sorted(set(pid.tolist()))}
    upd = {c: table.x[c].copy() for c in cols}
    for i, (a, b, c) in enumerate(zip(pid.tolist(), d.tolist(), ch.tolist())):
        u = shifts[a]
        j = lookup.get((a, b, float(c) + (100.0 if u > 0 else -100.0)))
        if j is None or u == 0:
            continue
        f = abs(u) / 100.0
        for col in cols:
            v0, v1 = table.x[col][i], table.x[col][j]
            if np.isfinite(v0) and np.isfinite(v1):
                upd[col][i] = (1 - f) * v0 + f * v1
    return derived_table(table, x_updates=upd)


def run_variant(ctx, name, table, cfg, base_fits):
    key = ctx.prim
    c, f = P.ind_channel(key), P.ind_feature(key)
    model = P.fit_indicator_model(table, c, f, ctx.cal_end, cfg)
    series = series_ext(table, model, ctx.FC)
    cal = cal_slices(series, ctx.cal_ord)
    fits = copy.deepcopy(base_fits)
    fits[ctx.pm] = P.fit_kalman(cal, P.KALMAN_KIND[ctx.pm], ctx.FC)
    tgt = lambda D: np.where(D['ord'] > ctx.cal_ord)[0]
    out = []
    npass = sum(model['directions'][d]['n_fit_passages_qc'] for d in P.DIRECTIONS)
    nbins = sum(len(model['directions'][d]['bins']['ids']) for d in P.DIRECTIONS)
    for hl, ofn in (('next observation day', origin_next), ('at least 30 days ahead', origin_days(30))):
        R = collect(series, fits, ctx.FC, tgt, ofn)
        if not R:
            continue
        sp = skill(R, ctx.pm, 'persistence', ctx)
        sf = skill(R, ctx.pm, 'fingerprint', ctx)
        sd = pred_sd(R, ctx.pm)
        z = np.abs(R['y'] - R[ctx.pm]) / sd
        rho, rk, rn, _ = rank_stats(R, ctx.pm, ctx.FC['min_bins_ranking'])
        out.append([name, hl, nbins, npass, len(R['y']), len(np.unique(R['episode'])), float(np.std(R['y'])),
                    float(np.mean(np.abs(R['y'] - R[ctx.pm]))), sp[0], sp[1], sp[2], f'{sp[4]} of {sp[5]}',
                    sf[0], sf[1], sf[2], f'{sf[4]} of {sf[5]}', float(np.mean(z <= stats.norm.ppf(0.9))),
                    float(np.mean(z <= stats.norm.ppf(0.975))), float(np.mean(2 * stats.norm.ppf(0.9) * sd)), rho,
                    fits[ctx.pm]['params'][0], math.sqrt(fits[ctx.pm]['params'][1] + fits[ctx.pm]['params'][2])])
    return out


def block_D(ctx):
    log('Block D: bin size, data-quality thresholds and positional misregistration')
    st = dev_store(ctx, ctx.prim)
    base_fits = st['fits']
    t = ctx.found.table
    flag = t.s['analysis_screening_flag']
    spd = t.x['estimated_speed_mps']
    g50, g95 = t.x['gps_median_track_residual_m'], t.x['gps_p95_track_residual_m']
    pcols = [f's4_{P.ind_feature(ctx.prim)}', f's5_{P.ind_feature(ctx.prim)}'] if P.ind_channel(ctx.prim) == 45 else \
        [f's{P.ind_channel(ctx.prim)}_{P.ind_feature(ctx.prim)}']

    def flagged(mask):
        return {'analysis_screening_flag': np.where(mask, 'speed_screen_pass', 'revision_excluded').astype(object)}
    scr = flag == 'speed_screen_pass'
    variants = [
        ('frozen settings (100 m bins)', t, ctx.cfg),
        ('200 m bins', coarse_table(t, 200, pcols), ctx.cfg),
        ('300 m bins', coarse_table(t, 300, pcols), ctx.cfg),
        ('GPS residual median <= 10 m, p95 <= 20 m', derived_table(t, flagged(scr & (g50 <= 10) & (g95 <= 20))), ctx.cfg),
        ('GPS residual median <= 5 m, p95 <= 10 m', derived_table(t, flagged(scr & (g50 <= 5) & (g95 <= 10))), ctx.cfg),
        ('minimum speed 3 m/s', derived_table(t, flagged(scr | ((flag == 'review_low_speed') & (spd >= 3)))), ctx.cfg),
        ('minimum speed 8 m/s', derived_table(t, flagged(scr & (spd >= 8))), ctx.cfg),
    ]
    for label, path, val in (('profile correlation >= 0.20', ('qc', 'min_profile_rho'), 0.20),
                             ('profile correlation >= 0.40', ('qc', 'min_profile_rho'), 0.40),
                             ('at least 15 bins per passage', ('qc', 'min_bins_per_passage'), 15)):
        cv = copy.deepcopy(ctx.cfg)
        cv[path[0]][path[1]] = val
        variants.append((label, t, cv))
    jr = np.random.default_rng(ctx.cfg['seed'] + 202)
    for J in (25, 50):
        variants.append((f'positional error up to {J} m', jitter_table(t, J, pcols, jr), ctx.cfg))
    rows = []
    for name, table, cfg in variants:
        t0 = time.monotonic()
        rows += run_variant(ctx, name, table, cfg, base_fits)
        log(f'  variant {name}: {time.monotonic() - t0:.0f} s')
    wcsv(ctx, 'D1_sensitivity_bins_quality_position.csv',
         ['variant', 'horizon', 'n_bins_both_directions', 'n_calibration_passages_qc', 'n_bin_days', 'n_episodes',
          'sd_of_target', 'mae', 'skill_vs_persistence', 'skill_vs_persistence_lo95', 'skill_vs_persistence_hi95',
          'episodes_better_than_persistence', 'skill_vs_fingerprint', 'skill_vs_fingerprint_lo95',
          'skill_vs_fingerprint_hi95', 'episodes_better_than_fingerprint', 'coverage80', 'coverage95', 'mean_width80_log',
          'mean_spearman', 'level_drift_variance_per_day', 'single_passage_noise_sd'], rows)
    ctx.summary['D'] = [{'variant': r[0], 'horizon': r[1], 'skill_vs_persistence': [r[8], r[9], r[10]],
                         'skill_vs_fingerprint': [r[12], r[13], r[14]], 'coverage80': r[16], 'n_bin_days': r[4]}
                        for r in rows]
    fig_D(ctx, rows)


# ----------------------------------------------------------------------------
# E: operating conditions, season, temperature, sensor degradation
# ----------------------------------------------------------------------------
def season(o):
    m = date.fromordinal(int(o)).month
    return {12: 'winter (Dec-Feb)', 1: 'winter (Dec-Feb)', 2: 'winter (Dec-Feb)', 3: 'spring (Mar-May)',
            4: 'spring (Mar-May)', 5: 'spring (Mar-May)', 6: 'summer (Jun-Aug)', 7: 'summer (Jun-Aug)',
            8: 'summer (Jun-Aug)'}.get(m, 'autumn (Sep-Nov)')


def tercile_labels(v, edges, name, unit):
    lab = np.full(len(v), 'missing', dtype=object)
    lab[np.isfinite(v) & (v <= edges[0])] = f'{name} low (<= {edges[0]:.1f} {unit})'
    lab[np.isfinite(v) & (v > edges[0]) & (v <= edges[1])] = f'{name} middle'
    lab[np.isfinite(v) & (v > edges[1])] = f'{name} high (> {edges[1]:.1f} {unit})'
    return lab


def strat_rows(ctx, scope, key, R, factor, labels):
    rows = []
    for lab in [l for l in np.unique(labels) if l != 'missing']:
        S = subset(R, labels == lab)
        if len(S['y']) < 30:
            continue
        sp = skill(S, ctx.pm, 'persistence', ctx)
        sf = skill(S, ctx.pm, 'fingerprint', ctx)
        z = np.abs(S['y'] - S[ctx.pm]) / pred_sd(S, ctx.pm)
        rows.append([scope, key, factor, lab, len(S['y']), len(np.unique(S['episode'])),
                     float(np.mean(np.abs(S['y'] - S[ctx.pm]))), float(np.mean(np.abs(S['y'] - S['persistence']))),
                     float(np.mean(np.abs(S['y']))), sp[0], sp[1], sp[2], f'{sp[4]} of {sp[5]}', sf[0], sf[1], sf[2],
                     f'{sf[4]} of {sf[5]}', float(np.mean(z <= stats.norm.ppf(0.9)))])
    return rows


def block_E(ctx):
    log('Block E: operating conditions, season, temperature and sensor degradation')
    rows, trows, srows = [], [], []
    key = ctx.prim
    st = dev_store(ctx, key)
    # Speed variability of each bin across calibration passages (stations, braking and acceleration zones).
    var_bin = {}
    for d in P.DIRECTIONS:
        DM = P.DirMatrix(ctx.found.table, d)
        V = DM.logmatrix('estimated_speed_mps')
        sd = P.nanstd(V[DM.pord <= ctx.cal_ord], axis=0)
        for j, b in enumerate(DM.bins):
            var_bin[(d, b)] = float(sd[j])
    Rdev = st['recs']['next observation day']
    sp_edges = np.nanpercentile(Rdev['speed'], [100 / 3, 200 / 3])
    gps_edges = np.nanpercentile(Rdev['gps'], [100 / 3, 200 / 3])
    all_sv = np.array([var_bin.get((a, st['series'][a]['bins'][b]), np.nan)
                       for a, b in zip(Rdev['direction'].tolist(), Rdev['bin'].tolist())])
    sv_edges = np.nanpercentile(all_sv, [100 / 3, 200 / 3])
    T_edges = dT_edges = None
    if ctx.temp is not None:
        tv = np.array([ctx.temp.get(int(o), np.nan) for o in Rdev['day']])
        dtv = np.abs(tv - np.array([ctx.temp.get(int(o), np.nan) for o in Rdev['origin']]))
        T_edges = np.nanpercentile(tv, [100 / 3, 200 / 3])
        dT_edges = np.nanpercentile(dtv, [100 / 3, 200 / 3])
    sets = [('development rolling origin, next day', Rdev, st['series'])]
    hs = holdout_store(ctx, key)
    if hs is not None:
        sets.append(('holdout sequential', hs['recs']['holdout sequential'], hs['series']))
    for scope, R, series in sets:
        sv = np.array([var_bin.get((a, series[a]['bins'][b]), np.nan) for a, b in zip(R['direction'].tolist(),
                                                                                       R['bin'].tolist())])
        facs = [('bin-day speed', tercile_labels(R['speed'], sp_edges, 'speed', 'm/s')),
                ('speed variability of bin (braking and acceleration zones)',
                 tercile_labels(sv, sv_edges, 'log-speed SD', '')),
                ('GPS residual of bin-day', tercile_labels(R['gps'], gps_edges, 'GPS residual', 'm')),
                ('corridor', np.where(R['corridor'], 'corridor', 'outside corridor').astype(object)),
                ('direction', R['direction'].astype(object)),
                ('season of target day', np.array([season(o) for o in R['day']], dtype=object))]
        if ctx.temp is not None:
            tv = np.array([ctx.temp.get(int(o), np.nan) for o in R['day']])
            dtv = np.abs(tv - np.array([ctx.temp.get(int(o), np.nan) for o in R['origin']]))
            facs.append(('temperature of target day', tercile_labels(tv, T_edges, 'temperature', 'deg')))
            facs.append(('temperature change since origin', tercile_labels(dtv, dT_edges, '|change|', 'deg')))
        for fac, lab in facs:
            rows += strat_rows(ctx, scope, key, R, fac, lab)
        if ctx.temp is not None:
            days = np.unique(R['day'])
            err = np.array([float(np.mean((R['y'] - R[ctx.pm])[R['day'] == o])) for o in days])
            aerr = np.array([float(np.mean(np.abs(R['y'] - R[ctx.pm])[R['day'] == o])) for o in days])
            org = np.array([int(np.median(R['origin'][R['day'] == o])) for o in days])
            tt = np.array([ctx.temp.get(int(o), np.nan) for o in days])
            dt = tt - np.array([ctx.temp.get(int(o), np.nan) for o in org])
            for lab, a, b in (('daily mean forecast error vs temperature change since origin', err, dt),
                              ('daily mean absolute error vs absolute temperature change', aerr, np.abs(dt)),
                              ('daily mean absolute error vs temperature', aerr, tt)):
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 8:
                    trows.append([scope, key, lab, int(ok.sum()), float(stats.spearmanr(a[ok], b[ok])[0]),
                                  float(stats.spearmanr(a[ok], b[ok])[1]), float(np.corrcoef(a[ok], b[ok])[0, 1])])
    if ctx.temp is not None:
        pooled = defaultdict(list)
        for D in st['series'].values():
            for o, v in zip(D['ord'].tolist(), D['offset'].tolist()):
                if np.isfinite(v):
                    pooled[o].append(v)
        days = np.array(sorted(pooled), int)
        offs = np.array([float(np.median(pooled[o])) for o in days])
        reg = regime_ids(ctx, channels_of(key), days)
        within = offs - np.array([offs[reg == r].mean() for r in reg])
        tt = np.array([ctx.temp.get(int(o), np.nan) for o in days])
        for lab, a in (('route-wide passage level vs temperature', offs),
                       ('route-wide passage level within instrument regimes vs temperature', within)):
            ok = np.isfinite(a) & np.isfinite(tt)
            if ok.sum() >= 8:
                trows.append(['development, all days', key, lab, int(ok.sum()), float(stats.spearmanr(a[ok], tt[ok])[0]),
                              float(stats.spearmanr(a[ok], tt[ok])[1]), float(np.corrcoef(a[ok], tt[ok])[0, 1])])
    wcsv(ctx, 'E1_errors_by_operating_condition.csv',
         ['scope', 'indicator', 'factor', 'level', 'n_bin_days', 'n_episodes', 'mae', 'mae_persistence', 'mae_fingerprint',
          'skill_vs_persistence', 'skill_vs_persistence_lo95', 'skill_vs_persistence_hi95',
          'episodes_better_than_persistence', 'skill_vs_fingerprint', 'skill_vs_fingerprint_lo95',
          'skill_vs_fingerprint_hi95', 'episodes_better_than_fingerprint', 'coverage80'], rows)
    if trows:
        wcsv(ctx, 'E2_temperature_associations.csv',
             ['scope', 'indicator', 'association', 'n_days', 'spearman', 'spearman_p', 'pearson'], trows)
    # Sensor degradation: quality control and monthly errors of every indicator in the holdout.
    if ctx.hold is not None:
        q_rows = []
        for k in ctx.keys:
            h = holdout_store(ctx, k)
            q_rows += h['qcinfo']
            R = h['recs']['holdout sequential']
            months = np.array([date.fromordinal(int(o)).strftime('%Y-%m') for o in R['day']], dtype=object)
            for mo in np.unique(months):
                S = subset(R, months == mo)
                ae = np.abs(S['y'] - S[ctx.pm])
                srows.append([k, mo, len(S['y']), float(ae.mean()),
                              1 - float(ae.sum()) / float(np.abs(S['y'] - S['persistence']).sum()),
                              1 - float(ae.sum()) / float(np.abs(S['y']).sum()),
                              float(np.mean(ae / pred_sd(S, ctx.pm) <= stats.norm.ppf(0.9)))])
        wcsv(ctx, 'E3_holdout_quality_control_by_day.csv',
             ['indicator', 'direction', 'date', 'passages', 'passages_passing_qc', 'median_profile_correlation'], q_rows)
        wcsv(ctx, 'E4_holdout_errors_by_month.csv',
             ['indicator', 'month', 'n_bin_days', 'mae', 'skill_vs_persistence', 'skill_vs_fingerprint', 'coverage80'],
             srows)
    ctx.summary['E'] = [{'scope': r[0], 'factor': r[2], 'level': r[3], 'n': r[4], 'skill_vs_persistence': r[9],
                         'skill_vs_fingerprint': r[13], 'episodes_fp': r[16], 'coverage80': r[17]} for r in rows]


# ----------------------------------------------------------------------------
# F: classification, regime confounding and further placebos
# ----------------------------------------------------------------------------
def cv_mean(X, y, groups, scheme, clf, C, rng, reps):
    accs = P.cv_scores(X, y, groups, scheme, clf, C, rng, reps)
    return (float(np.mean(accs)) if accs else float('nan')), len(accs)


def class_labels(pord, classes, shift_days=0):
    y = np.full(len(pord), -1)
    for k, (name, a, b) in enumerate(classes):
        y[(pord >= P.d2o(a) + shift_days) & (pord < P.d2o(b) + shift_days)] = k
    return y


def block_F(ctx):
    log('Block F: classification, regime confounding and further placebos')
    C = copy.deepcopy(ctx.cfg['classification'])
    reps = ctx.args.cv_repeats or C['repeats']
    draws = ctx.args.placebo_draws
    rows, ct_rows, pl_rows = [], [], []
    for d in C['directions']:
        DM, names, ABS, REL = P.passage_level_features(ctx.found.table, d)
        y = class_labels(DM.pord, C['classes'])
        counts = {k: int((y == k).sum()) for k in range(len(C['classes']))}
        keep = [k for k, n in counts.items() if n >= C['min_class_passages']]
        if len(keep) < 2:
            log(f'  {d}: fewer than two classes with enough passages; skipped.')
            continue
        use = np.isin(y, keep)
        remap = {k: i for i, k in enumerate(keep)}
        yu = np.array([remap[v] for v in y[use]], int)
        gu = DM.pord[use]
        reg = regime_ids(ctx, (1, 3, 4, 5), DM.pord)
        regu = reg[use]
        # Cross-tabulation of maintenance class and joint instrument regime, with Cramer's V.
        ur = np.unique(regu)
        tab = np.array([[int(np.sum((yu == i) & (regu == r))) for r in ur] for i in range(len(keep))])
        for i, k in enumerate(keep):
            for j, r in enumerate(ur):
                if tab[i, j]:
                    first = P.o2d(gu[regu == r].min())
                    last = P.o2d(gu[regu == r].max())
                    ct_rows.append([d, C['classes'][k][0], r, first, last, int(tab[i, j]), float(tab[i, j] / tab[i].sum())])
        chi2 = stats.chi2_contingency(tab)[0] if tab.shape[1] > 1 else 0.0
        cramer = math.sqrt(chi2 / (tab.sum() * max(1, min(tab.shape) - 1))) if tab.shape[1] > 1 else 1.0
        mixed = int(sum(1 for j in range(tab.shape[1]) if np.sum(tab[:, j] > 0) > 1))
        onehot = np.array([[1.0 if r == u else 0.0 for u in ur] for r in regu])
        lev_idx = [names.index(f's{c}_rms_demeaned') for c in (1, 2, 3, 4, 5)]
        regc_abs = ABS[use].copy()
        regc_rel = REL[use].copy()
        for r in ur:
            s = regu == r
            regc_abs[s] = regc_abs[s] - P.nanmed(regc_abs[s], axis=0)[None, :]
            regc_rel[s] = regc_rel[s] - P.nanmed(regc_rel[s], axis=0)[None, :]
        sets = [('real labels, absolute_110', ABS[use]), ('real labels, relative_110', REL[use]),
                ('regime identity only (one-hot)', onehot), ('channel levels only (log RMS of s1 to s5)', ABS[use][:, lev_idx]),
                ('absolute_110 centred within each regime', regc_abs), ('relative_110 centred within each regime', regc_rel)]
        for name, X in sets:
            for clf in ('knn', 'lda'):
                for scheme in ('random', 'blocked_in_time'):
                    m, n = cv_mean(X, yu, gu, scheme, clf, C, ctx.rng, 1 if scheme == 'blocked_in_time' else reps)
                    rows.append([d, name, clf, scheme, m, n, int(len(yu)), len(keep), 1.0 / len(keep), cramer, mixed])
        # Classification inside single regimes that hold at least two classes with enough passages.
        for r in ur:
            s = regu == r
            cls, cnt = np.unique(yu[s], return_counts=True)
            ok = cls[cnt >= C['min_class_passages']]
            if len(ok) < 2:
                rows.append([d, f'within regime {r} only', 'none', 'not possible', float('nan'), 0, int(s.sum()),
                             int(len(cls)), float('nan'), cramer, mixed])
                continue
            s2 = s & np.isin(yu, ok)
            y2 = np.array([list(ok).index(v) for v in yu[s2]], int)
            for clf in ('knn', 'lda'):
                m, n = cv_mean(ABS[use][s2], y2, gu[s2], 'random', clf, C, ctx.rng, reps)
                rows.append([d, f'within regime {r} only, absolute_110', clf, 'random', m, n, int(s2.sum()), len(ok),
                             1.0 / len(ok), cramer, mixed])
        # Calendar placebos: the real class windows moved by 6 and 12 months (same season in another year).
        for shift in (-365, -182, 182, 365):
            yp = class_labels(DM.pord, C['classes'], shift)
            cp = {k: int((yp == k).sum()) for k in range(len(C['classes']))}
            kp = [k for k, n in cp.items() if n >= C['min_class_passages']]
            if len(kp) < 2:
                rows.append([d, f'calendar placebo, class windows moved {shift} days', 'none', 'not possible',
                             float('nan'), 0, int(sum(cp.values())), len(kp), float('nan'), cramer, mixed])
                continue
            up = np.isin(yp, kp)
            yy = np.array([kp.index(v) for v in yp[up]], int)
            for set_name, X in (('absolute_110', ABS), ('relative_110', REL)):
                for clf in ('knn', 'lda'):
                    m, n = cv_mean(X[up], yy, DM.pord[up], 'random', clf, C, ctx.rng, reps)
                    rows.append([d, f'calendar placebo, class windows moved {shift} days, {set_name}', clf, 'random', m, n,
                                 int(up.sum()), len(kp), 1.0 / len(kp), cramer, mixed])
        # Regime-preserving placebo: day labels shuffled only among days of the same joint regime.
        days = np.unique(gu)
        lab_day = np.array([int(yu[gu == o][0]) for o in days])
        reg_day = np.array([regu[gu == o][0] for o in days], dtype=object)
        for draw in range(draws):
            perm = lab_day.copy()
            for r in np.unique(reg_day):
                s = np.where(reg_day == r)[0]
                perm[s] = perm[ctx.rng.permutation(s)]
            lk = dict(zip(days.tolist(), perm.tolist()))
            yp = np.array([lk[int(o)] for o in gu], int)
            if min(int((yp == k).sum()) for k in range(len(keep))) < C['min_class_passages']:
                continue
            changed = float(np.mean(yp != yu))
            for set_name, X in (('absolute_110', ABS[use]), ('relative_110', REL[use])):
                for clf in ('knn', 'lda'):
                    m, n = cv_mean(X, yp, gu, 'random', clf, C, ctx.rng, max(1, C['placebo_repeats']))
                    pl_rows.append([d, draw, set_name, clf, m, changed])
    wcsv(ctx, 'F1_classification_confounding.csv',
         ['direction', 'feature_set_or_test', 'classifier', 'validation', 'balanced_accuracy', 'n_repeats', 'n_passages',
          'n_classes', 'chance_level', 'cramers_v_class_by_regime', 'regimes_holding_more_than_one_class'], rows)
    wcsv(ctx, 'F2_class_by_joint_regime.csv',
         ['direction', 'class', 'joint_regime_s1_s3_s4_s5', 'regime_first_day_in_class_data', 'regime_last_day_in_class_data',
          'n_passages', 'fraction_of_class'], ct_rows)
    wcsv(ctx, 'F3_regime_preserving_placebo.csv',
         ['direction', 'draw', 'feature_set', 'classifier', 'balanced_accuracy', 'share_of_labels_changed'], pl_rows)
    summ = {}
    for d in C['directions']:
        for set_name in ('absolute_110', 'relative_110'):
            for clf in ('knn', 'lda'):
                real = [r[4] for r in rows if r[0] == d and r[1] == f'real labels, {set_name}' and r[2] == clf
                        and r[3] == 'random']
                pv = np.array([r[4] for r in pl_rows if r[0] == d and r[2] == set_name and r[3] == clf], float)
                if real and len(pv):
                    summ[f'{d}|{set_name}|{clf}'] = {'real_random_cv': real[0], 'placebo_median': float(np.median(pv)),
                                                     'real_percentile': float(np.mean(pv <= real[0])), 'n': int(len(pv))}
    ctx.summary['F_regime_preserving_placebo'] = summ
    ctx.summary['F_tests'] = [{'direction': r[0], 'test': r[1], 'clf': r[2], 'validation': r[3], 'balanced_accuracy': r[4],
                               'cramers_v': r[9], 'mixed_regimes': r[10]} for r in rows]


# ----------------------------------------------------------------------------
# G: detection limits
# ----------------------------------------------------------------------------
def first_hit(prob_fn, grid, sign, target):
    for g in grid:
        if prob_fn(sign * g) >= target:
            return float(sign * g)
    return float('nan')


def mdc_from(prob_fn, scale, grid_max, step=0.01, target=0.8):
    """Smallest step (on the frozen 0.01 grid) detected with probability target, upwards and downwards."""
    grid = np.arange(step, grid_max + 1e-9, step)
    mu = first_hit(prob_fn, grid, 1.0, target)
    md = first_hit(prob_fn, grid, -1.0, target)
    return (mu, md, amp_pct(mu, scale) if np.isfinite(mu) else float('nan'),
            amp_pct(md, scale) if np.isfinite(md) else float('nan'))


def block_G(ctx):
    log('Block G: sensitivity of detection limits')
    rows = []
    DT = ctx.DT
    rc = ctx.summary.get('B_residual_correlation', {}).get('development rolling origin', {})
    emp_t, emp_s = rc.get('temporal_lag1'), rc.get('spatial_lag1')
    for key in ctx.keys:
        st = dev_store(ctx, key)
        scale = P.quantity_scale(P.ind_feature(key))
        gmax = 3.0 * DT['delta_max'] * scale
        for D in st['series'].values():
            P.compute_states(D, st['fits'], ctx.FC)
        nulls = P.segment_null_stats(st['series'], ctx.pm, st['fits'][ctx.pm]['params'], DT, ctx.cal_ord + 1,
                                     P.extra_q(st['fits'], ctx.pm))
        s11 = nulls.get((1, 1))
        sd_z = float(np.std(s11[0])) if s11 is not None and len(s11[0]) else 1.0
        for (L, m), (Sv, Wv) in nulls.items():
            if len(Sv) < 20:
                continue
            T5 = float(np.quantile(np.abs(Sv), 0.95))
            for b in (0.01, 0.05, 0.10):
                T = float(np.quantile(np.abs(Sv), 1 - b))
                res = mdc_from(lambda dl: np.mean(np.abs(Sv + dl * Wv) > T), scale, gmax)
                rows.append([key, 'empirical null, alarm burden', b, L, m, T, float(np.mean(np.abs(Sv) > T))] + list(res))
            for c in (0.75, 1.0, 1.5):
                res = mdc_from(lambda dl: np.mean(np.abs(c * Sv + dl * Wv) > T5), scale, gmax)
                rows.append([key, 'noise scaled, threshold fixed at frozen 5% value', c, L, m, T5,
                             float(np.mean(np.abs(c * Sv) > T5))] + list(res))
                Tc = float(np.quantile(np.abs(c * Sv), 0.95))
                res = mdc_from(lambda dl: np.mean(np.abs(c * Sv + dl * Wv) > Tc), scale, gmax)
                rows.append([key, 'noise scaled, threshold recalibrated to 5%', c, L, m, Tc, 0.05] + list(res))
            ramp = (m + 1) / (2.0 * m)
            res = mdc_from(lambda dl: np.mean(np.abs(Sv + dl * ramp * Wv) > T5), scale, gmax)
            rows.append([key, 'linear ramp reaching the change on the last day', 1.0, L, m, T5, 0.05] + list(res))
            wbar = float(np.median(Wv))
            rho_t_list = [0.0, 0.3, 0.6] + ([round(emp_t, 3)] if emp_t is not None and np.isfinite(emp_t) else [])
            rho_s_list = [0.0, 0.3, 0.6] + ([round(emp_s, 3)] if emp_s is not None and np.isfinite(emp_s) else [])
            for rt in rho_t_list:
                for rs in rho_s_list:
                    it = np.arange(m)
                    isp = np.arange(L)
                    vt = float(np.sum(rt ** np.abs(it[:, None] - it[None, :]))) / m ** 2
                    vs = float(np.sum(rs ** np.abs(isp[:, None] - isp[None, :]))) / L ** 2
                    sS = sd_z * math.sqrt(vt * vs)
                    T = stats.norm.ppf(0.975) * sS
                    pf = lambda dl: stats.norm.cdf((dl * wbar - T) / sS) + stats.norm.cdf((-dl * wbar - T) / sS)
                    res = mdc_from(pf, scale, gmax)
                    rows.append([key, f'Gaussian errors, temporal correlation {rt}, spatial correlation {rs}', rt,
                                 L, m, T, 0.05] + list(res))
    wcsv(ctx, 'G1_detection_sensitivity.csv',
         ['indicator', 'variant', 'parameter', 'segment_bins', 'days_after', 'threshold', 'false_alarm_rate',
          'mdc80_increase_log', 'mdc80_decrease_log', 'mdc80_increase_amplitude_percent',
          'mdc80_decrease_amplitude_percent'], rows)
    ctx.summary['G_primary'] = [{'variant': r[1], 'parameter': r[2], 'L': r[3], 'm': r[4], 'false_alarm': r[6],
                                 'mdc_up_amp': r[9], 'mdc_down_amp': r[10]}
                                for r in rows if r[0] == ctx.prim and r[3] == 5 and r[4] == 3]
    fig_G(ctx, rows)


# ----------------------------------------------------------------------------
# H: episodes, leave-one-episode-out, inspection decisions
# ----------------------------------------------------------------------------
def block_H(ctx):
    log('Block H: episode definition, leave-one-episode-out and inspection decisions')
    g_rows, loo_rows, bud_rows, rule_rows = [], [], [], []
    key = ctx.prim
    st = dev_store(ctx, key)
    sets = [('development rolling origin', 'next observation day', st['recs']['next observation day']),
            ('development rolling origin', 'at least 30 days ahead', st['recs']['at least 30 days ahead'])]
    hs = holdout_store(ctx, key)
    if hs is not None:
        sets += [('holdout sequential', 'next observation day', hs['recs']['holdout sequential']),
                 ('holdout fixed origin', 'fixed origin', hs['recs']['holdout fixed origin'])]
    for scope, hl, R in sets:
        for gap in (3, 7, 14, 30):
            R2 = regroup(R, ctx.FC, gap)
            for base in ('persistence', 'fingerprint'):
                s = skill(R2, ctx.pm, base, ctx)
                g_rows.append([scope, key, hl, gap, base, s[3], s[0], s[1], s[2], s[4], s[5], s[6]])
        if scope.startswith('development'):
            eps = np.unique(R['episode'])
            for e in eps:
                S = subset(R, R['episode'] != e)
                for base in ('persistence', 'fingerprint'):
                    am = np.abs(S['y'] - S[ctx.pm])
                    ab = np.abs(S['y'] - S[base])
                    k, n, p = P.sign_summary(P.episode_skills(am, ab, S['episode'])[1])
                    loo_rows.append([scope, key, hl, e, base, 1 - am.sum() / ab.sum(), k, n, p])
    # Holdout leave-one-episode-out from the saved per-episode results (no holdout features needed).
    epath = ctx.study / 'holdout_results' / 'holdout_episode_metrics.csv'
    if epath.is_file():
        with open(epath, newline='', encoding='utf-8') as f:
            er = [r for r in csv.DictReader(f) if r['indicator'] == key]
        for scope in ('holdout sequential', 'holdout fixed origin'):
            sub = [r for r in er if r['scope'] == scope]
            byep = defaultdict(dict)
            for r in sub:
                byep[r['episode']][r['model']] = (float(r['mae']) * int(r['n_bin_days']), int(r['n_bin_days']))
            eps = sorted(byep)
            for e in eps:
                rest = [x for x in eps if x != e]
                for base in ('persistence', 'fingerprint'):
                    sm = sum(byep[x][ctx.pm][0] for x in rest if ctx.pm in byep[x])
                    sb = sum(byep[x][base][0] for x in rest if base in byep[x])
                    ks = [1 - byep[x][ctx.pm][0] / byep[x][base][0] for x in rest
                          if ctx.pm in byep[x] and base in byep[x] and byep[x][base][0] > 0]
                    k, n, p = P.sign_summary(ks)
                    loo_rows.append([scope + ' (saved episode results)', key, '', e, base,
                                     1 - sm / sb if sb > 0 else float('nan'), k, n, p])
    # Inspection decisions: bins chosen by the forecasts against the bins whose response turns out high.
    lim = P.priority_limit(ctx.cfg)
    for scope, hl, R in sets:
        keys = instance_keys(R)
        sd = pred_sd(R, ctx.pm)
        prob = 1 - stats.norm.cdf((lim - R[ctx.pm]) / sd)
        scores = OrderedDict([('local level, exceedance probability', prob), ('local level, forecast mean', R[ctx.pm]),
                              ('persistence', R['persistence']), ('moving mean', R['moving_mean'])])
        for event_name in ('anomaly above review limit', 'top 10% of bins of that day'):
            tot = {(s, b): [0, 0, 0] for s in scores for b in BUDGETS}
            n_inst, n_ev = 0, 0
            for kk in np.unique(keys):
                s = keys == kk
                n = int(s.sum())
                if n < ctx.FC['min_bins_ranking']:
                    continue
                yv = R['y'][s]
                if event_name.startswith('anomaly'):
                    ev = yv > lim
                else:
                    ev = np.zeros(n, bool)
                    ev[np.argsort(-yv, kind='stable')[:max(1, int(math.ceil(0.1 * n)))]] = True
                if not ev.any():
                    continue
                n_inst += 1
                n_ev += int(ev.sum())
                for sname, sc in scores.items():
                    order = np.argsort(-sc[s], kind='stable')
                    for b in BUDGETS:
                        kb = max(1, int(round(b * n)))
                        pick = order[:kb]
                        T_ = tot[(sname, b)]
                        T_[0] += int(ev[pick].sum())
                        T_[1] += int(ev.sum())
                        T_[2] += kb
            for (sname, b), (hit, evs, sel) in tot.items():
                if evs:
                    bud_rows.append([scope, key, hl, event_name, sname, b, n_inst, n_ev, hit / evs, hit / sel,
                                     (hit / evs) / b])
        for pstar in (0.2, 0.5, 0.8):
            flag = prob > pstar
            ev = R['y'] > lim
            n_inst = len(np.unique(keys))
            rule_rows.append([scope, key, hl, lim, pstar, n_inst, float(np.sum(flag)) / n_inst, float(np.mean(flag)),
                              float(np.sum(flag & ev) / max(1, np.sum(flag))), float(np.sum(flag & ev) / max(1, np.sum(ev))),
                              float(np.mean(ev))])
    wcsv(ctx, 'H1_episode_definition.csv', ['scope', 'indicator', 'horizon', 'episode_gap_days', 'baseline', 'n_episodes',
                                            'skill', 'skill_lo95', 'skill_hi95', 'episodes_better', 'episodes_scored',
                                            'sign_test_p'], g_rows)
    wcsv(ctx, 'H2_leave_one_episode_out.csv', ['scope', 'indicator', 'horizon', 'episode_left_out', 'baseline',
                                               'pooled_skill_without_episode', 'episodes_better', 'episodes_scored',
                                               'sign_test_p'], loo_rows)
    wcsv(ctx, 'H3_inspection_budget.csv', ['scope', 'indicator', 'horizon', 'event', 'ranking_score', 'budget_share_of_bins',
                                           'decision_instances', 'events', 'recall', 'precision', 'lift_over_random'],
         bud_rows)
    wcsv(ctx, 'H4_probability_rule.csv', ['scope', 'indicator', 'horizon', 'review_limit_log', 'probability_threshold',
                                          'decision_instances', 'bins_flagged_per_instance', 'share_of_bins_flagged',
                                          'precision', 'recall', 'share_of_bin_days_above_limit'], rule_rows)
    ctx.summary['H_episode_gap'] = [{'scope': r[0], 'horizon': r[2], 'gap': r[3], 'baseline': r[4], 'episodes': r[5],
                                     'skill': r[6], 'better': f'{r[9]} of {r[10]}', 'p': r[11]} for r in g_rows]
    ctx.summary['H_loo_min_skill'] = {}
    for r in loo_rows:
        k = f'{r[0]}|{r[2]}|{r[4]}'
        v = ctx.summary['H_loo_min_skill'].get(k)
        ctx.summary['H_loo_min_skill'][k] = r[5] if v is None else min(v, r[5])
    ctx.summary['H_budget_10pct'] = [{'scope': r[0], 'horizon': r[2], 'event': r[3], 'score': r[4], 'recall': r[8],
                                      'precision': r[9], 'lift': r[10]} for r in bud_rows if abs(r[5] - 0.10) < 1e-9]
    fig_H(ctx, bud_rows)


# ----------------------------------------------------------------------------
# I: audit of the confirmatory statistics
# ----------------------------------------------------------------------------
def block_I(ctx):
    log('Block I: audit of the confirmatory statistics')
    path = ctx.study / 'holdout_results' / 'HOLDOUT_RESULTS.json'
    if not path.is_file():
        log('  HOLDOUT_RESULTS.json not found; block I skipped.')
        return
    res = json.loads(path.read_text(encoding='utf-8'))
    hyp = res['hypotheses']
    alpha = res.get('decision_alpha', 0.05)
    rows = []
    order = [h for h in hyp if h.startswith('H')]
    pvals = OrderedDict()
    for h in order:
        v = hyp[h]
        k = v.get('episodes_better', v.get('episodes_positive'))
        n = v.get('episodes')
        p_saved = v.get('sign_test_p')
        p_re = float(stats.binomtest(int(k), int(n), 0.5, alternative='greater').pvalue) if k is not None else None
        pvals[h] = p_re
        rows.append([h, k, n, p_saved, p_re, (abs(p_saved - p_re) < 1e-12) if p_re is not None else None,
                     bool(v.get('supported')), 'interval criterion 0.70 to 0.90' if p_re is None else 'exact one-sided sign test',
                     float(0.5 ** n) if n else None])
    # Multiplicity: registered rule, fixed sequence in the registered order, Holm and Bonferroni over the sign tests.
    fixed, gate = OrderedDict(), True
    for h in order:
        ok = (pvals[h] < alpha) if pvals[h] is not None else bool(hyp[h].get('supported'))
        fixed[h] = bool(gate and ok)
        gate = gate and ok
    tested = [h for h in order if pvals[h] is not None]
    srt = sorted(tested, key=lambda h: pvals[h])
    holm, stop = {}, False
    for i, h in enumerate(srt):
        thr = alpha / (len(srt) - i)
        if stop or pvals[h] >= thr:
            stop = True
            holm[h] = False
        else:
            holm[h] = True
    bonf = {h: pvals[h] < alpha / len(tested) for h in tested}
    mrows = [[h, pvals[h], bool(hyp[h].get('supported')), fixed[h], holm.get(h), bonf.get(h)] for h in order]
    wcsv(ctx, 'I1_sign_test_audit.csv', ['hypothesis', 'episodes_favourable', 'episodes', 'p_saved', 'p_recomputed_binomtest',
                                         'identical', 'supported_as_registered', 'decision_basis',
                                         'smallest_attainable_p'], rows)
    wcsv(ctx, 'I2_multiplicity.csv', ['hypothesis', 'p', 'registered_rule', 'fixed_sequence_registered_order',
                                      'holm_over_sign_tests', 'bonferroni_over_sign_tests'], mrows)
    n_ep = max(int(hyp[h].get('episodes') or 0) for h in order)
    ctx.summary['I'] = {'p_recomputed': pvals, 'fixed_sequence': fixed, 'holm': holm, 'bonferroni': bonf,
                        'distinct_bootstrap_resamples_of_episodes': math.comb(2 * n_ep - 1, n_ep) if n_ep else None}


# ----------------------------------------------------------------------------
# Figures (optional; need Matplotlib)
# ----------------------------------------------------------------------------
MM = 1 / 25.4
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#6250d6', '#eda100']


def get_plt():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                             'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                             'grid.color': '#e9e9e9', 'grid.linewidth': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42,
                             'axes.linewidth': 0.6})
        return plt
    except Exception as exc:
        log(f'Matplotlib unavailable ({exc}); figures skipped.')
        return None


def save_fig(ctx, fig, stem):
    for ext, kw in (('png', {'dpi': 600}), ('pdf', {})):
        p = ctx.out / 'figures' / f'{stem}.{ext}'
        fig.savefig(p, bbox_inches='tight', pad_inches=0.03, **kw)
        ctx.files.append(f'figures/{stem}.{ext}')
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_A(ctx, rel_rows, pit_rows):
    plt = get_plt()
    if plt is None:
        return
    try:
        fig, ax = plt.subplots(1, 2, figsize=(160 * MM, 62 * MM))
        cases = [('development rolling origin', 'next observation day'), ('development rolling origin', 'at least 30 days ahead'),
                 ('holdout sequential', 'next observation day'), ('holdout fixed origin', 'fixed origin')]
        ax[0].plot([0, 1], [0, 1], color='#000000', lw=0.6)
        for i, (sc, hl) in enumerate(cases):
            r = [x for x in rel_rows if x[0] == sc and x[2] == hl and x[1] == ctx.prim and x[3] == ctx.pm]
            if r:
                ax[0].plot([x[4] for x in r], [x[5] for x in r], marker='o', ms=3, lw=1.2, color=COL[i], label=f'{sc}, {hl}')
        ax[0].set_xlabel('Nominal coverage')
        ax[0].set_ylabel('Empirical coverage')
        ax[0].set_title('(a) Reliability of central intervals', loc='left')
        ax[0].legend(fontsize=6.5, loc='upper left')
        for i, (sc, hl) in enumerate([cases[0], cases[2]]):
            r = [x for x in pit_rows if x[0] == sc and x[2] == hl and x[1] == ctx.prim and x[3] == ctx.pm]
            if r:
                ax[1].bar(np.arange(10) + (i - 0.5) * 0.4, [x[6] for x in r], width=0.38, color=COL[2 * i], label=sc)
        ax[1].axhline(0.1, color='#000000', lw=0.6, ls=(0, (3, 3)))
        ax[1].set_xticks(range(10))
        ax[1].set_xticklabels([f'{i / 10:.1f}' for i in range(10)])
        ax[1].set_xlabel('PIT bin (lower edge)')
        ax[1].set_ylabel('Fraction of bin-days')
        ax[1].set_title('(b) PIT histogram, next-day forecasts', loc='left')
        ax[1].legend(fontsize=6.5)
        fig.tight_layout()
        save_fig(ctx, fig, 'R1_fig_A_calibration')
    except Exception as exc:
        log(f'Figure A failed: {exc}')


def fig_B(ctx, res_rows):
    plt = get_plt()
    if plt is None:
        return
    try:
        fig, ax = plt.subplots(1, 2, figsize=(160 * MM, 58 * MM))
        for i, sc in enumerate(['development rolling origin', 'holdout sequential']):
            r = [x for x in res_rows if x[0] == sc and x[1] == ctx.prim and x[2] == ctx.pm]
            if not r:
                continue
            r = r[0]
            ax[0].plot([1, 2, 3, 4, 5], r[10:15], marker='o', ms=3, color=COL[i], label=sc)
            ax[1].plot([1, 2, 3], r[16:19], marker='o', ms=3, color=COL[i], label=sc)
        for a, t, xl in ((ax[0], '(a) Between successive observation days', 'Lag (observation days)'),
                         (ax[1], '(b) Between bins on the same day', 'Lag (100 m bins)')):
            a.axhline(0, color='#000000', lw=0.6)
            a.set_title(t, loc='left')
            a.set_xlabel(xl)
            a.set_ylabel('Correlation of standardised errors')
            a.legend(fontsize=6.5)
        ax[0].set_xticks([1, 2, 3, 4, 5])
        ax[1].set_xticks([1, 2, 3])
        fig.tight_layout()
        save_fig(ctx, fig, 'R1_fig_B_residual_correlation')
    except Exception as exc:
        log(f'Figure B failed: {exc}')


def fig_D(ctx, rows):
    plt = get_plt()
    if plt is None:
        return
    try:
        r = [x for x in rows if x[1] == 'next observation day']
        fig, ax = plt.subplots(1, 2, figsize=(160 * MM, (30 + 6 * len(r)) * MM), sharey=True)
        yy = np.arange(len(r))[::-1]
        for a, (c, lo, hi), t in ((ax[0], (8, 9, 10), '(a) Skill against persistence'),
                                  (ax[1], (12, 13, 14), '(b) Skill against the fingerprint')):
            a.errorbar([x[c] for x in r], yy, xerr=[[x[c] - x[lo] for x in r], [x[hi] - x[c] for x in r]], fmt='o',
                       ms=3.5, color=COL[0], ecolor=COL[0], elinewidth=1, capsize=0)
            a.axvline(0, color='#000000', lw=0.6)
            a.axvline(r[0][c], color=COL[1], lw=0.8, ls=(0, (3, 3)))
            a.set_title(t, loc='left')
            a.set_xlabel('Skill (1 - MAE ratio), 95% episode bootstrap')
        ax[0].set_yticks(yy)
        ax[0].set_yticklabels([x[0] for x in r], fontsize=7)
        fig.tight_layout()
        save_fig(ctx, fig, 'R1_fig_D_sensitivity')
    except Exception as exc:
        log(f'Figure D failed: {exc}')


def fig_G(ctx, rows):
    plt = get_plt()
    if plt is None:
        return
    try:
        fig, ax = plt.subplots(1, 2, figsize=(160 * MM, 60 * MM))
        r = [x for x in rows if x[0] == ctx.prim and x[1] == 'empirical null, alarm burden' and x[4] == 3]
        for i, L in enumerate((1, 3, 5)):
            s = sorted([x for x in r if x[3] == L], key=lambda x: x[2])
            ax[0].plot([100 * x[2] for x in s], [x[9] for x in s], marker='o', ms=3, color=COL[i], label=f'{100 * L} m')
        ax[0].set_xlabel('Alarm burden (%)')
        ax[0].set_ylabel('MDC80, increase (% amplitude)')
        ax[0].set_title('(a) Empirical null, 3 days', loc='left')
        ax[0].legend(fontsize=6.5)
        g = [x for x in rows if x[0] == ctx.prim and x[1].startswith('Gaussian') and x[3] == 5 and x[4] == 3]
        for i, rs in enumerate((0.0, 0.3, 0.6)):
            s = sorted([x for x in g if x[1].endswith(f'spatial correlation {rs}') and x[2] in (0.0, 0.3, 0.6)],
                       key=lambda x: x[2])
            ax[1].plot([x[2] for x in s], [x[9] for x in s], marker='o', ms=3, color=COL[i],
                       label=f'spatial correlation {rs}')
        ax[1].set_xlabel('Temporal correlation between successive days')
        ax[1].set_ylabel('MDC80, increase (% amplitude)')
        ax[1].set_title('(b) Gaussian errors, 500 m and 3 days', loc='left')
        ax[1].legend(fontsize=6.5)
        fig.tight_layout()
        save_fig(ctx, fig, 'R1_fig_G_detection_sensitivity')
    except Exception as exc:
        log(f'Figure G failed: {exc}')


def fig_H(ctx, rows):
    plt = get_plt()
    if plt is None:
        return
    try:
        scs = [s for s in ('development rolling origin', 'holdout sequential') if any(x[0] == s for x in rows)]
        fig, ax = plt.subplots(1, len(scs), figsize=(160 * MM, 60 * MM), squeeze=False)
        for j, sc in enumerate(scs):
            a = ax[0, j]
            for i, sname in enumerate(['local level, exceedance probability', 'local level, forecast mean', 'persistence',
                                       'moving mean']):
                s = [x for x in rows if x[0] == sc and x[2] == 'next observation day' and x[3] == 'anomaly above review limit'
                     and x[4] == sname]
                if s:
                    a.plot([100 * x[5] for x in s], [100 * x[8] for x in s], marker='o', ms=3, color=COL[i], label=sname)
            a.plot([0, 50], [0, 50], color='#000000', lw=0.6, ls=(0, (3, 3)), label='random selection')
            a.set_xlabel('Bins inspected (% of route)')
            a.set_ylabel('High-response bin-days found (%)')
            a.set_title(f'({"ab"[j]}) {sc}, next day', loc='left')
            a.legend(fontsize=6)
        fig.tight_layout()
        save_fig(ctx, fig, 'R1_fig_H_inspection_budget')
    except Exception as exc:
        log(f'Figure H failed: {exc}')


# ----------------------------------------------------------------------------
# Summary, manifest and command line
# ----------------------------------------------------------------------------
def finish(ctx):
    P.write_json(ctx.out / 'REVISION_SUMMARY.json', OrderedDict([('checks', ctx.checks), ('summary', ctx.summary)]))
    ctx.files.append('REVISION_SUMMARY.json')
    P.write_text(ctx.out / 'revision_log.txt', '\n'.join(P.LOG_LINES) + '\n')
    files = OrderedDict((f, sha256_file(ctx.out / f)) for f in sorted(set(ctx.files)) if (ctx.out / f).is_file())
    man = OrderedDict([
        ('created_utc', datetime.now(timezone.utc).isoformat(timespec='seconds')),
        ('module', 'railtrack_revision_analyses.py'), ('module_version', MODULE_VERSION),
        ('module_sha256', sha256_file(Path(__file__).resolve())),
        ('purpose', 'Post hoc analyses for the first revision; the confirmatory protocol and its results are unchanged.'),
        ('frozen_pipeline_sha256', ctx.pipeline_sha), ('protocol_sha256', ctx.psha),
        ('foundation_sha256', ctx.found.sha256), ('holdout_features_read', ctx.hold is not None),
        ('holdout_features_sha256', ctx.fsha), ('temperature_file', ctx.args.temperature_csv),
        ('temperature_file_sha256', sha256_file(ctx.args.temperature_csv) if ctx.args.temperature_csv else None),
        ('blocks', ctx.args.blocks), ('bootstrap', ctx.B), ('python', sys.version.split()[0]),
        ('numpy', np.__version__), ('scipy', __import__('scipy').__version__), ('host', platform.node()),
        ('run_minutes', round((time.monotonic() - ctx.t_start) / 60, 1)), ('checks', ctx.checks), ('files', files)])
    P.write_json(ctx.out / 'REVISION_MANIFEST.json', man)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline', default=DEFAULTS['pipeline'])
    ap.add_argument('--foundation-dir', default=DEFAULTS['foundation_dir'])
    ap.add_argument('--study-dir', default=DEFAULTS['study_dir'])
    ap.add_argument('--allow-holdout-diagnostics', action='store_true',
                    help='read the frozen holdout features again to describe the frozen models (post hoc)')
    ap.add_argument('--temperature-csv', default=None, help='optional daily temperatures (DATE, TAVG or TMAX and TMIN)')
    ap.add_argument('--blocks', default=ALL_BLOCKS)
    ap.add_argument('--bootstrap', type=int, default=None)
    ap.add_argument('--cv-repeats', type=int, default=None)
    ap.add_argument('--placebo-draws', type=int, default=100)
    ap.add_argument('--no-ledger', action='store_true', help='do not record the run in the protocol ledger')
    ap.add_argument('--allow-pipeline-mismatch', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    ctx = setup(args)
    if not args.no_ledger:
        P.ledger_append(ctx.study, 'post-hoc revision analyses started', module_version=MODULE_VERSION,
                        module_sha256=sha256_file(Path(__file__).resolve()), protocol_sha256=ctx.psha,
                        holdout_features_read=ctx.hold is not None, holdout_features_sha256=ctx.fsha,
                        note='Post hoc, for the journal revision; confirmatory results unchanged.')
    reproduction_checks(ctx)
    blocks = {'A': block_A, 'B': block_B, 'C': block_C, 'D': block_D, 'E': block_E, 'F': block_F, 'G': block_G,
              'H': block_H, 'I': block_I}
    failed = []
    for b in ALL_BLOCKS:
        if b not in args.blocks.upper():
            continue
        t0 = time.monotonic()
        # Each block has its own random stream, so a block re-run alone gives the same numbers.
        ctx.rng = np.random.default_rng(ctx.cfg['seed'] + 101 + ord(b))
        try:
            blocks[b](ctx)
        except Exception as exc:
            import traceback
            failed.append(b)
            log(f'BLOCK {b} FAILED: {exc}')
            log(traceback.format_exc())
        log(f'Block {b} finished in {time.monotonic() - t0:.0f} s')
    ctx.checks['failed_blocks'] = failed
    man = finish(ctx)
    if not args.no_ledger:
        P.ledger_append(ctx.study, 'post-hoc revision analyses completed', module_sha256=man['module_sha256'],
                        manifest_sha256=sha256_file(ctx.out / 'REVISION_MANIFEST.json'), failed_blocks=failed)
    log(f'Revision analyses complete in {man["run_minutes"]} min: {ctx.out}' + (f'; FAILED blocks {failed}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
