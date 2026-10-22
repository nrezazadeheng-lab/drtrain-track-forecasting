#!/usr/bin/env python3
"""
DR-Train LRV4306 Region 5 track-monitoring study: single pipeline, version 1.2.0
================================================================================

This one file runs the whole study that follows the verified three-file
foundation produced by build_predictive_maintenance_foundation.py.

Commands, in the order they are meant to be used:

  develop            Development-only analyses from the three foundation files:
                     1 data audit (instrument level regimes, dead-channel
                       periods, timing flags, fingerprint stability),
                     2 re-analysis of maintenance classification (random,
                       date-grouped and time-blocked validation, placebo labels
                       including time-shifted copies of the real labels),
                     3 regime-robust condition indicator and probabilistic
                       forecasting against simple baselines (local level,
                       mean-reverting level, regime-aware level, local trend),
                       with a holdout-like check at four cut dates,
                     4 agreement of anomalies between independent sensors,
                     5 scheduled-work tests: corridor minus control change
                       against a random-walk-plus-noise null fitted outside
                       scheduled-work windows, checked on clean placebo windows,
                       and a multi-channel composite,
                     6 detection limits by synthetic step injection,
                     7 end-of-development priority list,
                     plus figures and an HTML report.
  freeze             Fits the final models on all development data and writes
                     FROZEN_PROTOCOL.json (models, thresholds, hypotheses, code
                     and input hashes). Run only after reading the report.
  verify-extraction  Re-extracts a few DEVELOPMENT passages from the raw files
                     with the foundation builder and checks exact agreement
                     with the foundation table. Run before extract-holdout.
  extract-holdout    Extracts features for the 89 frozen holdout passages.
                     Needs the frozen protocol and --confirm-single-use.
  score-holdout      The single prospective evaluation, using the frozen
                     protocol and the extraction record only.
  self-test          Runs develop, freeze and score on a pseudo-holdout cut from
                     DEVELOPMENT data. It never touches the real holdout.

Every protocol step is recorded in the append-only PROTOCOL_LEDGER.jsonl in the
output folder. Inference uses calendar episodes (observation days separated by
more than seven days) as independent units; the holdout hypotheses are decided
by exact one-sided sign tests over episodes.

Dependencies: Python 3.9 or later, NumPy, SciPy. Matplotlib (figures) and
scikit-learn (random forest) are used when installed. Raw-data commands also
need h5py, as the foundation builder does.

Nothing in this program is a validated maintenance predictor. It forecasts and
monitors a measured track-response indicator and reports how well that works.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import time
import warnings
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy import optimize, stats

VERSION = '1.2.0'
EXPECTED_BUILDER_SHA256 = 'e8c5b403306691c1f101cfce75f28714903ce87682ebdb024f687f342da38852'
DEFAULTS = {
    'foundation_dir': '/data2/nr1/V1_RailTrack/FOUNDATION_for_new_chat',
    'output_dir': '/data2/nr1/V1_RailTrack/STUDY_v1',
    'data_root': '/data2/nr1/data_files',
    'builder': '/data2/nr1/V1_RailTrack/build_predictive_maintenance_foundation.py',
}
FOUNDATION_FILES = OrderedDict([
    ('features', 'development_vibration_features.csv.gz'),
    ('coverage', 'maintenance_event_coverage.csv'),
    ('manifest', 'foundation_manifest.json'),
])
DIRECTIONS = ('inbound', 'outbound')
CHANNELS = (1, 2, 3, 4, 5)
CHANNEL_NAMES = {
    1: 'bogie longitudinal (tri-axial X)',
    2: 'bogie transverse (tri-axial Y)',
    3: 'bogie vertical (tri-axial Z)',
    4: 'front cabin vertical (uni-axial)',
    5: 'back cabin vertical (uni-axial)',
    45: 'cabin vertical, geometric mean of front and back',
}
# Derived cabin channel 45: geometric mean of channels 4 and 5 for these features.
VIRTUAL_CABIN_FEATURES = ('rms_demeaned', 'bandpower_0p5_5hz', 'bandpower_5_20hz')
FEATURES = [
    'mean', 'rms_raw', 'rms_demeaned', 'peak_to_peak', 'median',
    'mean_abs_demeaned', 'p95_abs_demeaned', 'peak_abs_demeaned',
    'crest_factor', 'iqr', 'mad_median', 'skewness', 'excess_kurtosis',
    'zero_crossing_fraction', 'bandpower_0p5_5hz', 'bandpower_5_20hz',
    'bandpower_20_80hz', 'bandpower_80_250hz', 'bandpower_250_600hz',
    'bandpower_total_0p5_600hz', 'spectral_centroid_hz', 'spectral_entropy',
]
# Strictly positive amplitude or energy features: analysed on a log scale.
LOG_FEATURES = {
    'rms_raw', 'rms_demeaned', 'peak_to_peak', 'mean_abs_demeaned',
    'p95_abs_demeaned', 'peak_abs_demeaned', 'crest_factor', 'iqr',
    'mad_median', 'bandpower_0p5_5hz', 'bandpower_5_20hz', 'bandpower_20_80hz',
    'bandpower_80_250hz', 'bandpower_250_600hz', 'bandpower_total_0p5_600hz',
}
STRING_COLUMNS = [
    'passage_id', 'vehicle', 'region', 'direction', 'metadata_utc', 'local_date',
    'track_bin_id', 'corridor_candidate_bin', 'raw_acc_time_alignment',
    'analysis_screening_flag',
]
MODELS = ['fingerprint', 'persistence', 'historical_mean', 'moving_mean', 'local_level', 'mean_reverting',
          'local_level_regime', 'local_trend']
KALMAN_MODELS = ['local_level', 'mean_reverting', 'local_level_regime', 'local_trend']
KALMAN_KIND = {'local_level': 'll', 'mean_reverting': 'ou', 'local_level_regime': 'llr', 'local_trend': 'llt'}
MODEL_LABELS = {
    'fingerprint': 'Static fingerprint (no change)',
    'persistence': 'Persistence (last value)',
    'historical_mean': 'Historical mean',
    'moving_mean': 'Moving mean (30 days)',
    'local_level': 'Local level (Kalman)',
    'mean_reverting': 'Mean-reverting level (Kalman)',
    'local_level_regime': 'Regime-aware local level (Kalman)',
    'local_trend': 'Local linear trend (Kalman)',
}

CONFIG = {
    'version': VERSION,
    'seed': 20261017,
    'calibration_end': '2014-09-30',
    'speed_adjustment': True,
    'qc': {
        'min_profile_rho': 0.30,
        'min_bins_per_passage': 10,
        'min_fit_passages_per_bin': 5,
        'timing_flag_seconds': 120.0,
    },
    'indicator_family': [[c, f] for c in (1, 3, 4, 5, 45)
                         for f in ('rms_demeaned', 'bandpower_0p5_5hz', 'bandpower_5_20hz')],
    # Chosen on development evidence (forecast skill against the static fingerprint, agreement of the two
    # cabin sensors, physical link to long-wavelength geometry); fixed before any holdout data are read.
    'primary_indicator': [45, 'bandpower_0p5_5hz'],
    'secondary_indicators': [[3, 'rms_demeaned'], [45, 'rms_demeaned'], [1, 'bandpower_0p5_5hz']],
    'primary_model': 'local_level',
    'regimes': {
        'penalty_multiplier': 3.0,
        'min_segment_days': 3,
        'material_shift_log': 0.405,
        'dead_fail_fraction': 0.5,
        'min_passages_fingerprint': 10,
    },
    'classification': {
        'directions': ['outbound', 'inbound'],
        'classes': [
            ['tamping', '2014-04-01', '2014-06-12'],
            ['tamping_and_surfacing', '2015-07-01', '2015-09-28'],
            ['surfacing', '2015-09-28', '2015-11-23'],
            ['no_maintenance', '2015-11-23', '2016-01-01'],
        ],
        'folds': 5,
        'repeats': 20,
        'rf_repeats': 5,
        'rf_trees': 200,
        # shifted: the real label pattern moved by every possible offset in observation days (circular);
        # scattered: blocks of the real class sizes at random positions; adjacent: blocks back to back.
        'placebo_variants': ['shifted', 'scattered', 'adjacent'],
        'placebo_draws': 200,
        'placebo_repeats': 3,
        'rf_placebo_draws': 30,
        'knn_k': 5,
        'lda_shrinkage': 0.5,
        'min_class_passages': 5,
    },
    'forecast': {
        'horizons': ['next', 30, 90],
        'innovation_clip': 3.0,
        'initial_variance': 1.0,
        'slope_prior_sd_per_year': 1.0,
        'regime_break_log': 0.405,
        'regime_reference_days': 5,
        'regime_confirm_days': 2,
        'regime_reset_variance': 1.0,
        'moving_mean_days': 30,
        # Calendar episodes (observation days of both directions separated by more than this gap) are the
        # independent units for intervals and for the exact sign tests that decide the holdout hypotheses.
        'episode_gap_days': 7,
        'decision_alpha': 0.05,
        'bootstrap': 2000,
        'top_fraction': 0.10,
        'min_bins_ranking': 10,
        'priority_horizon_days': 30,
        'priority_limit_log': 0.20,
        'burn_in_days': 60,
    },
    'pseudo_prospective': {
        'cuts': ['2014-12-31', '2015-03-31', '2015-06-30', '2015-08-31'],
        'embargo_days': 7,
        'span_days': 134,
    },
    'cross_sensor': {
        'pairs': [['s3_rms_demeaned', 's4_rms_demeaned'], ['s3_rms_demeaned', 's5_rms_demeaned'],
                  ['s4_rms_demeaned', 's5_rms_demeaned'], ['s1_rms_demeaned', 's3_rms_demeaned'],
                  ['s3_bandpower_0p5_5hz', 's4_bandpower_0p5_5hz'],
                  ['s4_bandpower_0p5_5hz', 's5_bandpower_0p5_5hz'],
                  ['s4_bandpower_5_20hz', 's5_bandpower_5_20hz']],
        'min_bins': 10,
    },
    'events': {
        'window_days': 30,
        'min_days_each_side': 4,
        'min_bins_each_group': 5,
        'fdr_q': 0.10,
        # Null: random walk plus noise for the daily corridor-minus-control contrast, fitted on days more than
        # window_days from any scheduled work and restarted at working-regime changes. Clean placebo windows
        # (same duration, no scheduled work, no regime change) check it; the standard error is inflated by the
        # root-mean-square placebo z when that exceeds one.
        'min_clean_placebo': 30,
        'outlier_half_window': 3,
        'outlier_robust_sd': 6.0,
        'key_events': ['M0032', 'M0082', 'M0144'],
        'composite_channels': [1, 3, 4, 5],
        'composite_min_channels': 2,
        # Tamping and surfacing are expected to reduce the corridor response (reported, tests stay two-sided).
        'expected_sign': -1,
        'sensitivity_windows': [[20, 3], [45, 4], [60, 5]],
    },
    'detection': {
        'segment_bins': [1, 3, 5],
        'days_after': [1, 3, 5],
        'max_days_ahead': 60,
        'alarm_burden': 0.05,
        'power_target': 0.80,
        'delta_max': 1.0,
        'delta_step': 0.01,
    },
    'holdout_plan': {
        'origin': ('Sequential monitoring is primary: each holdout observation day is forecast from the previous observation '
                   'day (the last development day for the first holdout day), with the model updated by every new day. '
                   'Forecasts made once from the last development day (fixed origin) are secondary.'),
        'units': ('Calendar episodes: holdout observation days of both directions separated by more than seven days. The '
                  'holdout holds 13 dates in five episodes, so each decision rests on five independent units.'),
        'decision_rule': ('Exact one-sided sign test over calendar episodes at the 0.05 level. With five episodes this '
                          'requires every episode to favour the model (p = 0.031, one chance in 32). Pooled skill and 95% episode-bootstrap '
                          'intervals are reported as descriptions, not as decision rules.'),
        # Which baseline H1 is about: 'persistence' or 'fingerprint'. The hypothesis keys below must match
        # (H1_sequential_vs_<primary>, H2_sequential_vs_<other>); the program checks this.
        'primary_comparison': 'persistence',
        'power_basis': ('Chosen on development evidence before any holdout data were read: in the rolling-origin development '
                        'evaluation the primary model beat persistence in every calendar episode but beat the static '
                        'fingerprint in about three of four, so with five holdout episodes H1 has high power and H2 low '
                        'power. The development report gives the episode counts and the implied power.'),
        'hypotheses': {
            'H1_sequential_vs_persistence': 'Primary. For the primary indicator, sequential one-step-ahead forecasts of the primary model have a lower mean absolute error than persistence (the last observed daily anomaly). Supported if the skill against persistence is positive in enough calendar episodes for the exact one-sided sign test to give p below 0.05.',
            'H2_sequential_vs_fingerprint': 'Key secondary. The same comparison and rule against the static development fingerprint (no change since development), which tests for forecastable change in the track response. Development evidence gives this test low power with five episodes.',
            'H3_sequential_ranking': 'The mean Spearman correlation across bins between sequential forecasts and realised anomalies is positive. Supported if the episode means are positive in enough episodes for the exact one-sided sign test to give p below 0.05.',
            'H4_sequential_calibration': 'Coverage of the 80% sequential prediction interval of the primary model, pooled over holdout bin-days, lies between 0.70 and 0.90.',
            'H5_fixed_origin_vs_fingerprint': 'Secondary. Forecasts made once from the last development day beat the static fingerprint, by the same episode sign test. Development evidence suggests this may fail when maintenance or instrument changes intervene.',
            'D1_alarms': 'Descriptive only: share of tested segments exceeding the frozen development alarm threshold (5% expected without change), with origins at the last development day and at every holdout day, as in development; share of alarms touching the corridor.',
            'D2_secondary': 'Descriptive only: the same metrics for the secondary indicators and the other models, including the 30-day moving mean, and per-episode results.',
        },
        'note': 'The holdout holds 13 dates in five separated episodes; eight dates fall in the first episode.',
    },
}
QUICK_OVERRIDES = {
    ('classification', 'repeats'): 3,
    ('classification', 'rf_repeats'): 1,
    ('classification', 'placebo_draws'): 20,
    ('classification', 'placebo_repeats'): 1,
    ('classification', 'rf_placebo_draws'): 3,
    ('forecast', 'bootstrap'): 200,
}

# Colour tokens (reference data-visualisation palette, light mode).
PAL = {
    'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
    'magenta': '#e87ba4', 'green': '#008300', 'violet': '#6250d6', 'red': '#e34948',
}
SERIES = [PAL['blue'], PAL['orange'], PAL['aqua'], PAL['yellow'], PAL['magenta'],
          PAL['green'], PAL['violet'], PAL['red']]
INK, INK2, MUTED, GRID, AXIS, NEUTRAL = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#f0efec'

LOG_LINES = []


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def log(msg):
    line = '[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] ' + str(msg)
    print(line, flush=True)
    LOG_LINES.append(line)


def quiet_numpy_warnings():
    for msg in ('All-NaN slice encountered', 'Mean of empty slice', 'Degrees of freedom <= 0',
                'invalid value encountered', 'divide by zero encountered'):
        warnings.filterwarnings('ignore', message='.*' + msg + '.*', category=RuntimeWarning)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def this_file_sha256():
    return sha256_file(Path(__file__).resolve())


def config_hash(cfg):
    return sha256_text(json.dumps(to_jsonable(cfg), sort_keys=True))


def nanmed(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(a, axis=axis)


def nanstd(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanstd(a, axis=axis)


def nanpct(a, q, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanpercentile(a, q, axis=axis)


def safe_log(v):
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v) & (v > 0)
    out[ok] = np.log(v[ok])
    return out


def quantity_scale(feature):
    """Log power moves twice as much as log amplitude for the same gain change."""
    return 2.0 if str(feature).startswith('bandpower') else 1.0


def quantity_name(feature):
    return 'power' if str(feature).startswith('bandpower') else 'amplitude'


def mad_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float('nan')
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def d2o(s):
    return date.fromisoformat(str(s)).toordinal()


def o2d(o):
    return date.fromordinal(int(o)).isoformat()


def calendar_episodes(ords, gap_days):
    """Map every observation day to its calendar episode, labelled by the episode's first date.

    Days of both travel directions are pooled first, so that one calendar episode is one unit whatever
    the direction, indicator or cut date it is evaluated in.
    """
    uniq = np.unique(np.asarray(list(ords), int))
    out = {}
    first = None
    prev = None
    for o in uniq.tolist():
        if prev is None or o - prev > gap_days:
            first = o
        out[o] = o2d(first)
        prev = o
    return out


def sign_test_p(k, n):
    """Exact one-sided sign test: probability of k or more successes out of n under p = 0.5."""
    if n <= 0:
        return float('nan')
    return float(stats.binom.sf(k - 1, n, 0.5))


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3:
        return float('nan')
    ra = stats.rankdata(a)
    rb = stats.rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float(np.dot(ra, ra)) * float(np.dot(rb, rb)))
    return float(np.dot(ra, rb) / den) if den > 0 else float('nan')


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(len(pv))
    qq[order] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q


def gauss_crps(mu, sd, y):
    sd = np.maximum(sd, 1e-12)
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return o


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def write_text(path, text):
    atomic_write_bytes(path, text.encode('utf-8'))


def write_json(path, obj):
    write_text(path, json.dumps(to_jsonable(obj), indent=2, allow_nan=False) + '\n')


def fmt_cell(v):
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return ''
        if f != 0.0 and abs(f) < 1e-3:
            return f'{f:.6g}'  # keep six significant digits for small variances and probabilities
        return repr(round(f, 6))
    if isinstance(v, (np.integer,)):
        return str(int(v))
    return str(v)


def write_csv(path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    for r in rows:
        w.writerow([fmt_cell(v) for v in r])
    write_text(path, buf.getvalue())


def fnum(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return 'n.a.'
    return f'{f:.{nd}f}'


def ordinal(n):
    n = int(round(n))
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def pct_change(delta_log):
    return 100.0 * (math.exp(delta_log) - 1.0)


def parse_float_column(col):
    try:
        return np.array(col, dtype=float)
    except (TypeError, ValueError):
        out = np.empty(len(col))
        for i, v in enumerate(col):
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                out[i] = np.nan
        return out


def apply_quick(cfg):
    cfg = copy.deepcopy(cfg)
    for (a, b), v in QUICK_OVERRIDES.items():
        cfg[a][b] = v
    cfg['quick_mode'] = True
    return cfg


def validate_config(cfg):
    """Stop early on settings the program cannot honour."""
    fam = [list(x) for x in cfg['indicator_family']]
    if cfg['primary_model'] not in KALMAN_MODELS:
        raise ValueError(f'primary_model must be one of {KALMAN_MODELS}, not {cfg["primary_model"]!r}.')
    for ind in [cfg['primary_indicator']] + list(cfg['secondary_indicators']):
        if list(ind) not in fam:
            raise ValueError(f'Indicator {ind} is not in indicator_family.')
        if ind[0] not in CHANNEL_NAMES or ind[1] not in FEATURES:
            raise ValueError(f'Unknown channel or feature in {ind}.')
    for c, f in fam:
        if c == 45 and f not in VIRTUAL_CABIN_FEATURES:
            raise ValueError(f'The derived cabin channel 45 exists only for {VIRTUAL_CABIN_FEATURES}.')
    unknown = set(cfg['classification']['placebo_variants']) - {'shifted', 'scattered', 'adjacent'}
    if unknown:
        raise ValueError(f'Unknown placebo variants {sorted(unknown)}.')
    c1, c2 = comparison_order(cfg)
    hyp = cfg['holdout_plan']['hypotheses']
    for name in (f'H1_sequential_vs_{c1}', f'H2_sequential_vs_{c2}'):
        if name not in hyp:
            raise ValueError(f'holdout_plan hypotheses must contain {name} to match primary_comparison {c1!r}.')


def comparison_order(cfg):
    c1 = cfg['holdout_plan'].get('primary_comparison', 'persistence')
    if c1 not in ('persistence', 'fingerprint'):
        raise ValueError("primary_comparison must be 'persistence' or 'fingerprint'.")
    return c1, ('fingerprint' if c1 == 'persistence' else 'persistence')


# ----------------------------------------------------------------------------
# Protocol ledger: append-only record of every protocol step
# ----------------------------------------------------------------------------
LEDGER_NAME = 'PROTOCOL_LEDGER.jsonl'


def ledger_append(output_dir, event, **info):
    """Append one line to the ledger; earlier lines are never rewritten."""
    path = Path(output_dir) / LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = OrderedDict([('utc', datetime.now(timezone.utc).isoformat(timespec='seconds')), ('event', event),
                       ('pipeline_version', VERSION), ('pipeline_sha256', this_file_sha256()),
                       ('host', platform.node()), ('user', os.environ.get('USER', ''))])
    rec.update(info)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(to_jsonable(rec), sort_keys=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    return rec


def ledger_read(output_dir):
    path = Path(output_dir) / LEDGER_NAME
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({'event': 'unreadable ledger line', 'text': line[:200]})
    return out


def sklearn_available():
    try:
        import sklearn  # noqa: F401
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        return True
    except Exception:
        return False


def ind_key(c, f):
    return f's{c}_{f}'


def ind_feature(key):
    return key.split('_', 1)[1]


def ind_channel(key):
    return int(key.split('_', 1)[0][1:])


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
class FeatureTable:
    """Column store for a foundation-format feature table."""

    def __init__(self, s, x, header, source):
        self.s = s
        self.x = x
        self.header = header
        self.source = source
        self.n = len(s['passage_id'])
        for f in VIRTUAL_CABIN_FEATURES:
            a, b = x.get(f's4_{f}'), x.get(f's5_{f}')
            if a is not None and b is not None:
                ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
                v = np.full(len(a), np.nan)
                v[ok] = np.sqrt(a[ok] * b[ok])
                x[f's45_{f}'] = v
        self.screened = np.array(s['analysis_screening_flag'] == 'speed_screen_pass', bool)
        self.corridor = np.array(s['corridor_candidate_bin'] == 'yes', bool)
        self.ord = np.array([d2o(v) for v in s['local_date']], int) if self.n else np.zeros(0, int)

    @classmethod
    def read(cls, path):
        path = Path(path)
        opener = gzip.open if path.name.endswith('.gz') else open
        with opener(path, 'rt', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)
        cols = list(zip(*data)) if data else [() for _ in header]
        s, x = {}, {}
        for name, col in zip(header, cols):
            if name in STRING_COLUMNS:
                s[name] = np.array(col, dtype=object)
            else:
                x[name] = parse_float_column(col)
        missing = [c for c in STRING_COLUMNS if c not in s]
        if missing:
            raise ValueError(f'{path}: missing columns {missing}')
        return cls(s, x, header, str(path))

    def passages(self):
        return sorted(set(self.s['passage_id'].tolist()))


class Foundation:
    """The three verified foundation files, with consistency checks."""

    def __init__(self, folder, allow_count_mismatch=False):
        self.folder = Path(folder)
        self.paths = {k: self.folder / v for k, v in FOUNDATION_FILES.items()}
        for k, p in self.paths.items():
            if not p.is_file():
                raise FileNotFoundError(f'Foundation file not found: {p}')
        self.sha256 = {k: sha256_file(p) for k, p in self.paths.items()}
        self.manifest = json.loads(self.paths['manifest'].read_text(encoding='utf-8'))
        with open(self.paths['coverage'], newline='', encoding='utf-8-sig') as f:
            self.coverage = list(csv.DictReader(f))
        log(f'Reading {self.paths["features"]}')
        self.table = FeatureTable.read(self.paths['features'])
        self._verify(allow_count_mismatch)

    def _verify(self, allow_count_mismatch):
        m = self.manifest
        missing = [c for c in m['feature_columns'] if c not in self.table.x]
        if missing:
            raise ValueError(f'Feature table lacks {len(missing)} manifest feature columns, e.g. {missing[:3]}')
        split = m['chronological_split']
        ids = sorted(split['holdout_passage_ids'])
        h = sha256_text('\n'.join(ids))
        if h != split['holdout_id_sha256'] or len(ids) != split['complete_heldout_recordings']:
            raise ValueError('Manifest holdout identifiers do not match their recorded SHA-256 digest or count.')
        overlap = set(ids) & set(self.table.s['passage_id'].tolist())
        if overlap:
            raise ValueError(f'{len(overlap)} frozen holdout passages appear in the development table. Stop.')
        self.holdout_ids = ids
        self.holdout_hash = h
        self.dev_last_date = split['development_last_date']
        self.embargo_end = split['seven_day_embargo_end']
        if self.table.n and o2d(self.table.ord.max()) > self.dev_last_date:
            raise ValueError('Development table contains dates after the development cut-off.')
        c = m.get('counts', {})
        checks = [
            ('rows', self.table.n, c.get('development_complete_feature_windows')),
            ('speed-screened rows', int(self.table.screened.sum()), c.get('development_speed_screened_feature_windows')),
            ('passages', len(self.table.passages()), c.get('development_accepted_passages')),
        ]
        bad = [f'{name}: table {got}, manifest {exp}' for name, got, exp in checks if exp is not None and got != exp]
        if bad:
            msg = 'Foundation counts differ from the manifest: ' + '; '.join(bad)
            if not allow_count_mismatch:
                raise ValueError(msg)
            log('WARNING ' + msg)
        log(f'Foundation verified: {self.table.n} windows, {len(self.table.passages())} passages, '
            f'{len(ids)} holdout identifiers absent, development ends {self.dev_last_date}.')

    def schedule_spans(self):
        spans = defaultdict(list)
        for r in self.coverage:
            spans[r['direction']].append((d2o(r['start_date']), d2o(r['end_date']), r['record_id'], r['activity']))
        return spans


class DirMatrix:
    """Passage-by-bin matrices for one travel direction (speed-screened windows)."""

    def __init__(self, table, direction, bins=None, screened_only=True):
        m = np.array(table.s['direction'] == direction, bool)
        if screened_only:
            m &= table.screened
        rows = np.where(m)[0]
        pid = table.s['passage_id'][rows]
        dts = table.s['local_date'][rows]
        utc = table.s['metadata_utc'][rows]
        first = {}
        for p, d_, u in zip(pid.tolist(), dts.tolist(), utc.tolist()):
            first.setdefault(p, (d_, u))
        self.pids = sorted(first, key=lambda p: (first[p][0], first[p][1], p))
        self.pdate = np.array([first[p][0] for p in self.pids], dtype=object)
        self.pord = np.array([d2o(v) for v in self.pdate], int) if self.pids else np.zeros(0, int)
        pidx = {p: i for i, p in enumerate(self.pids)}
        binid = table.s['track_bin_id'][rows]
        chain = table.x['chainage_start_m'][rows]
        cor = table.corridor[rows]
        if bins is None:
            info = {}
            for b, c, k in zip(binid.tolist(), chain.tolist(), cor.tolist()):
                info.setdefault(b, (c, k))
            self.bins = sorted(info, key=lambda b: info[b][0])
            self.chainage = np.array([info[b][0] for b in self.bins], float)
            self.corridor = np.array([info[b][1] for b in self.bins], bool)
        else:
            self.bins = list(bins['ids'])
            self.chainage = np.array(bins['chainage'], float)
            self.corridor = np.array(bins['corridor'], bool)
        bidx = {b: j for j, b in enumerate(self.bins)}
        ci = np.array([bidx.get(b, -1) for b in binid.tolist()], int)
        keep = ci >= 0
        self.n_dropped_rows = int((~keep).sum())
        self.rows = rows[keep]
        self.ri = np.array([pidx[p] for p in pid[keep].tolist()], int)
        self.ci = ci[keep]
        if len(set(zip(self.ri.tolist(), self.ci.tolist()))) != len(self.ri):
            raise ValueError(f'Duplicate passage and bin combinations in direction {direction}.')
        self.table = table
        self.direction = direction
        self.P = len(self.pids)
        self.B = len(self.bins)

    def matrix(self, values):
        M = np.full((self.P, self.B), np.nan)
        if len(self.rows):
            M[self.ri, self.ci] = np.asarray(values, float)[self.rows]
        return M

    def logmatrix(self, column):
        return self.matrix(safe_log(self.table.x[column]))

    def bins_dict(self):
        return {'ids': list(self.bins), 'chainage': self.chainage.tolist(), 'corridor': self.corridor.tolist()}


# ----------------------------------------------------------------------------
# Regime-robust indicator: fingerprint, quality control, speed adjustment
# ----------------------------------------------------------------------------
def fit_fingerprint(Y, use_rows, min_count, n_iter=4):
    """Median spatial profile after removing each passage's overall level."""
    Ys = Y[np.asarray(use_rows, bool)]
    F = np.full(Y.shape[1], np.nan)
    cnt = np.zeros(Y.shape[1], int)
    if Ys.shape[0] == 0:
        return F, cnt
    off = nanmed(Ys, axis=1)
    for _ in range(n_iter):
        F = nanmed(Ys - off[:, None], axis=0)
        cnt = np.sum(np.isfinite(Ys), axis=0)
        F[cnt < min_count] = np.nan
        off = nanmed(Ys - F[None, :], axis=1)
    F = F - nanmed(F)
    return F, cnt


def profile_rho(Y, F, min_bins):
    rho = np.full(Y.shape[0], np.nan)
    nb = np.zeros(Y.shape[0], int)
    fF = np.isfinite(F)
    for i in range(Y.shape[0]):
        m = np.isfinite(Y[i]) & fF
        nb[i] = int(m.sum())
        if nb[i] >= max(min_bins, 3):
            rho[i] = spearman(Y[i, m], F[m])
    return rho, nb


def speed_deviation(V, vref):
    dv = V - np.asarray(vref, float)[None, :]
    return dv - nanmed(dv, axis=1)[:, None]


def robust_slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 50 or np.var(x) <= 1e-12:
        return 0.0
    keep = np.ones(len(x), bool)
    b, a = 0.0, 0.0
    for _ in range(4):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x)
        s = mad_sd(r[keep])
        if not np.isfinite(s) or s <= 0:
            break
        keep = np.abs(r) <= 3 * s
    return float(b)


def fit_indicator_model(table, channel, feature, fit_last_date, cfg):
    """Fingerprints, speed reference and speed slope from passages up to fit_last_date."""
    col = f's{channel}_{feature}'
    qc = cfg['qc']
    rho_min, min_bins, min_cnt = qc['min_profile_rho'], qc['min_bins_per_passage'], qc['min_fit_passages_per_bin']
    last_ord = d2o(fit_last_date)
    mats, xs, ys = {}, [], []
    for d in DIRECTIONS:
        DM = DirMatrix(table, d)
        Y = DM.logmatrix(col)
        V = DM.logmatrix('estimated_speed_mps')
        fit = DM.pord <= last_ord
        F0, _ = fit_fingerprint(Y, fit, min_cnt)
        rho0, nb0 = profile_rho(Y, F0, min_bins)
        good = fit & (rho0 >= rho_min) & (nb0 >= min_bins)
        F1, _ = fit_fingerprint(Y, good, min_cnt)
        vref = nanmed(V[good], axis=0) if good.any() else np.full(DM.B, np.nan)
        dvr = speed_deviation(V, vref)
        off1 = nanmed(Y - F1[None, :], axis=1)
        R = Y - off1[:, None] - F1[None, :]
        sel = good[:, None] & np.isfinite(R) & np.isfinite(dvr)
        xs.append(dvr[sel])
        ys.append(R[sel])
        mats[d] = (DM, Y, V, fit, vref)
    beta = robust_slope(np.concatenate(xs), np.concatenate(ys)) if cfg.get('speed_adjustment', True) else 0.0
    dirs = {}
    for d in DIRECTIONS:
        DM, Y, V, fit, vref = mats[d]
        Ya = Y - beta * speed_deviation(V, vref)
        F2a, _ = fit_fingerprint(Ya, fit, min_cnt)
        rho2, nb2 = profile_rho(Ya, F2a, min_bins)
        good2 = fit & (rho2 >= rho_min) & (nb2 >= min_bins)
        F2, cnt2 = fit_fingerprint(Ya, good2, min_cnt)
        dirs[d] = {
            'bins': DM.bins_dict(), 'F': F2.tolist(), 'vref': np.asarray(vref).tolist(),
            'fit_count': cnt2.tolist(), 'n_fit_passages': int(fit.sum()),
            'n_fit_passages_qc': int(good2.sum()),
        }
    return {
        'channel': int(channel), 'feature': feature, 'column': col,
        'label': f's{channel} {feature} ({CHANNEL_NAMES[channel]})',
        'beta_speed': float(beta), 'rho_min': rho_min, 'min_bins': min_bins,
        'fit_last_date': fit_last_date, 'directions': dirs,
    }


def apply_indicator_model(model, table, d):
    md = model['directions'][d]
    DM = DirMatrix(table, d, bins=md['bins'])
    Y = DM.logmatrix(model['column'])
    V = DM.logmatrix('estimated_speed_mps')
    F = np.array([np.nan if v is None else v for v in md['F']], float)
    vref = np.array([np.nan if v is None else v for v in md['vref']], float)
    dv = speed_deviation(V, vref)
    Ya = Y - model['beta_speed'] * dv
    off = nanmed(Ya - F[None, :], axis=1)
    E = Ya - off[:, None] - F[None, :]
    rho, nb = profile_rho(Ya, F, model['min_bins'])
    qc = (rho >= model['rho_min']) & (nb >= model['min_bins'])
    return {'DM': DM, 'E': E, 'qc': qc, 'rho': rho, 'nb': nb, 'offset': off,
            'pord': DM.pord, 'pids': DM.pids, 'dv': dv}


def residual_speed_correlation(table, model, d, after_ord):
    """Correlation between the passage anomaly and the speed deviation after a date (QC-passing passages).

    A value near zero shows that one speed slope, pooled over both directions, removes the speed effect.
    """
    A = apply_indicator_model(model, table, d)
    rows = A['qc'] & (A['pord'] > after_ord)
    m = rows[:, None] & np.isfinite(A['E']) & np.isfinite(A['dv'])
    if m.sum() < 50:
        return float('nan')
    return float(np.corrcoef(A['E'][m], A['dv'][m])[0, 1])


def day_aggregate(A):
    E, qc, pord = A['E'], A['qc'], A['pord']
    days = np.unique(pord[qc]) if qc.any() else np.zeros(0, int)
    Ed = np.full((len(days), E.shape[1]), np.nan)
    Nd = np.zeros((len(days), E.shape[1]))
    npass = np.zeros(len(days), int)
    offd = np.full(len(days), np.nan)
    for k, o in enumerate(days):
        rows = qc & (pord == o)
        npass[k] = int(rows.sum())
        Ed[k] = nanmed(E[rows], axis=0)
        Nd[k] = np.sum(np.isfinite(E[rows]), axis=0)
        offd[k] = nanmed(A['offset'][rows])
    DM = A['DM']
    return {'ord': days.astype(int), 'E': Ed, 'N': Nd, 'n_passages': npass, 'offset': offd,
            'chainage': DM.chainage, 'corridor': DM.corridor, 'bins': DM.bins,
            'qc_pass_fraction': float(qc.mean()) if len(qc) else float('nan')}


def causal_break_days(days, values, thr, ref_n, confirm=2):
    """Causal instrument-regime breaks from daily passage levels.

    A day is a candidate when its level differs from the median of the last ref_n accepted levels by more
    than thr. The break is confirmed when the next confirm - 1 valid days depart in the same direction; an
    isolated aberrant day is ignored. Returns (first day of the new regime, confirmation day) pairs; only the
    confirmation day is known in real time, so filters reset there.
    """
    out = []
    hist = []
    pending = []
    for o, v in zip(days, values):
        if not np.isfinite(v):
            continue
        if hist:
            dev = float(v) - float(np.median(hist[-ref_n:]))
            if abs(dev) > thr:
                if pending and np.sign(dev) != np.sign(pending[0][2]):
                    pending = []
                pending.append((int(o), float(v), dev))
                if len(pending) >= max(1, int(confirm)):
                    out.append((pending[0][0], int(o)))
                    hist = [p[1] for p in pending]
                    pending = []
                continue
            pending = []
        hist.append(float(v))
    return out


def attach_breaks(series, feature, FC):
    """Causal regime breaks from passage levels pooled over both directions."""
    thr = FC['regime_break_log'] * quantity_scale(feature)
    pooled = defaultdict(list)
    for D in series.values():
        for o, v in zip(D['ord'].tolist(), D['offset'].tolist()):
            if np.isfinite(v):
                pooled[o].append(v)
    days = sorted(pooled)
    vals = [float(np.median(pooled[o])) for o in days]
    pairs = causal_break_days(days, vals, thr, FC['regime_reference_days'], FC.get('regime_confirm_days', 2))
    for D in series.values():
        td = D['ord']
        br = np.zeros(len(td), bool)
        for _, conf in pairs:
            k = int(np.searchsorted(td, conf, side='left'))
            if k < len(td):
                br[k] = True
        D['breaks'] = br
        D['break_days'] = [a for a, _ in pairs]
        D['break_confirm_days'] = [b for _, b in pairs]
    return pairs


def add_break_hazard(fits, series_list, FC):
    """Expected extra state variance per day of the regime-aware model from the rate of confirmed breaks.

    Forecasts more than one day ahead must allow for resets that may happen before the target day.
    """
    days = [int(o) for D in series_list for o in D['ord'].tolist()]
    if 'local_level_regime' not in fits or not days:
        return fits
    lo, hi = min(days), max(days)
    conf = set()
    for D in series_list:
        conf |= {int(b) for b in D.get('break_confirm_days', []) if lo < int(b) <= hi}
    rate = len(conf) / max(hi - lo, 1)
    fits['local_level_regime']['break_rate_per_day'] = float(rate)
    fits['local_level_regime']['extra_q'] = float(rate * FC['regime_reset_variance'])
    return fits


def extra_q(fits, m):
    return float(fits[m].get('extra_q', 0.0) or 0.0)


def build_series(table, model, FC):
    series = {d: day_aggregate(apply_indicator_model(model, table, d)) for d in DIRECTIONS}
    attach_breaks(series, model['feature'], FC)
    return series


# ----------------------------------------------------------------------------
# State-space models with irregular observation days
# ----------------------------------------------------------------------------
def kalman_local_level(E, N, td, params, clip=3.0, P0=1.0, store=False, breaks=None, reset_var=0.0):
    q, rd, rp = [float(v) for v in params]
    T, B = E.shape
    x = np.zeros(B)
    P = np.full(B, P0)
    last = np.full(B, np.nan)
    seen = np.zeros(B, bool)
    ll, nll = 0.0, 0
    if store:
        X = np.full((T, B), np.nan)
        PP = np.full((T, B), np.nan)
        LAST = np.full((T, B), np.nan)
    for t in range(T):
        if breaks is not None and breaks[t]:
            P = P + reset_var
        y = E[t]
        obs = np.isfinite(y)
        dt = np.where(seen, td[t] - np.where(seen, last, 0.0), 0.0)
        Pp = P + q * dt
        R = rd + rp / np.maximum(N[t], 1.0)
        S = Pp + R
        v = np.where(obs, y - x, 0.0)
        use = obs & seen
        if use.any():
            ll += float(-0.5 * np.sum(np.log(2 * np.pi * S[use]) + v[use] ** 2 / S[use]))
            nll += int(use.sum())
        sq = np.sqrt(S)
        vc = np.clip(v / sq, -clip, clip) * sq if clip else v
        K = Pp / S
        x = np.where(obs, x + K * vc, x)
        P = np.where(obs, (1 - K) * Pp, P)
        last = np.where(obs, td[t], last)
        seen = seen | obs
        if store:
            X[t], PP[t], LAST[t] = x, P, last
    return ll, nll, ((X, PP, LAST) if store else None)


def kalman_local_trend(E, N, td, params, clip=3.0, P0=1.0, Ps0=1e-5, store=False):
    q, qs, rd, rp = [float(v) for v in params]
    T, B = E.shape
    lv = np.zeros(B)
    sl = np.zeros(B)
    P11 = np.full(B, P0)
    P12 = np.zeros(B)
    P22 = np.full(B, Ps0)
    last = np.full(B, np.nan)
    seen = np.zeros(B, bool)
    ll, nll = 0.0, 0
    if store:
        Ls = np.full((T, B), np.nan)
        Ss = np.full((T, B), np.nan)
        A11 = np.full((T, B), np.nan)
        A12 = np.full((T, B), np.nan)
        A22 = np.full((T, B), np.nan)
        LAST = np.full((T, B), np.nan)
    for t in range(T):
        y = E[t]
        obs = np.isfinite(y)
        dt = np.where(seen, td[t] - np.where(seen, last, 0.0), 0.0)
        lp = lv + dt * sl
        p11 = P11 + 2 * dt * P12 + dt * dt * P22 + q * dt + qs * dt ** 3 / 3.0
        p12 = P12 + dt * P22 + qs * dt ** 2 / 2.0
        p22 = P22 + qs * dt
        R = rd + rp / np.maximum(N[t], 1.0)
        S = p11 + R
        v = np.where(obs, y - lp, 0.0)
        use = obs & seen
        if use.any():
            ll += float(-0.5 * np.sum(np.log(2 * np.pi * S[use]) + v[use] ** 2 / S[use]))
            nll += int(use.sum())
        sq = np.sqrt(S)
        vc = np.clip(v / sq, -clip, clip) * sq if clip else v
        K1 = p11 / S
        K2 = p12 / S
        lv = np.where(obs, lp + K1 * vc, lv)
        sl = np.where(obs, sl + K2 * vc, sl)
        n11 = (1 - K1) * p11
        n12 = (1 - K1) * p12
        n22 = p22 - K2 * p12
        P11 = np.where(obs, n11, P11)
        P12 = np.where(obs, n12, P12)
        P22 = np.where(obs, n22, P22)
        last = np.where(obs, td[t], last)
        seen = seen | obs
        if store:
            Ls[t], Ss[t], A11[t], A12[t], A22[t], LAST[t] = lv, sl, P11, P12, P22, last
    return ll, nll, ((Ls, Ss, A11, A12, A22, LAST) if store else None)


def kalman_mean_reverting(E, N, td, params, clip=3.0, store=False):
    """Level that reverts to the fingerprint (Ornstein-Uhlenbeck) with time constant tau days.

    params: tau (days), stationary variance s2 of the level, day noise variance, passage noise variance.
    """
    tau, s2, rd, rp = [float(v) for v in params]
    T, B = E.shape
    x = np.zeros(B)
    P = np.full(B, s2)
    last = np.full(B, np.nan)
    seen = np.zeros(B, bool)
    ll, nll = 0.0, 0
    if store:
        X = np.full((T, B), np.nan)
        PP = np.full((T, B), np.nan)
        LAST = np.full((T, B), np.nan)
    for t in range(T):
        y = E[t]
        obs = np.isfinite(y)
        dt = np.where(seen, td[t] - np.where(seen, last, 0.0), 0.0)
        a = np.exp(-dt / tau)
        xp = a * x
        Pp = a * a * P + s2 * (1.0 - a * a)
        R = rd + rp / np.maximum(N[t], 1.0)
        S = Pp + R
        v = np.where(obs, y - xp, 0.0)
        use = obs & seen
        if use.any():
            ll += float(-0.5 * np.sum(np.log(2 * np.pi * S[use]) + v[use] ** 2 / S[use]))
            nll += int(use.sum())
        sq = np.sqrt(S)
        vc = np.clip(v / sq, -clip, clip) * sq if clip else v
        K = Pp / S
        x = np.where(obs, xp + K * vc, x)
        P = np.where(obs, (1 - K) * Pp, P)
        last = np.where(obs, td[t], last)
        seen = seen | obs
        if store:
            X[t], PP[t], LAST[t] = x, P, last
    return ll, nll, ((X, PP, LAST) if store else None)


def run_kalman(kind, D, params, FC, store=False, E=None, N=None, td=None, breaks=None):
    """kind: 'll' (local level), 'ou' (mean-reverting level), 'llr' (regime-aware local level) or 'llt' (local linear trend)."""
    E = D['E'] if E is None else E
    N = D['N'] if N is None else N
    td = D['ord'] if td is None else td
    clip = FC['innovation_clip']
    P0 = FC['initial_variance']
    if kind == 'll':
        return kalman_local_level(E, N, td, params, clip, P0, store)
    if kind == 'ou':
        return kalman_mean_reverting(E, N, td, params, clip, store)
    if kind == 'llr':
        br = D.get('breaks') if breaks is None else breaks
        return kalman_local_level(E, N, td, params, clip, P0, store, breaks=br,
                                  reset_var=FC['regime_reset_variance'])
    ps0 = (FC['slope_prior_sd_per_year'] / 365.25) ** 2
    return kalman_local_trend(E, N, td, params, clip, P0, ps0, store)


def series_slice(D, mask):
    out = dict(D)
    for k in ('E', 'N'):
        out[k] = D[k][mask]
    out['ord'] = D['ord'][mask]
    out['breaks'] = D['breaks'][mask] if 'breaks' in D else np.zeros(int(mask.sum()), bool)
    return out


def kalman_param_bounds(kind):
    """Bounds on the natural logarithm of each parameter."""
    if kind == 'ou':
        return np.array([[0.0, 10.0], [-20.0, 3.0], [-40.0, 3.0], [-40.0, 3.0]])  # tau 1 day to 60 years
    if kind == 'llt':
        return np.array([[-40.0, 3.0]] * 4)
    return np.array([[-40.0, 3.0]] * 3)


def fit_kalman(series_list, kind, FC):
    """Maximum-likelihood variances pooled over directions and bins."""
    if kind in ('ll', 'llr'):
        starts = [np.log([1e-5, 0.01, 0.02]), np.log([1e-3, 0.003, 0.05]), np.log([1e-7, 0.02, 0.005])]
    elif kind == 'ou':
        starts = [np.log([30.0, 0.02, 0.01, 0.02]), np.log([180.0, 0.1, 0.005, 0.05]), np.log([1000.0, 0.05, 0.02, 0.01])]
    else:
        starts = [np.log([1e-5, 1e-9, 0.01, 0.02]), np.log([1e-4, 1e-7, 0.005, 0.03]), np.log([1e-6, 1e-10, 0.02, 0.01])]
    bounds = kalman_param_bounds(kind)

    def unpack(theta):
        return np.exp(np.clip(theta, bounds[:, 0], bounds[:, 1]))

    def nll(theta):
        p = unpack(theta)
        tot = 0.0
        for D in series_list:
            if D['E'].shape[0] == 0:
                continue
            tot += run_kalman(kind, D, p, FC)[0]
        return -tot if np.isfinite(tot) else 1e12

    best = None
    for st in starts:
        res = optimize.minimize(nll, st, method='Nelder-Mead',
                                options={'maxiter': 4000, 'maxfev': 8000, 'xatol': 1e-4, 'fatol': 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    p = unpack(best.x)
    n_obs = sum(run_kalman(kind, D, p, FC)[1] for D in series_list if D['E'].shape[0])
    return {'params': p.tolist(), 'loglik': float(-best.fun), 'n_obs': int(n_obs), 'converged': bool(best.success)}


def fit_all_kalman(series_list, FC):
    return OrderedDict((m, fit_kalman(series_list, KALMAN_KIND[m], FC)) for m in KALMAN_MODELS)


def running_baselines(E, td, window_days):
    """Last value, expanding mean and moving mean (window in days) up to each day."""
    T, B = E.shape
    LV = np.full((T, B), np.nan)
    MEAN = np.full((T, B), np.nan)
    MOV = np.full((T, B), np.nan)
    last = np.full(B, np.nan)
    s = np.zeros(B)
    n = np.zeros(B)
    for t in range(T):
        o = np.isfinite(E[t])
        last = np.where(o, E[t], last)
        s = s + np.where(o, E[t], 0.0)
        n = n + o
        LV[t] = last
        MEAN[t] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        k0 = int(np.searchsorted(td, td[t] - window_days, side='right'))
        mm = nanmean(E[k0:t + 1], axis=0)
        MOV[t] = np.where(np.isfinite(mm), mm, last)
    return LV, MEAN, MOV


def state_forecast(model, state, o, t_ord, Nt, params, extra_q=0.0):
    """Predictive mean and variance of the daily anomaly on day(s) t_ord from the filtered state at index o.

    t_ord may be a scalar (Nt has one value per bin) or a vector of m days (Nt has shape m by bins).
    extra_q adds level variance per day (the break hazard of the regime-aware model).
    """
    scalar = np.ndim(t_ord) == 0
    tt = np.atleast_1d(np.asarray(t_ord, float))[:, None]
    Nt = np.asarray(Nt, float)
    Nt = Nt[None, :] if Nt.ndim == 1 else Nt
    if model == 'local_trend':
        L, S, A11, A12, A22, LAST = state
        q, qs, rd, rp = params
        dt = tt - LAST[o][None, :]
        mu = L[o][None, :] + dt * S[o][None, :]
        var = (A11[o][None, :] + 2 * dt * A12[o][None, :] + dt * dt * A22[o][None, :] + q * dt + qs * dt ** 3 / 3.0
               + rd + rp / np.maximum(Nt, 1.0))
    elif model == 'mean_reverting':
        X, PP, LAST = state
        tau, s2, rd, rp = params
        dt = tt - LAST[o][None, :]
        a = np.exp(-dt / tau)
        mu = a * X[o][None, :]
        var = a * a * PP[o][None, :] + s2 * (1.0 - a * a) + rd + rp / np.maximum(Nt, 1.0)
    else:
        X, PP, LAST = state
        q, rd, rp = params
        dt = tt - LAST[o][None, :]
        mu = np.broadcast_to(X[o][None, :], dt.shape).copy()
        var = PP[o][None, :] + (q + extra_q) * dt + rd + rp / np.maximum(Nt, 1.0)
    if scalar:
        return mu[0], var[0]
    return mu, var


def compute_states(D, fits, FC):
    D['states'] = OrderedDict()
    for m in KALMAN_MODELS:
        D['states'][m] = run_kalman(KALMAN_KIND[m], D, fits[m]['params'], FC, store=True)[2]
    return D['states']


def horizon_label(h):
    if h == 'next':
        return 'next observation day'
    if isinstance(h, str):
        return h
    return f'at least {h} days ahead'


def collect_forecasts(series, fits, target_fn, FC, origin_fn=None, episode_of=None):
    """Forecast records for all models.

    target_fn(D) gives target day indices; origin_fn(D, t, h) gives the origin
    index (default: rolling origin for each horizon). Records are clustered by
    calendar episode: episode_of maps a day to its episode label; by default the
    episodes are formed from the target days of both directions together.
    """
    recs = {}
    ranks = {}
    targets = {d: (target_fn(D) if D['E'].shape[0] else np.zeros(0, int)) for d, D in series.items()}
    if episode_of is None:
        episode_of = calendar_episodes([int(series[d]['ord'][t]) for d in series for t in targets[d]],
                                       FC['episode_gap_days'])
    for d, D in series.items():
        E, N, td = D['E'], D['N'], D['ord']
        if E.shape[0] == 0:
            continue
        tgt = targets[d]
        if len(tgt) == 0:
            continue
        states = compute_states(D, fits, FC)
        LV, MEAN, MOV = running_baselines(E, td, FC['moving_mean_days'])
        for k, t in enumerate(tgt):
            y = E[t]
            obs = np.isfinite(y)
            for h in FC['horizons']:
                if origin_fn is not None:
                    o = origin_fn(D, t, h)
                elif h == 'next':
                    o = t - 1
                else:
                    o = int(np.searchsorted(td, td[t] - h, side='right') - 1)
                if o is None or o < 0 or o >= t:
                    continue
                v = obs & np.isfinite(states['local_level'][2][o]) & np.isfinite(LV[o])
                if not v.any():
                    continue
                preds = {'fingerprint': np.zeros(len(y)), 'persistence': LV[o], 'historical_mean': MEAN[o],
                         'moving_mean': MOV[o]}
                variances = {}
                for m in KALMAN_MODELS:
                    preds[m], variances[m] = state_forecast(m, states[m], o, td[t], N[t], fits[m]['params'],
                                                            extra_q(fits, m))
                hl = horizon_label(h)
                R = recs.setdefault(hl, defaultdict(list))
                n = int(v.sum())
                cid = episode_of[int(td[t])]
                R['y'].append(y[v])
                R['cluster'].append(np.full(n, cid, dtype=object))
                R['day'].append(np.full(n, td[t]))
                R['lead'].append(np.full(n, td[t] - td[o]))
                for m in MODELS:
                    R[m].append(preds[m][v])
                for m in KALMAN_MODELS:
                    R['var_' + m].append(variances[m][v])
                if n >= FC['min_bins_ranking']:
                    K = ranks.setdefault(hl, defaultdict(list))
                    K['cluster'].append(cid)
                    k_top = max(1, int(math.ceil(FC['top_fraction'] * n)))
                    yv = y[v]
                    top_true = set(np.argsort(-yv, kind='stable')[:k_top].tolist())
                    for m in MODELS[1:]:
                        pv = preds[m][v]
                        K[m + '_rho'].append(spearman(pv, yv))
                        top_pred = set(np.argsort(-pv, kind='stable')[:k_top].tolist())
                        K[m + '_top'].append(len(top_true & top_pred) / k_top)
    out = {hl: {k: np.concatenate(v) for k, v in R.items()} for hl, R in recs.items()}
    rk = {hl: {k: np.array(v) for k, v in K.items()} for hl, K in ranks.items()}
    return out, rk


def cluster_bootstrap_skill(abs_m, abs_b, clusters, B, rng):
    keys, inv = np.unique(clusters.astype(str), return_inverse=True)
    C = len(keys)
    sm = np.bincount(inv, weights=abs_m, minlength=C)
    sb = np.bincount(inv, weights=abs_b, minlength=C)
    point = 1.0 - sm.sum() / sb.sum() if sb.sum() > 0 else float('nan')
    if C < 2:
        return float(point), float('nan'), float('nan'), C
    vals = np.empty(B)
    for b in range(B):
        ii = rng.integers(0, C, C)
        den = sb[ii].sum()
        vals[b] = 1.0 - sm[ii].sum() / den if den > 0 else np.nan
    lo, hi = nanpct(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), C


def cluster_bootstrap_mean(values, clusters, B, rng):
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values, clusters = values[ok], np.asarray(clusters)[ok]
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan'), 0
    keys, inv = np.unique(clusters.astype(str), return_inverse=True)
    C = len(keys)
    s = np.bincount(inv, weights=values, minlength=C)
    n = np.bincount(inv, minlength=C).astype(float)
    point = float(values.mean())
    if C < 2:
        return point, float('nan'), float('nan'), C
    vals = np.empty(B)
    for b in range(B):
        ii = rng.integers(0, C, C)
        vals[b] = s[ii].sum() / n[ii].sum()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), C


def episode_skills(abs_m, abs_b, clusters):
    """Skill (one minus the ratio of mean absolute errors) within each episode."""
    keys, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    sm = np.bincount(inv, weights=abs_m, minlength=len(keys))
    sb = np.bincount(inv, weights=abs_b, minlength=len(keys))
    with np.errstate(invalid='ignore', divide='ignore'):
        sk = np.where(sb > 0, 1.0 - sm / sb, np.nan)
    return keys, sk


def sign_summary(values):
    """Number of positive values, number of finite values and the exact one-sided sign-test p-value."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    k = int(np.sum(v > 0))
    return k, int(len(v)), sign_test_p(k, len(v))


def forecast_metric_rows(ind_label, recs, ranks, FC, rng, scope):
    rows, rrows = [], []
    for hl, R in recs.items():
        y = R['y']
        cl = R['cluster']
        n = len(y)
        sd_y = float(np.std(y))
        abs_err = {m: np.abs(y - R[m]) for m in MODELS}
        for m in MODELS:
            mae = float(abs_err[m].mean())
            rmse = float(np.sqrt(np.mean((y - R[m]) ** 2)))
            sp, slo, shi, C = cluster_bootstrap_skill(abs_err[m], abs_err['persistence'], cl, FC['bootstrap'], rng)
            sf, flo, fhi, _ = cluster_bootstrap_skill(abs_err[m], abs_err['fingerprint'], cl, FC['bootstrap'], rng)
            kf, nf, pf = sign_summary(episode_skills(abs_err[m], abs_err['fingerprint'], cl)[1])
            kp, np_, pp = sign_summary(episode_skills(abs_err[m], abs_err['persistence'], cl)[1])
            crps = cov80 = cov95 = float('nan')
            if m in KALMAN_MODELS:
                sd = np.sqrt(np.maximum(R['var_' + m], 1e-12))
                crps = float(np.mean(gauss_crps(R[m], sd, y)))
                z = np.abs(y - R[m]) / sd
                cov80 = float(np.mean(z <= stats.norm.ppf(0.9)))
                cov95 = float(np.mean(z <= stats.norm.ppf(0.975)))
            rows.append([scope, ind_label, hl, m, n, C, float(np.median(R['lead'])), sd_y, mae, rmse,
                         sp, slo, shi, sf, flo, fhi, crps, cov80, cov95, kf, nf, pf, kp, np_, pp])
        K = ranks.get(hl)
        if K is not None and len(K.get('cluster', [])):
            for m in MODELS[1:]:
                r_mean, r_lo, r_hi, C = cluster_bootstrap_mean(K[m + '_rho'], K['cluster'], FC['bootstrap'], rng)
                t_mean, t_lo, t_hi, _ = cluster_bootstrap_mean(K[m + '_top'], K['cluster'], FC['bootstrap'], rng)
                ek, em = episode_means(K[m + '_rho'], K['cluster'])
                k, ne, p = sign_summary(em)
                rrows.append([scope, ind_label, hl, m, int(np.isfinite(K[m + '_rho']).sum()), C,
                              r_mean, r_lo, r_hi, t_mean, t_lo, t_hi, k, ne, p])
    return rows, rrows


def episode_means(values, clusters):
    values = np.asarray(values, float)
    clusters = np.asarray(clusters).astype(str)
    ok = np.isfinite(values)
    keys = np.unique(clusters[ok])
    return keys, np.array([float(values[ok & (clusters == k)].mean()) for k in keys])


def episode_rows(ind_label, recs, ranks, scope, models=('fingerprint', 'persistence', 'moving_mean', 'local_level',
                                                        'mean_reverting')):
    """Per-episode errors, skills, interval coverage and ranking for one indicator."""
    rows = []
    for hl, R in recs.items():
        cl = np.asarray(R['cluster']).astype(str)
        y = R['y']
        K = ranks.get(hl, {})
        kcl = np.asarray(K.get('cluster', []), dtype=object).astype(str) if K else np.zeros(0, str)
        for e in np.unique(cl):
            m_e = cl == e
            ae_f = np.abs(y[m_e] - R['fingerprint'][m_e])
            ae_p = np.abs(y[m_e] - R['persistence'][m_e])
            days = np.unique(R['day'][m_e])
            for m in models:
                ae = np.abs(y[m_e] - R[m][m_e])
                cov = float('nan')
                if m in KALMAN_MODELS:
                    sd = np.sqrt(np.maximum(R['var_' + m][m_e], 1e-12))
                    cov = float(np.mean(np.abs(y[m_e] - R[m][m_e]) / sd <= stats.norm.ppf(0.9)))
                rho = float('nan')
                if len(kcl) and (m + '_rho') in K:
                    vv = np.asarray(K[m + '_rho'], float)[kcl == e]
                    vv = vv[np.isfinite(vv)]
                    rho = float(vv.mean()) if len(vv) else float('nan')
                rows.append([scope, ind_label, hl, m, e, o2d(days.min()), o2d(days.max()), len(days), int(m_e.sum()),
                             float(ae.mean()), 1.0 - ae.sum() / ae_f.sum() if ae_f.sum() > 0 else float('nan'),
                             1.0 - ae.sum() / ae_p.sum() if ae_p.sum() > 0 else float('nan'), cov, rho])
    return rows


FORECAST_HEADER = ['scope', 'indicator', 'horizon', 'model', 'n_bin_days', 'n_episodes', 'median_lead_days',
                   'sd_of_target', 'mae', 'rmse', 'skill_vs_persistence', 'skill_vs_persistence_lo95',
                   'skill_vs_persistence_hi95', 'skill_vs_fingerprint', 'skill_vs_fingerprint_lo95',
                   'skill_vs_fingerprint_hi95', 'crps', 'coverage80', 'coverage95',
                   'episodes_better_than_fingerprint', 'episodes_scored', 'sign_test_p_vs_fingerprint',
                   'episodes_better_than_persistence', 'episodes_scored_persistence', 'sign_test_p_vs_persistence']
RANK_HEADER = ['scope', 'indicator', 'horizon', 'model', 'n_days', 'n_episodes', 'mean_spearman',
               'spearman_lo95', 'spearman_hi95', 'top_decile_precision', 'top_lo95', 'top_hi95',
               'episodes_positive', 'episodes_scored', 'sign_test_p']
EPISODE_HEADER = ['scope', 'indicator', 'horizon', 'model', 'episode', 'first_day', 'last_day', 'n_days', 'n_bin_days',
                  'mae', 'skill_vs_fingerprint', 'skill_vs_persistence', 'coverage80', 'mean_spearman']


# ----------------------------------------------------------------------------
# Change-point segmentation for instrument level regimes
# ----------------------------------------------------------------------------
def segment_mean_changes(y, min_size=3, pen_mult=3.0):
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return [], float('nan'), float('nan')
    if n < 2 * min_size:
        return [(0, n)], float('nan'), float('nan')
    dif = np.diff(y)
    sigma = 1.4826 * np.median(np.abs(dif - np.median(dif))) / math.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = max(float(np.std(y)), 1e-6)
    pen = pen_mult * sigma ** 2 * math.log(n)
    s1 = np.concatenate([[0.0], np.cumsum(y)])
    s2 = np.concatenate([[0.0], np.cumsum(y * y)])
    best = np.full(n + 1, np.inf)
    best[0] = -pen
    arg = np.zeros(n + 1, int)
    for t in range(min_size, n + 1):
        s = np.arange(0, t - min_size + 1)
        s = s[np.isfinite(best[s])]
        if len(s) == 0:
            continue
        m = t - s
        cost = (s2[t] - s2[s]) - (s1[t] - s1[s]) ** 2 / m
        tot = best[s] + cost + pen
        k = int(np.argmin(tot))
        best[t] = tot[k]
        arg[t] = s[k]
    segs = []
    t = n
    while t > 0:
        s = int(arg[t])
        segs.append((s, t))
        t = s
    return segs[::-1], float(sigma), float(pen)


def merge_small_shifts(y, segs, thr):
    """Merge neighbouring segments whose medians differ by less than thr."""
    segs = list(segs)
    while len(segs) > 1:
        meds = [float(np.median(y[a:b])) for a, b in segs]
        diffs = [abs(meds[k + 1] - meds[k]) for k in range(len(segs) - 1)]
        k = int(np.argmin(diffs))
        if diffs[k] >= thr:
            break
        segs[k] = (segs[k][0], segs[k + 1][1])
        del segs[k + 1]
    return segs


# ----------------------------------------------------------------------------
# Module 1: data audit
# ----------------------------------------------------------------------------
def run_audit(found, outdir, cfg, S):
    log('Module 1: data audit')
    table = found.table
    qc = cfg['qc']
    RG = cfg['regimes']
    gps = {}
    for p, v in zip(table.s['passage_id'].tolist(), table.x['gps_metadata_start_difference_s'].tolist()):
        gps.setdefault(p, v)
    rows = []
    levels = {c: [] for c in CHANNELS}
    for c in CHANNELS:
        col = f's{c}_rms_demeaned'
        for d in DIRECTIONS:
            DM = DirMatrix(table, d)
            Y = DM.logmatrix(col)
            Fp, _ = fit_fingerprint(Y, np.ones(DM.P, bool), qc['min_fit_passages_per_bin'])
            rho, nb = profile_rho(Y, Fp, qc['min_bins_per_passage'])
            off = nanmed(Y - Fp[None, :], axis=1)
            R = Y - off[:, None] - Fp[None, :]
            sd = nanstd(R, axis=1)
            spd = nanmed(DM.matrix(table.x['estimated_speed_mps']), axis=1)
            for i, p in enumerate(DM.pids):
                ok = bool(np.isfinite(rho[i]) and rho[i] >= qc['min_profile_rho'] and nb[i] >= qc['min_bins_per_passage'])
                rows.append([p, DM.pdate[i], d, c, off[i], rho[i], sd[i], nb[i], spd[i], gps.get(p), ok])
                levels[c].append((int(DM.pord[i]), float(off[i]), ok, d))
    write_csv(outdir / '01_audit_passage_channel.csv',
              ['passage_id', 'local_date', 'direction', 'channel', 'level_log_rms', 'profile_spearman',
               'residual_spatial_sd', 'n_bins', 'median_speed_mps', 'gps_metadata_start_difference_s', 'qc_pass'],
              rows)
    seg_rows, shift_rows, dead_rows = [], [], []
    seg_lookup, boundaries, summ = {}, {}, {}
    for c in CHANNELS:
        arr = levels[c]
        ords = np.array([a[0] for a in arr], int)
        lev = np.array([a[1] for a in arr], float)
        okq = np.array([a[2] for a in arr], bool)
        # Dead or abnormal periods: most passages on a day fail the profile check.
        all_days = np.unique(ords)
        fail = np.array([1.0 - okq[ords == o].mean() for o in all_days])
        dead = fail >= RG['dead_fail_fraction']
        k = 0
        while k < len(all_days):
            if dead[k]:
                j = k
                while j + 1 < len(all_days) and dead[j + 1]:
                    j += 1
                m = (ords >= all_days[k]) & (ords <= all_days[j])
                dead_rows.append([c, o2d(all_days[k]), o2d(all_days[j]), j - k + 1, int(m.sum()),
                                  float(1 - okq[m].mean()), float(np.median(lev[m]))])
                k = j + 1
            else:
                k += 1
        # Working regimes: levels of quality-passing passages only.
        good_days = np.unique(ords[okq])
        yday = np.array([nanmed(lev[(ords == o) & okq]) for o in good_days])
        fin = np.isfinite(yday)
        good_days, yday = good_days[fin], yday[fin]
        segs, sigma, pen = segment_mean_changes(yday, RG['min_segment_days'], RG['penalty_multiplier'])
        segs = merge_small_shifts(yday, segs, RG['material_shift_log'])
        # A working regime needs most passages in its span to pass; sparse passes inside dead periods do not count.
        kept = []
        for a, b in segs:
            m = (ords >= good_days[a]) & (ords <= good_days[b - 1])
            if m.any() and okq[m].mean() >= 1.0 - RG['dead_fail_fraction']:
                kept.append(list(range(a, b)))
        while len(kept) > 1:
            meds = [float(np.median(yday[idx])) for idx in kept]
            diffs = [abs(meds[k + 1] - meds[k]) for k in range(len(kept) - 1)]
            k = int(np.argmin(diffs))
            if diffs[k] >= RG['material_shift_log']:
                break
            kept[k] = kept[k] + kept[k + 1]
            del kept[k + 1]
        info = []
        for kk, idx in enumerate(kept):
            a, b = idx[0], idx[-1] + 1
            m = (ords >= good_days[a]) & (ords <= good_days[b - 1])
            lvl = float(np.median(yday[idx]))
            info.append((int(good_days[a]), int(good_days[b - 1]), lvl))
            seg_rows.append([c, kk + 1, o2d(good_days[a]), o2d(good_days[b - 1]), len(idx), int((m & okq).sum()), lvl,
                             float(okq[m].mean()) if m.any() else float('nan')])
        seg_lookup[c] = info
        bl = []
        biggest = (0.0, None)
        for kk in range(1, len(info)):
            shift = info[kk][2] - info[kk - 1][2]
            shift_rows.append([c, kk, o2d(info[kk - 1][1]), o2d(info[kk][0]), shift, math.exp(shift)])
            bl.append((info[kk - 1][1], info[kk][0], shift))
            if abs(shift) > abs(biggest[0]):
                biggest = (shift, (o2d(info[kk - 1][1]), o2d(info[kk][0])))
        boundaries[c] = bl
        summ[c] = {'working_regimes': len(info), 'level_shifts': len(bl),
                   'largest_shift_log': biggest[0], 'largest_shift_factor': math.exp(abs(biggest[0])),
                   'largest_shift_between': biggest[1], 'qc_pass_fraction': float(okq.mean()),
                   'dead_periods': int(sum(1 for r in dead_rows if r[0] == c)),
                   'dead_period_days': int(sum(r[3] for r in dead_rows if r[0] == c)),
                   'noise_sigma_day': sigma}
    write_csv(outdir / '01_audit_regimes.csv',
              ['channel', 'regime', 'first_date', 'last_date', 'n_days', 'n_qc_passages', 'median_level_log',
               'qc_pass_fraction_in_span'], seg_rows)
    write_csv(outdir / '01_audit_level_shifts.csv',
              ['channel', 'shift_index', 'last_date_before', 'first_date_after', 'shift_log', 'factor'], shift_rows)
    write_csv(outdir / '01_audit_dead_periods.csv',
              ['channel', 'first_date', 'last_date', 'n_days', 'n_passages', 'fail_fraction', 'median_level_log'],
              dead_rows)
    # Level changes that happen at the same time in more than one channel point to a system-wide change.
    allb = sorted([(fa, lb, c, sh) for c, bl in boundaries.items() for lb, fa, sh in bl])
    groups = []
    for fa, lb, c, sh in allb:
        if groups and fa - groups[-1]['first_after'][-1] <= 7:
            g = groups[-1]
        else:
            g = {'first_after': [], 'last_before': [], 'channels': [], 'factors': []}
            groups.append(g)
        g['first_after'].append(fa)
        g['last_before'].append(lb)
        g['channels'].append(c)
        g['factors'].append(math.exp(sh))
    sim_rows = []
    for g in groups:
        sim_rows.append([o2d(max(g['last_before'])), o2d(min(g['first_after'])), len(set(g['channels'])),
                         ' '.join(f's{c}:{f:.2f}' for c, f in zip(g['channels'], g['factors']))])
    write_csv(outdir / '01_audit_simultaneous_changes.csv',
              ['last_day_before', 'first_day_after', 'n_channels', 'channel_factors'], sim_rows)
    # Fingerprint stability across working regimes.
    stab_rows = []
    for c in (1, 3, 4, 5):
        col = f's{c}_rms_demeaned'
        for d in DIRECTIONS:
            DM = DirMatrix(table, d)
            Y = DM.logmatrix(col)
            Fall, _ = fit_fingerprint(Y, np.ones(DM.P, bool), qc['min_fit_passages_per_bin'])
            rho, nb = profile_rho(Y, Fall, qc['min_bins_per_passage'])
            good = (rho >= qc['min_profile_rho']) & (nb >= qc['min_bins_per_passage'])
            prof = []
            for kk, (a, b, lvl) in enumerate(seg_lookup[c]):
                m = good & (DM.pord >= a) & (DM.pord <= b)
                if m.sum() >= RG['min_passages_fingerprint']:
                    Fk, _ = fit_fingerprint(Y, m, 3)
                    prof.append((kk + 1, Fk, int(m.sum())))
            for i in range(len(prof)):
                for j in range(i + 1, len(prof)):
                    ok = np.isfinite(prof[i][1]) & np.isfinite(prof[j][1])
                    stab_rows.append([c, d, prof[i][0], prof[j][0], prof[i][2], prof[j][2],
                                      spearman(prof[i][1][ok], prof[j][1][ok])])
    write_csv(outdir / '01_audit_fingerprint_stability.csv',
              ['channel', 'direction', 'regime_a', 'regime_b', 'passages_a', 'passages_b', 'spearman'], stab_rows)
    flag_rows = []
    seen = set()
    for p, d_, dr, v in zip(table.s['passage_id'].tolist(), table.s['local_date'].tolist(),
                            table.s['direction'].tolist(), table.x['gps_metadata_start_difference_s'].tolist()):
        if p in seen:
            continue
        seen.add(p)
        if v is not None and np.isfinite(v) and abs(v) > qc['timing_flag_seconds']:
            flag_rows.append([p, d_, dr, v])
    write_csv(outdir / '01_audit_timing_flags.csv',
              ['passage_id', 'local_date', 'direction', 'gps_metadata_start_difference_s'],
              sorted(flag_rows, key=lambda r: r[1]))
    stab = {}
    for c in (1, 3, 4, 5):
        v = [r[6] for r in stab_rows if r[0] == c and r[6] is not None and np.isfinite(r[6])]
        stab[c] = {'median': float(np.median(v)) if v else None, 'min': float(np.min(v)) if v else None, 'pairs': len(v)}
    S['audit'] = {'channels': summ, 'timing_flagged_passages': len(flag_rows), 'fingerprint_stability': stab,
                  'simultaneous_changes': [r for r in sim_rows if r[2] >= 3]}
    return {'levels': levels, 'segments': seg_lookup, 'boundaries': boundaries, 'dead': dead_rows}


def segment_of(seg_info, o):
    idx = 0
    for k, (a, b, _) in enumerate(seg_info):
        if o >= a:
            idx = k
    return idx


def boundary_in_span(boundaries, lo, hi):
    """True when a regime starts inside (lo, hi]."""
    return any(lo < first_after <= hi for _, first_after, _ in boundaries)


def channel_boundaries(audit, ch):
    """Working-regime changes of a channel; the derived cabin channel inherits those of channels 4 and 5."""
    if ch == 45:
        return list(audit['boundaries'].get(4, [])) + list(audit['boundaries'].get(5, []))
    return list(audit['boundaries'].get(ch, []))


# ----------------------------------------------------------------------------
# Module 2: classification re-analysis
# ----------------------------------------------------------------------------
def passage_level_features(table, direction):
    DM = DirMatrix(table, direction)
    names = [f's{c}_{f}' for c in CHANNELS for f in FEATURES]
    ABS = np.full((DM.P, len(names)), np.nan)
    REL = np.full((DM.P, len(names)), np.nan)
    cor = DM.corridor
    for k, col in enumerate(names):
        feat = col.split('_', 1)[1]
        v = table.x[col]
        if feat in LOG_FEATURES:
            v = safe_log(v)
        M = DM.matrix(v)
        a = nanmed(M[:, cor], axis=1)
        b = nanmed(M[:, ~cor], axis=1)
        ABS[:, k] = a
        REL[:, k] = a - b
    return DM, names, ABS, REL


def std_fit(X):
    med = nanmed(X, axis=0)
    Xf = np.where(np.isnan(X), med[None, :], X)
    mu = nanmean(Xf, axis=0)
    sd = nanstd(Xf, axis=0)
    keep = np.isfinite(med) & np.isfinite(sd) & (sd > 1e-12)
    return med, mu, sd, keep


def std_apply(X, prm):
    med, mu, sd, keep = prm
    Xf = np.where(np.isnan(X), med[None, :], X)
    Z = (Xf - mu[None, :]) / np.where(sd > 0, sd, 1.0)[None, :]
    return Z[:, keep]


def knn_predict(Ztr, ytr, Zte, k):
    d2 = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(d2, axis=1, kind='stable')[:, :k]
    pred = np.empty(len(Zte), int)
    for i in range(len(Zte)):
        labs = ytr[order[i]]
        vals, cnt = np.unique(labs, return_counts=True)
        top = vals[cnt == cnt.max()]
        if len(top) == 1:
            pred[i] = top[0]
        else:
            for j in order[i]:
                if ytr[j] in top:
                    pred[i] = ytr[j]
                    break
    return pred


def lda_predict(Ztr, ytr, Zte, lam):
    cls = np.unique(ytr)
    M = np.array([Ztr[ytr == c].mean(axis=0) for c in cls])
    Xc = Ztr - M[np.searchsorted(cls, ytr)]
    dof = max(len(ytr) - len(cls), 1)
    Sg = Xc.T @ Xc / dof
    Sig = (1 - lam) * Sg + lam * np.diag(np.diag(Sg)) + 1e-6 * np.eye(Sg.shape[0])
    W = np.linalg.solve(Sig, M.T)
    scores = Zte @ W - 0.5 * np.sum(M.T * W, axis=0)[None, :]
    return cls[np.argmax(scores, axis=1)]


def rf_predict(Ztr, ytr, Zte, trees, seed):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=trees, class_weight='balanced', random_state=seed, n_jobs=1)
    m.fit(Ztr, ytr)
    return m.predict(Zte)


def balanced_accuracy(y, p):
    cls = np.unique(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in cls]))


def folds_random(y, K, rng):
    f = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        f[idx] = (np.arange(len(idx)) + int(rng.integers(K))) % K
    return f


def folds_grouped(y, groups, K, rng):
    f = np.empty(len(y), int)
    for c in np.unique(y):
        g = rng.permutation(np.unique(groups[y == c]))
        start = int(rng.integers(K))
        gmap = {int(gg): (i + start) % K for i, gg in enumerate(g)}
        for i in np.where(y == c)[0]:
            f[i] = gmap[int(groups[i])]
    return f


def folds_blocked(y, groups, K):
    f = np.empty(len(y), int)
    for c in np.unique(y):
        g = np.unique(groups[y == c])
        gmap = {}
        for k, chunk in enumerate(np.array_split(g, K)):
            for gg in chunk:
                gmap[int(gg)] = k
        for i in np.where(y == c)[0]:
            f[i] = gmap[int(groups[i])]
    return f


def cv_scores(X, y, groups, scheme, clf, C, rng, repeats):
    K = C['folds']
    accs = []
    for _ in range(repeats):
        if scheme == 'random':
            f = folds_random(y, K, rng)
        elif scheme == 'grouped_by_date':
            f = folds_grouped(y, groups, K, rng)
        else:
            f = folds_blocked(y, groups, K)
        pred = np.full(len(y), -1)
        for k in range(K):
            te = f == k
            if not te.any():
                continue
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            prm = std_fit(X[tr])
            Ztr = std_apply(X[tr], prm)
            Zte = std_apply(X[te], prm)
            if Ztr.shape[1] == 0:
                continue
            if clf == 'knn':
                pred[te] = knn_predict(Ztr, y[tr], Zte, C['knn_k'])
            elif clf == 'lda':
                pred[te] = lda_predict(Ztr, y[tr], Zte, C['lda_shrinkage'])
            else:
                pred[te] = rf_predict(Ztr, y[tr], Zte, C['rf_trees'], int(rng.integers(1, 2 ** 31 - 1)))
        ok = pred >= 0
        if ok.any():
            accs.append(balanced_accuracy(y[ok], pred[ok]))
    return accs


def placebo_labels(pord, sizes, rng, variant='scattered', max_tries=500):
    """Contiguous date blocks with the real class sizes.

    scattered: blocks placed at random positions (no overlap);
    adjacent:  blocks placed back to back in the real class order from a random start.
    """
    days, counts = np.unique(pord, return_counts=True)
    nd = len(days)
    K = len(sizes)
    lookup_days = days.tolist()
    if variant == 'adjacent':
        starts = []
        for s in range(nd):
            e, ok = s, True
            spans = []
            for k in range(K):
                tot, b = 0, e
                while b < nd and tot < sizes[k]:
                    tot += counts[b]
                    b += 1
                if tot < sizes[k]:
                    ok = False
                    break
                spans.append((e, b))
                e = b
            if ok:
                starts.append(spans)
        if not starts:
            return None
        spans = starts[int(rng.integers(len(starts)))]
        lab_day = np.full(nd, -1)
        for k, (a, b) in enumerate(spans):
            lab_day[a:b] = k
        lookup = dict(zip(lookup_days, lab_day.tolist()))
        return np.array([lookup[int(o)] for o in pord], int)
    for _ in range(max_tries):
        order = rng.permutation(K)
        used = np.zeros(nd, bool)
        lab_day = np.full(nd, -1)
        ok = True
        for k in order:
            need = sizes[k]
            cands = []
            for s in range(nd):
                if used[s]:
                    continue
                tot, e = 0, s
                while e < nd and not used[e] and tot < need:
                    tot += counts[e]
                    e += 1
                if tot >= need:
                    cands.append((s, e))
            if not cands:
                ok = False
                break
            s, e = cands[int(rng.integers(len(cands)))]
            used[s:e] = True
            lab_day[s:e] = k
        if ok:
            lookup = dict(zip(lookup_days, lab_day.tolist()))
            return np.array([lookup[int(o)] for o in pord], int)
    return None


def shifted_labels(pord, y, shift):
    """The real label pattern moved by shift observation days, wrapping round the end of the record.

    Every day keeps one label, so class blocks keep their lengths in days and their arrangement (an isolated
    block, gaps, back-to-back blocks); only the dates change.
    """
    days = np.unique(pord)
    lab_day = np.full(len(days), -1)
    for i, o in enumerate(days.tolist()):
        v = y[pord == o]
        lab_day[i] = int(v[0]) if len(v) else -1
    rolled = np.roll(lab_day, int(shift))
    lookup = dict(zip(days.tolist(), rolled.tolist()))
    return np.array([lookup[int(o)] for o in pord], int)


def run_classification(found, outdir, cfg, S, audit, rng):
    log('Module 2: classification re-analysis')
    C = cfg['classification']
    have_rf = sklearn_available()
    clfs = ['knn', 'lda'] + (['rf'] if have_rf else [])
    if not have_rf:
        log('scikit-learn not available: random forest omitted, kNN and shrinkage LDA retained.')
    res_rows, pl_rows, ct_rows, size_rows = [], [], [], []
    summary = {}
    for d in C['directions']:
        DM, names, ABS, REL = passage_level_features(found.table, d)
        y = np.full(DM.P, -1)
        for k, (name, a, b) in enumerate(C['classes']):
            y[(DM.pord >= d2o(a)) & (DM.pord < d2o(b))] = k
        counts = {k: int((y == k).sum()) for k in range(len(C['classes']))}
        keep_cls = [k for k, n in counts.items() if n >= C['min_class_passages']]
        for k, (name, a, b) in enumerate(C['classes']):
            ndays = len(np.unique(DM.pord[y == k])) if counts[k] else 0
            size_rows.append([d, name, a, b, counts[k], ndays, k in keep_cls])
        if len(keep_cls) < 2:
            log(f'  {d}: fewer than two classes with enough passages; classification skipped.')
            continue
        use = np.isin(y, keep_cls)
        remap = {k: i for i, k in enumerate(keep_cls)}
        yu = np.array([remap[v] for v in y[use]], int)
        gu = DM.pord[use]
        sets = OrderedDict([
            ('absolute_110', ABS),
            ('relative_110', REL),
            ('single_s1_rms', ABS[:, [names.index('s1_rms_demeaned')]]),
        ])
        real = {}
        real_sd = {}
        for set_name, X in sets.items():
            Xu = X[use]
            for clf in clfs:
                for scheme in ('random', 'grouped_by_date', 'blocked_in_time'):
                    reps = 1 if scheme == 'blocked_in_time' else (C['rf_repeats'] if clf == 'rf' else C['repeats'])
                    accs = cv_scores(Xu, yu, gu, scheme, clf, C, rng, reps)
                    mean = float(np.mean(accs)) if accs else float('nan')
                    sd = float(np.std(accs)) if len(accs) > 1 else float('nan')
                    res_rows.append([d, set_name, clf, scheme, mean, sd, len(accs), int(use.sum()),
                                     len(keep_cls), 1.0 / len(keep_cls)])
                    real[(set_name, clf, scheme)] = mean
                    if scheme == 'random':
                        real_sd[(set_name, clf)] = sd
        sizes = [int((yu == i).sum()) for i in range(len(keep_cls))]
        y_used = np.full(DM.P, -1)
        y_used[use] = yu
        n_days = len(np.unique(DM.pord))
        pl = defaultdict(list)
        for variant in C['placebo_variants']:
            if variant == 'shifted':
                # Every non-zero shift once (deterministic), spread evenly when there are more shifts than draws.
                shifts = np.unique(np.linspace(1, n_days - 1, min(n_days - 1, C['placebo_draws'])).round().astype(int))
            else:
                shifts = [None] * C['placebo_draws']
            n_rf = min(len(shifts), C['rf_placebo_draws'])
            rf_pick = set(np.linspace(0, len(shifts) - 1, n_rf).round().astype(int).tolist()) if n_rf > 0 else set()
            for draw, shift in enumerate(shifts):
                if variant == 'shifted':
                    lab = shifted_labels(DM.pord, y_used, shift)
                    cnt = [int((lab == k).sum()) for k in range(len(sizes))]
                    if min(cnt) < C['min_class_passages']:
                        continue
                else:
                    lab = placebo_labels(DM.pord, sizes, rng, variant)
                if lab is None:
                    continue
                m = lab >= 0
                blocks = []
                for k in range(len(sizes)):
                    oo = DM.pord[lab == k]
                    blocks.append(f'{o2d(oo.min())} to {o2d(oo.max())}' if len(oo) else 'empty')
                if variant == 'shifted':
                    blocks.append(f'shift {int(shift)} observation days')
                for set_name in ('absolute_110', 'relative_110'):
                    cl = ['knn', 'lda']
                    if have_rf and draw in rf_pick and set_name == 'absolute_110':
                        cl.append('rf')
                    for clf in cl:
                        reps = 1 if clf == 'rf' else C['placebo_repeats']
                        accs = cv_scores(sets[set_name][m], lab[m], DM.pord[m], 'random', clf, C, rng, reps)
                        if accs:
                            v = float(np.mean(accs))
                            pl[(variant, set_name, clf)].append(v)
                            pl_rows.append([d, variant, draw, set_name, clf, v, '; '.join(blocks)])
        dsum = {'class_sizes': sizes, 'classes_used': [C['classes'][k][0] for k in keep_cls], 'chance': 1.0 / len(keep_cls)}
        for (variant, set_name, clf), vals in pl.items():
            vals = np.array(vals)
            r = real.get((set_name, clf, 'random'), float('nan'))
            reps = 1 if clf == 'rf' else C['placebo_repeats']
            dsum[f'{variant}|{set_name}|{clf}'] = {
                'random_cv': r,
                'grouped_cv': real.get((set_name, clf, 'grouped_by_date')),
                'blocked_cv': real.get((set_name, clf, 'blocked_in_time')),
                'placebo_median': float(np.median(vals)), 'placebo_p05': float(np.percentile(vals, 5)),
                'placebo_p95': float(np.percentile(vals, 95)), 'placebo_n': int(len(vals)),
                'placebo_sd': float(np.std(vals)),
                # Placebo values average fewer cross-validation repeats than the real labels; this is the extra
                # spread that alone causes, to compare with placebo_sd.
                'repeat_noise_sd': (real_sd.get((set_name, clf), float('nan')) / math.sqrt(reps)),
                'real_percentile_in_placebo': float(np.mean(vals <= r)) if np.isfinite(r) else None,
            }
        dsum['real'] = {f'{s}|{c}|{v}': real[(s, c, v)] for (s, c, v) in real}
        summary[d] = dsum
        for c in (1, 3):
            seg = audit['segments'].get(c) or []
            if not seg:
                continue
            for k in keep_cls:
                oo = DM.pord[y == k]
                segs = np.array([segment_of(seg, o) for o in oo])
                for sidx in np.unique(segs):
                    a, b, lvl = seg[sidx]
                    ct_rows.append([d, C['classes'][k][0], c, int(sidx) + 1, o2d(a), o2d(b), lvl,
                                    int((segs == sidx).sum()), float(np.mean(segs == sidx))])
    write_csv(outdir / '02_classification_class_sizes.csv',
              ['direction', 'class', 'start_inclusive', 'end_exclusive', 'n_passages', 'n_dates', 'used'], size_rows)
    write_csv(outdir / '02_classification_results.csv',
              ['direction', 'feature_set', 'classifier', 'validation', 'balanced_accuracy_mean', 'balanced_accuracy_sd',
               'n_repeats', 'n_passages', 'n_classes', 'chance_level'], res_rows)
    write_csv(outdir / '02_classification_placebo.csv',
              ['direction', 'variant', 'draw', 'feature_set', 'classifier', 'balanced_accuracy', 'placebo_blocks'], pl_rows)
    write_csv(outdir / '02_classification_regime_crosstab.csv',
              ['direction', 'class', 'channel', 'working_regime', 'regime_first', 'regime_last', 'regime_level_log',
               'n_passages', 'fraction_of_class'], ct_rows)
    S['classification'] = {'random_forest_available': have_rf, 'by_direction': summary}
    return res_rows, pl_rows


# ----------------------------------------------------------------------------
# Module 3: indicator and forecasting
# ----------------------------------------------------------------------------
def run_forecasting(found, outdir, cfg, S, rng):
    log('Module 3: regime-robust indicator and forecasting (calibration to ' + cfg['calibration_end'] + ')')
    FC = cfg['forecast']
    cal_ord = d2o(cfg['calibration_end'])
    store = OrderedDict()
    mrows, rrows, prow, qrows, brows = [], [], [], [], []
    for c, f in cfg['indicator_family']:
        t0 = time.monotonic()
        key = ind_key(c, f)
        model = fit_indicator_model(found.table, c, f, cfg['calibration_end'], cfg)
        series = build_series(found.table, model, FC)
        cal = [series_slice(D, D['ord'] <= cal_ord) for D in series.values()]
        fits = add_break_hazard(fit_all_kalman(cal, FC), cal, FC)
        recs, ranks = collect_forecasts(series, fits, lambda D: np.where(D['ord'] > cal_ord)[0], FC)
        a, b = forecast_metric_rows(key, recs, ranks, FC, rng, 'development')
        mrows += a
        rrows += b
        for m in KALMAN_MODELS:
            p = fits[m]['params']
            qs = tau = s2 = float('nan')
            if m == 'local_trend':
                q, qs, rd, rp = p
                ch90 = math.sqrt(90 * q)
            elif m == 'mean_reverting':
                tau, s2, rd, rp = p
                q = 2.0 * s2 / tau
                ch90 = math.sqrt(2.0 * s2 * (1.0 - math.exp(-90.0 / tau)))
            else:
                q, rd, rp = p
                ch90 = math.sqrt(90 * (q + extra_q(fits, m)))
            noise = math.sqrt(rd + rp)
            prow.append([key, m, q, qs, tau, s2, rd, rp, fits[m]['loglik'], fits[m]['n_obs'], ch90, noise,
                         ch90 / noise if noise > 0 else float('nan'), model['beta_speed'],
                         fits[m].get('break_rate_per_day', float('nan'))])
        for d in DIRECTIONS:
            D = series[d]
            qrows.append([key, d, model['directions'][d]['n_fit_passages'], model['directions'][d]['n_fit_passages_qc'],
                          D['qc_pass_fraction'], len(D['ord']), int(np.sum(D['ord'] > cal_ord)),
                          residual_speed_correlation(found.table, model, d, cal_ord)])
        D0 = next(iter(series.values()))
        for bd, bc in zip(D0['break_days'], D0['break_confirm_days']):
            brows.append([key, o2d(bd), o2d(bc)])
        store[key] = {'channel': c, 'feature': f, 'model': model, 'series': series, 'fits': fits}
        ll = fits['local_level']['params']
        log(f'  {key}: level drift sd over 90 days {math.sqrt(90 * ll[0]):.3f}, passage noise sd {math.sqrt(ll[1] + ll[2]):.3f}, '
            f'speed slope {model["beta_speed"]:.2f}, confirmed regime breaks {sum(1 for r in brows if r[0] == key)}, '
            f'{time.monotonic() - t0:.0f} s')
    write_csv(outdir / '03_forecast_metrics.csv', FORECAST_HEADER, mrows)
    write_csv(outdir / '03_forecast_ranking.csv', RANK_HEADER, rrows)
    write_csv(outdir / '03_kalman_parameters.csv',
              ['indicator', 'model', 'level_drift_variance_per_day', 'slope_variance_per_day', 'reversion_time_days',
               'stationary_level_variance', 'day_noise_variance', 'passage_noise_variance', 'loglik', 'n_obs',
               'level_change_sd_90_days', 'single_passage_noise_sd', 'change_to_noise_ratio_90_days', 'speed_slope',
               'regime_break_rate_per_day'], prow)
    run_pseudo_prospective(store, outdir, cfg, S, rng)
    write_csv(outdir / '03_indicator_qc.csv',
              ['indicator', 'direction', 'n_calibration_passages', 'n_calibration_passages_qc',
               'qc_pass_fraction_all_passages', 'n_days', 'n_evaluation_days',
               'evaluation_anomaly_speed_correlation'], qrows)
    write_csv(outdir / '03_causal_regime_breaks.csv', ['indicator', 'first_day_of_new_regime', 'confirmed_on'], brows)
    prim = ind_key(*cfg['primary_indicator'])
    keys = [prim] + [ind_key(*s) for s in cfg['secondary_indicators']]
    summ = {}
    for r in mrows:
        if r[1] in keys:
            summ.setdefault(r[1], {}).setdefault(r[2], {})[r[3]] = {
                'mae': r[8], 'skill_vs_persistence': [r[10], r[11], r[12]],
                'skill_vs_fingerprint': [r[13], r[14], r[15]], 'coverage80': r[17], 'n_bin_days': r[4],
                'n_episodes': r[5], 'sd_of_target': r[7], 'episodes_better_than_fingerprint': [r[19], r[20], r[21]],
                'episodes_better_than_persistence': [r[22], r[23], r[24]]}
    S.setdefault('forecast', {})
    S['forecast'].update({'primary': prim, 'primary_model': cfg['primary_model'], 'key_indicators': summ,
                          'ranking': {r[1] + '|' + r[2] + '|' + r[3]: [r[6], r[7], r[8], r[9]] for r in rrows if r[1] in keys},
                          'parameters': {r[0] + '|' + r[1]: {'tau_days': r[4], 'change_sd_90': r[10], 'noise_sd': r[11],
                                                             'ratio': r[12]} for r in prow},
                          'family_best_model_30d': {}})
    for key in store:
        cand = [(r[8], r[3]) for r in mrows if r[1] == key and r[2] == horizon_label(30)]
        if cand:
            S['forecast']['family_best_model_30d'][key] = min(cand)[1]
    return store


def run_pseudo_prospective(store, outdir, cfg, S, rng):
    """Holdout-like evaluation inside development data.

    For each cut date C, targets are days in (C + embargo, C + span], as for the real holdout
    (18 to 134 days after the last development day). Fixed origin: forecasts made once at C.
    Sequential: each target forecast from the previous observation day, never from the embargo.
    Models stay those fitted on the calibration period, so no information after C is used.
    """
    FC = cfg['forecast']
    PP = cfg['pseudo_prospective']
    emb, span = PP['embargo_days'], PP['span_days']
    pm = cfg['primary_model']
    rows, rrows, erows = [], [], []
    pooled = {}
    prim = ind_key(*cfg['primary_indicator'])
    keys = [prim] + [ind_key(*s) for s in cfg['secondary_indicators']]
    for key in keys:
        if key not in store:
            continue
        st = store[key]
        # One calendar-episode map for all cut dates, so an episode scored after two cuts stays one unit.
        ep_map = calendar_episodes([o for D in st['series'].values() for o in D['ord'].tolist()], FC['episode_gap_days'])
        for cut in PP['cuts']:
            C = d2o(cut)

            def last_before(D, C=C):
                return int(np.searchsorted(D['ord'], C, side='right') - 1)

            def fixed(D, t, h, C=C):
                return last_before(D)

            def seq(D, t, h, C=C):
                prev = t - 1
                if prev >= 0 and D['ord'][prev] > C + emb:
                    return prev
                return last_before(D)

            def targets(D, C=C):
                return np.where((D['ord'] > C + emb) & (D['ord'] <= C + span))[0]

            for mode, ofn in (('fixed origin', fixed), ('sequential', seq)):
                recs, ranks = collect_forecasts(st['series'], st['fits'], targets, dict(FC, horizons=[mode]), origin_fn=ofn,
                                                episode_of=ep_map)
                scope = f'pseudo-prospective cut {cut}'
                a, b = forecast_metric_rows(key, recs, ranks, FC, rng, scope)
                rows += a
                rrows += b
                if key == prim:
                    erows += episode_rows(key, recs, ranks, scope)
                P = pooled.setdefault((key, mode), {'recs': defaultdict(list), 'ranks': defaultdict(list)})
                for hl, R in recs.items():
                    for k2, v in R.items():
                        P['recs'][k2].append(v)
                for hl, K in ranks.items():
                    for k2, v in K.items():
                        P['ranks'][k2].append(np.asarray(v, dtype=object) if k2 == 'cluster' else v)
    for (key, mode), P in pooled.items():
        if not P['recs']:
            continue
        recs = {mode: {k: np.concatenate(v) for k, v in P['recs'].items()}}
        ranks = {mode: {k: np.concatenate(v) for k, v in P['ranks'].items()}} if P['ranks'] else {}
        a, b = forecast_metric_rows(key, recs, ranks, FC, rng, 'pseudo-prospective pooled cuts')
        rows += a
        rrows += b
    write_csv(outdir / '03_pseudo_prospective.csv', FORECAST_HEADER, rows)
    write_csv(outdir / '03_pseudo_prospective_ranking.csv', RANK_HEADER, rrows)
    write_csv(outdir / '03_pseudo_prospective_episodes.csv', EPISODE_HEADER, erows)
    summ = {}
    for r in rows:
        if r[0] == 'pseudo-prospective pooled cuts' and r[3] in ('local_level', 'mean_reverting', 'moving_mean',
                                                                    'historical_mean', 'persistence'):
            summ[f'{r[1]}|{r[2]}|{r[3]}'] = {'skill_vs_fingerprint': [r[13], r[14], r[15]],
                                             'skill_vs_persistence': [r[10], r[11], r[12]], 'coverage80': r[17],
                                             'n_episodes': r[5], 'episodes_better_than_fingerprint': [r[19], r[20], r[21]]}
    # How the frozen decision rule (exact sign test over episodes) would have decided at each cut date.
    alpha = FC['decision_alpha']

    def ok(p):
        return bool(p is not None and np.isfinite(p) and p < alpha)

    rule = []
    for r in rows:
        if r[0].startswith('pseudo-prospective cut') and r[1] == prim and r[3] == pm:
            rule.append({'cut': r[0].split()[-1], 'mode': r[2], 'episodes': r[20],
                         'vs_persistence': {'skill': r[10], 'episodes_better': r[22], 'p': r[24], 'supported': ok(r[24])},
                         'vs_fingerprint': {'skill': r[13], 'episodes_better': r[19], 'p': r[21], 'supported': ok(r[21])}})
    for r in rrows:
        if r[0].startswith('pseudo-prospective cut') and r[1] == prim and r[3] == pm:
            for x in rule:
                if x['cut'] == r[0].split()[-1] and x['mode'] == r[2]:
                    x['ranking'] = {'mean_spearman': r[6], 'episodes_positive': r[12], 'p': r[14], 'supported': ok(r[14])}
    S.setdefault('forecast', {})['pseudo_prospective'] = summ
    S['forecast']['pseudo_prospective_rule'] = rule
    return rows


# ----------------------------------------------------------------------------
# Module 4: agreement of anomalies between sensors
# ----------------------------------------------------------------------------
def run_cross_sensor(store, outdir, cfg, S, rng):
    log('Module 4: agreement of anomalies between independent sensors')
    CS = cfg['cross_sensor']
    FC = cfg['forecast']
    cal_ord = d2o(cfg['calibration_end'])
    pm = cfg['primary_model']
    rows = []
    summ = {}
    for a, b in CS['pairs']:
        if a not in store or b not in store:
            continue
        vals = {'daily': [], 'filtered': []}
        cl = []
        commons = {}
        for d in DIRECTIONS:
            Da, Db = store[a]['series'][d], store[b]['series'][d]
            commons[d] = np.intersect1d(Da['ord'][Da['ord'] > cal_ord], Db['ord'][Db['ord'] > cal_ord])
        ep_map = calendar_episodes([o for v in commons.values() for o in v.tolist()], FC['episode_gap_days'])
        for d in DIRECTIONS:
            Da, Db = store[a]['series'][d], store[b]['series'][d]
            common = commons[d]
            if len(common) == 0:
                continue
            Xa = Da['states'][pm][0]
            Xb = Db['states'][pm][0]
            La = Da['states'][pm][-1]  # last observation day per bin (the final state element for every model)
            Lb = Db['states'][pm][-1]
            for k, o in enumerate(common):
                ia = int(np.searchsorted(Da['ord'], o))
                ib = int(np.searchsorted(Db['ord'], o))
                ea, eb = Da['E'][ia], Db['E'][ib]
                m = np.isfinite(ea) & np.isfinite(eb)
                fa, fb = Xa[ia], Xb[ib]
                mf = np.isfinite(La[ia]) & np.isfinite(Lb[ib])
                if m.sum() < CS['min_bins'] or mf.sum() < CS['min_bins']:
                    continue
                vals['daily'].append(spearman(ea[m], eb[m]))
                vals['filtered'].append(spearman(fa[mf], fb[mf]))
                cl.append(ep_map[int(o)])
        if not cl:
            continue
        cl = np.array(cl, dtype=object)
        for kind in ('daily', 'filtered'):
            mean, lo, hi, C = cluster_bootstrap_mean(vals[kind], cl, FC['bootstrap'], rng)
            rows.append([a, b, kind, len(cl), C, mean, lo, hi])
            summ[f'{a}|{b}|{kind}'] = [mean, lo, hi]
    write_csv(outdir / '04_cross_sensor_agreement.csv',
              ['indicator_a', 'indicator_b', 'anomaly', 'n_days', 'n_episodes', 'mean_spearman', 'lo95', 'hi95'], rows)
    S['cross_sensor'] = summ
    return rows


# ----------------------------------------------------------------------------
# Module 5: scheduled-work tests against a model-based null
# ----------------------------------------------------------------------------
EVENT_HEADER = ['indicator', 'record_id', 'direction', 'activity', 'start_date', 'end_date', 'days_before', 'days_after',
                'corridor_minus_control_change', 'model_sd', 'z_model', 'placebo_windows', 'independent_placebo_windows',
                'placebo_rms_z', 'calibration_source', 'calibration_factor', 'z', 'p_two_sided', 'q_value_bh',
                'regime_change_in_window', 'valid_test', 'in_expected_direction', 'fdr_bins_corridor',
                'fdr_bins_control', 'bins_tested', 'key_event']
EH = {name: i for i, name in enumerate(EVENT_HEADER)}
EVENT_BIN_HEADER = ['indicator', 'record_id', 'direction', 'track_bin_id', 'chainage_start_m', 'corridor', 'change',
                    'null_sd_calibrated', 'z', 'p_two_sided', 'q_value', 'fdr_significant']
EBH = {name: i for i, name in enumerate(EVENT_BIN_HEADER)}
COMPOSITE_HEADER = ['feature', 'record_id', 'direction', 'activity', 'start_date', 'n_channels', 'channels',
                    'composite_change', 'z_model', 'placebo_windows', 'independent_placebo_windows', 'placebo_rms_z',
                    'calibration_factor', 'composite_z', 'p_two_sided', 'q_value_bh', 'channel_z',
                    'channels_agree_in_sign', 'in_expected_direction', 'key_event']
CH = {name: i for i, name in enumerate(COMPOSITE_HEADER)}
NULL_HEADER = ['indicator', 'direction', 'n_days', 'n_days_outside_work_windows', 'n_fit_observations',
               'isolated_days_removed', 'drift_variance_per_day', 'noise_variance', 'level_change_sd_90_days',
               'noise_sd', 'bin_drift_variance_per_day', 'bin_noise_variance']


def screen_isolated_days(Y, half, k):
    """Remove gross single-day errors.

    A value is removed when it lies more than k robust SDs from the centred running median of 2 * half + 1
    observed days. A running median follows a lasting step, so genuine level changes are kept.
    """
    Y = np.array(Y, float, copy=True)
    Y2 = Y if Y.ndim == 2 else Y[:, None]
    n_out = 0
    for j in range(Y2.shape[1]):
        idx = np.where(np.isfinite(Y2[:, j]))[0]
        if len(idx) < 2 * half + 1:
            continue
        v = Y2[idx, j]
        med = np.array([np.median(v[max(0, i - half):i + half + 1]) for i in range(len(v))])
        res = v - med
        s = 1.4826 * float(np.median(np.abs(res - np.median(res))))
        if not np.isfinite(s) or s <= 0:
            continue
        bad = np.abs(res) > k * s
        Y2[idx[bad], j] = np.nan
        n_out += int(bad.sum())
    return (Y2 if Y.ndim == 2 else Y2[:, 0]), n_out


def contrast_series(D, min_bins):
    """Daily corridor median minus control median of the indicator anomaly."""
    E, cor = D['E'], np.asarray(D['corridor'], bool)
    a, b = E[:, cor], E[:, ~cor]
    c = nanmed(a, axis=1) - nanmed(b, axis=1)
    c = np.asarray(c, float)
    c[(np.sum(np.isfinite(a), axis=1) < min_bins) | (np.sum(np.isfinite(b), axis=1) < min_bins)] = np.nan
    return c


def bin_contrast_series(D, min_bins):
    """Each bin's anomaly minus the control median of the same day."""
    E, cor = D['E'], np.asarray(D['corridor'], bool)
    b = E[:, ~cor]
    ref = np.asarray(nanmed(b, axis=1), float)
    ref[np.sum(np.isfinite(b), axis=1) < min_bins] = np.nan
    return E - ref[:, None]


def rw_noise_loglik(Y, td, restart, q, r):
    """Gaussian log-likelihood of a random walk plus noise for each column of Y (days by series).

    The level starts from a diffuse state at each column's first observation and after each restart day.
    """
    T, B = Y.shape
    x = np.zeros(B)
    P = np.zeros(B)
    last = np.zeros(B)
    seen = np.zeros(B, bool)
    ll, n = 0.0, 0
    for t in range(T):
        if restart[t]:
            seen = np.zeros(B, bool)
        y = Y[t]
        obs = np.isfinite(y)
        use = obs & seen
        dt = np.where(seen, td[t] - last, 0.0)
        Pp = P + q * dt
        S = Pp + r
        v = np.where(obs, y - x, 0.0)
        if use.any():
            ll += float(-0.5 * np.sum(np.log(2 * np.pi * S[use]) + v[use] ** 2 / S[use]))
            n += int(use.sum())
        K = Pp / S
        new = obs & ~seen
        x = np.where(new, np.where(obs, y, 0.0), np.where(use, x + K * v, x))
        P = np.where(new, r, np.where(use, (1 - K) * Pp, P))
        last = np.where(obs, td[t], last)
        seen = seen | obs
    return ll, n


def fit_rw_noise(Y, td, restart):
    """Maximum-likelihood drift and noise variances of a random walk plus noise, shared by all columns."""
    Y = Y if Y.ndim == 2 else Y[:, None]
    fin = Y[np.isfinite(Y)]
    if fin.size < 10:
        return None
    dif = np.diff(Y, axis=0)
    dif = dif[np.isfinite(dif)]
    r0 = (mad_sd(dif) ** 2) / 2.0 if dif.size >= 5 else float(np.var(fin))
    r0 = max(float(r0) if np.isfinite(r0) else 0.0, 1e-8)
    lo = np.array([math.log(1e-10), math.log(r0 * 1e-4)])
    hi = np.array([math.log(10.0 * r0 + 1.0), math.log(r0 * 1e3)])

    def nll(th):
        q, r = np.exp(np.clip(th, lo, hi))
        ll, _ = rw_noise_loglik(Y, td, restart, q, r)
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for qs in (r0 / 300.0, r0 / 30.0, r0 / 3.0):
        res = optimize.minimize(nll, np.log([qs, r0]), method='Nelder-Mead',
                                options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
        if best is None or res.fun < best.fun:
            best = res
    q, r = np.exp(np.clip(best.x, lo, hi))
    ll, n = rw_noise_loglik(Y, td, restart, q, r)
    if n < 5:
        return None
    return {'q': float(q), 'r': float(r), 'n_obs': int(n), 'loglik': float(ll)}


def change_and_variance(Y, td, ds, de, W, m, q, r):
    """Mean after the work minus mean before it, for each column, and its variance under the null.

    Before: days in [ds - W, ds); after: days in (de, de + W]. Under a random walk with drift variance q per
    day plus independent noise r, the difference of window means has variance
    -q/2 * sum_ij a_i a_j |t_i - t_j| + r * sum_i a_i^2, with weights a summing to zero.
    """
    Y = Y if Y.ndim == 2 else Y[:, None]
    B = Y.shape[1]
    bef = (td >= ds - W) & (td < ds)
    aft = (td > de) & (td <= de + W)
    idx = np.where(bef | aft)[0]
    if len(idx) == 0:
        return np.full(B, np.nan), np.full(B, np.nan), np.zeros(B, int), np.zeros(B, int)
    sub = Y[idx]
    M = np.isfinite(sub)
    isb = bef[idx][:, None]
    nb = (M & isb).sum(axis=0)
    na = (M & ~isb).sum(axis=0)
    A = np.where(M, np.where(isb, -1.0 / np.maximum(nb, 1), 1.0 / np.maximum(na, 1)), 0.0)
    tt = td[idx].astype(float)
    Dm = np.abs(tt[:, None] - tt[None, :])
    vrw = -0.5 * np.einsum('ib,ij,jb->b', A, Dm, A)
    var = q * vrw + r * np.sum(A * A, axis=0)
    delta = np.sum(A * np.where(M, sub, 0.0), axis=0)
    ok = (nb >= m) & (na >= m) & (var > 0)
    return np.where(ok, delta, np.nan), np.where(ok, var, np.nan), nb, na


def null_masks(td, spans_d, boundaries, W):
    """Days left out of the null fit (within W days of scheduled work) and days where the level restarts
    (after each left-out stretch and at each working-regime change)."""
    td = np.asarray(td, int)
    excl = np.zeros(len(td), bool)
    for a, b, _, _ in spans_d:
        excl |= (td >= a - W) & (td <= b + W)
    restart = np.zeros(len(td), bool)
    for i in range(len(td)):
        if not excl[i] and (i == 0 or excl[i - 1]):
            restart[i] = True
    for _, first_after, _ in boundaries:
        k = int(np.searchsorted(td, first_after, side='left'))
        if k < len(td):
            restart[k] = True
    return excl, restart


def window_is_clean(lo, hi, spans_d, boundaries):
    if any(a <= hi and b >= lo for a, b, _, _ in spans_d):
        return False
    return not boundary_in_span(boundaries, lo, hi)


def placebo_z(Y, td, dur, W, m, q, r, spans_d, boundaries):
    """Model z-values of the change statistic for windows of the event's shape on every clean calendar day."""
    Y = Y if Y.ndim == 2 else Y[:, None]
    zs, starts = [], []
    if len(td):
        for t in range(int(td.min()) + W, int(td.max()) - W - dur + 1):
            if not window_is_clean(t - W, t + dur + W, spans_d, boundaries):
                continue
            dl, vv, _, _ = change_and_variance(Y, td, t, t + dur, W, m, q, r)
            if not np.any(np.isfinite(dl)):
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
                zs.append(dl / np.sqrt(vv))
            starts.append(t)
    return (np.array(zs) if zs else np.zeros((0, Y.shape[1]))), starts


def independent_windows(starts, length):
    """Greedy count of non-overlapping windows among the clean placebo windows."""
    n, last = 0, None
    for t in sorted(starts):
        if last is None or t >= last + length:
            n += 1
            last = t
    return n


def calibration_factor(z, min_n):
    """Root-mean-square placebo z floored at one: it can widen the model standard error, never narrow it."""
    z = np.asarray(z, float).ravel()
    z = z[np.isfinite(z)]
    if len(z) < min_n:
        return float('nan'), float('nan'), int(len(z))
    rms = float(np.sqrt(np.mean(z ** 2)))
    return max(1.0, rms), rms, int(len(z))


def run_events(found, store, audit, outdir, cfg, S, with_bins=True):
    log('Module 5: scheduled-work tests against a model-based null' + ('' if with_bins else
                                                                        f' (window {cfg["events"]["window_days"]} days)'))
    EV = cfg['events']
    W, m, mb = EV['window_days'], EV['min_days_each_side'], EV['min_bins_each_group']
    half, kout, min_pl = EV['outlier_half_window'], EV['outlier_robust_sd'], EV['min_clean_placebo']
    sgn = EV.get('expected_sign', -1)
    spans = found.schedule_spans()
    sum_rows, bin_rows, comp_rows, null_rows = [], [], [], []
    chan_info = {}
    single = {}
    for key, st in store.items():
        ch, feat = st['channel'], st['feature']
        bnd = channel_boundaries(audit, ch)
        for d in DIRECTIONS:
            D = st['series'][d]
            td = D['ord'].astype(int)
            if len(td) == 0:
                continue
            sp = spans.get(d, [])
            excl, restart = null_masks(td, sp, bnd, W)
            C, n_out = screen_isolated_days(contrast_series(D, mb), half, kout)
            fit = fit_rw_noise(np.where(excl, np.nan, C), td, restart)
            if fit is None:
                log(f'  {key} {d}: too few days outside scheduled-work windows for the null model; events not tested')
                continue
            q, r = fit['q'], fit['r']
            chan_info[(feat, d, ch)] = {'td': td, 'C': C, 'r': r, 'bnd': bnd}
            Bc = bfit = None
            if with_bins:
                Bc, _ = screen_isolated_days(bin_contrast_series(D, mb), half, kout)
                bfit = fit_rw_noise(np.where(excl[:, None], np.nan, Bc), td, restart)
            null_rows.append([key, d, len(td), int((~excl).sum()), fit['n_obs'], n_out, q, r, math.sqrt(90 * q),
                              math.sqrt(r), bfit['q'] if bfit else None, bfit['r'] if bfit else None])
            cache, cache_b = {}, {}

            def placebo(dur, bins=False):
                store_ = cache_b if bins else cache
                if dur not in store_:
                    if bins:
                        store_[dur] = placebo_z(Bc, td, dur, W, m, bfit['q'], bfit['r'], sp, bnd)
                    else:
                        store_[dur] = placebo_z(C, td, dur, W, m, q, r, sp, bnd)
                return store_[dur]

            for rec in found.coverage:
                if rec['direction'] != d:
                    continue
                rid = rec['record_id']
                ds, de = d2o(rec['start_date']), d2o(rec['end_date'])
                dur = de - ds
                delta, var, nbd, nad = change_and_variance(C, td, ds, de, W, m, q, r)
                delta, var = float(delta[0]), float(var[0])
                regime = boundary_in_span(bnd, ds - W, de + W)
                testable = bool(np.isfinite(delta) and np.isfinite(var) and var > 0)
                z_model = delta / math.sqrt(var) if testable else float('nan')
                zp, starts = placebo(dur)
                cf, rms, n_pl = calibration_factor(zp, min_pl)
                source = 'windows of the same duration'
                if not np.isfinite(cf):
                    zg, _ = placebo(0)
                    cf, rms, _ = calibration_factor(zg, min_pl)
                    source = 'one-day windows'
                    if not np.isfinite(cf):
                        source = 'none (too few clean windows)'
                n_ind = independent_windows(starts, 2 * W + dur + 1)
                z = z_model / cf if (testable and np.isfinite(cf)) else float('nan')
                p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float('nan')
                valid = bool(np.isfinite(z) and not regime)
                n_sig_c = n_sig_k = n_tested = 0
                if with_bins and bfit is not None:
                    bd_, bv_, _, _ = change_and_variance(Bc, td, ds, de, W, m, bfit['q'], bfit['r'])
                    zpb, starts_b = placebo(dur, bins=True)
                    cfb = calibration_factor(zpb, min_pl * 5)[0] if len(starts_b) >= min_pl else float('nan')
                    if not np.isfinite(cfb):
                        zgb, sgb = placebo(0, bins=True)
                        cfb = calibration_factor(zgb, min_pl * 5)[0] if len(sgb) >= min_pl else float('nan')
                    with np.errstate(invalid='ignore', divide='ignore'):
                        sdb = np.sqrt(bv_) * cfb
                        zb = bd_ / sdb
                    pb = np.where(np.isfinite(zb), 2 * stats.norm.sf(np.abs(zb)), np.nan)
                    qb = bh_fdr(pb)
                    sig = np.isfinite(qb) & (qb <= EV['fdr_q'])
                    cor = np.asarray(D['corridor'], bool)
                    n_sig_c, n_sig_k, n_tested = int((sig & cor).sum()), int((sig & ~cor).sum()), int(np.isfinite(pb).sum())
                    for b in range(len(bd_)):
                        if np.isfinite(bd_[b]):
                            bin_rows.append([key, rid, d, D['bins'][b], D['chainage'][b], bool(cor[b]), bd_[b], sdb[b],
                                             zb[b], pb[b], qb[b], bool(sig[b])])
                sum_rows.append([key, rid, d, rec['activity'], rec['start_date'], rec['end_date'], int(nbd[0]), int(nad[0]),
                                 delta if testable else float('nan'), math.sqrt(var) if testable else float('nan'),
                                 z_model, n_pl, n_ind, rms, source, cf, z, p, float('nan'), regime, valid,
                                 bool(np.isfinite(z) and np.sign(z) == sgn), n_sig_c, n_sig_k, n_tested,
                                 rid in EV['key_events']])
                if valid:
                    single[(rid, d, feat, ch)] = z
    # Benjamini-Hochberg over all valid single-indicator tests.
    vi = [i for i, r in enumerate(sum_rows) if r[EH['valid_test']]]
    qv = bh_fdr(np.array([sum_rows[i][EH['p_two_sided']] for i in vi], float)) if vi else []
    for i, qq in zip(vi, qv):
        sum_rows[i][EH['q_value_bh']] = float(qq)
    # Multi-channel composite: the event's own valid channels only, each contrast scaled by its noise SD.
    comp_cache = {}
    for feat in sorted({st['feature'] for st in store.values()}):
        for d in DIRECTIONS:
            for rec in found.coverage:
                if rec['direction'] != d:
                    continue
                rid = rec['record_id']
                ds, de = d2o(rec['start_date']), d2o(rec['end_date'])
                dur = de - ds
                chans = [c for c in EV['composite_channels'] if (rid, d, feat, c) in single]
                base = [feat, rid, d, rec['activity'], rec['start_date'], len(chans), ' '.join(f's{c}' for c in chans)]
                if len(chans) < EV['composite_min_channels']:
                    comp_rows.append(base + [float('nan')] * 2 + [0, 0] + [float('nan')] * 5 + ['', None, None,
                                                                                               rid in EV['key_events']])
                    continue
                ck = (feat, d, tuple(chans))
                if ck not in comp_cache:
                    days = np.array(sorted(set().union(*[set(chan_info[(feat, d, c)]['td'].tolist()) for c in chans])), int)
                    Mx = np.full((len(days), len(chans)), np.nan)
                    for j, c in enumerate(chans):
                        ci = chan_info[(feat, d, c)]
                        Mx[np.searchsorted(days, ci['td']), j] = ci['C'] / math.sqrt(ci['r'])
                    comp = np.where(np.all(np.isfinite(Mx), axis=1), np.mean(Mx, axis=1), np.nan)
                    bnd_all = [b for c in chans for b in chan_info[(feat, d, c)]['bnd']]
                    excl, restart = null_masks(days, spans.get(d, []), bnd_all, W)
                    fit = fit_rw_noise(np.where(excl, np.nan, comp), days, restart)
                    comp_cache[ck] = (days, comp, bnd_all, fit, {})
                days, comp, bnd_all, fit, pcache = comp_cache[ck]
                zz = {c: single[(rid, d, feat, c)] for c in chans}
                detail = ' '.join(f's{c}:{zz[c]:+.2f}' for c in chans)
                same = bool(all(v < 0 for v in zz.values()) or all(v > 0 for v in zz.values()))
                if fit is None:
                    comp_rows.append(base + [float('nan')] * 2 + [0, 0] + [float('nan')] * 5 +
                                     [detail, same, None, rid in EV['key_events']])
                    continue
                delta, var, _, _ = change_and_variance(comp, days, ds, de, W, m, fit['q'], fit['r'])
                delta, var = float(delta[0]), float(var[0])
                if dur not in pcache:
                    pcache[dur] = placebo_z(comp, days, dur, W, m, fit['q'], fit['r'], spans.get(d, []), bnd_all)
                zp, starts = pcache[dur]
                cf, rms, n_pl = calibration_factor(zp, min_pl)
                if not np.isfinite(cf):
                    if 0 not in pcache:
                        pcache[0] = placebo_z(comp, days, 0, W, m, fit['q'], fit['r'], spans.get(d, []), bnd_all)
                    cf, rms, _ = calibration_factor(pcache[0][0], min_pl)
                zm = delta / math.sqrt(var) if (np.isfinite(delta) and var > 0) else float('nan')
                cz = zm / cf if (np.isfinite(zm) and np.isfinite(cf)) else float('nan')
                p = float(2 * stats.norm.sf(abs(cz))) if np.isfinite(cz) else float('nan')
                comp_rows.append(base + [delta, zm, n_pl, independent_windows(starts, 2 * W + dur + 1), rms, cf, cz, p,
                                         float('nan'), detail, same, bool(np.isfinite(cz) and np.sign(cz) == sgn),
                                         rid in EV['key_events']])
    ci_ = [i for i, r in enumerate(comp_rows) if r[CH['p_two_sided']] is not None and np.isfinite(r[CH['p_two_sided']])]
    qv = bh_fdr(np.array([comp_rows[i][CH['p_two_sided']] for i in ci_], float)) if ci_ else []
    for i, qq in zip(ci_, qv):
        comp_rows[i][CH['q_value_bh']] = float(qq)
    write_csv(outdir / '05_events_summary.csv', EVENT_HEADER, sum_rows)
    if with_bins:
        write_csv(outdir / '05_events_bins.csv', EVENT_BIN_HEADER, bin_rows)
        write_csv(outdir / '05_events_null_models.csv', NULL_HEADER, null_rows)
    write_csv(outdir / '05_events_composite.csv', COMPOSITE_HEADER, comp_rows)
    prim = ind_key(*cfg['primary_indicator'])
    valid = [r for r in sum_rows if r[EH['valid_test']]]
    rms_all = [r[EH['placebo_rms_z']] for r in sum_rows if r[EH['placebo_rms_z']] is not None and np.isfinite(r[EH['placebo_rms_z']])]
    S['events'] = {
        'primary_indicator': prim,
        'record_direction_rows': len(found.coverage),
        'valid_tests_primary': sum(1 for r in valid if r[0] == prim),
        'regime_confounded_primary': sum(1 for r in sum_rows if r[0] == prim and r[EH['regime_change_in_window']]
                                         and np.isfinite(r[EH['z_model']])),
        'valid_tests_all': len(valid),
        'valid_p_below_0_05_all': sum(1 for r in valid if r[EH['p_two_sided']] < 0.05),
        'valid_q_below_fdr_all': sum(1 for r in valid if np.isfinite(r[EH['q_value_bh']]) and r[EH['q_value_bh']] <= EV['fdr_q']),
        'valid_negative_all': sum(1 for r in valid if r[EH['z']] < 0),
        'placebo_rms_median': float(np.median(rms_all)) if rms_all else None,
        'placebo_rms_above_one_fraction': float(np.mean(np.array(rms_all) > 1.0)) if rms_all else None,
        'key_primary': {r[1] + '|' + r[2]: {'change': r[EH['corridor_minus_control_change']], 'z': r[EH['z']],
                                            'p': r[EH['p_two_sided']], 'regime_confounded': r[EH['regime_change_in_window']]}
                        for r in sum_rows if r[0] == prim and r[EH['key_event']]},
        'composite': {r[0] + '|' + r[1] + '|' + r[2]: {'z': r[CH['composite_z']], 'p': r[CH['p_two_sided']],
                                                       'q': r[CH['q_value_bh']], 'channels': r[CH['channels']],
                                                       'detail': r[CH['channel_z']], 'signs_agree': r[CH['channels_agree_in_sign']],
                                                       'independent_placebo_windows': r[CH['independent_placebo_windows']]}
                      for r in comp_rows if r[CH['key_event']] and r[CH['composite_z']] is not None
                      and np.isfinite(r[CH['composite_z']])},
    }
    return sum_rows, bin_rows, comp_rows


def run_event_window_sensitivity(found, store, audit, outdir, cfg, S, main_comp):
    """Repeat the composite event tests with other window lengths (pre-declared)."""
    log('Module 5b: window-length sensitivity of the composite event tests')
    rows = []
    W0 = cfg['events']['window_days']

    def take(W, comp):
        for r in comp:
            if r[CH['key_event']] and r[CH['composite_z']] is not None and np.isfinite(r[CH['composite_z']]):
                rows.append([int(W), r[0], r[1], r[2], r[CH['channels']], r[CH['composite_z']], r[CH['p_two_sided']],
                             r[CH['placebo_windows']], r[CH['independent_placebo_windows']], r[CH['channels_agree_in_sign']]])

    take(W0, main_comp)
    for W, md in cfg['events'].get('sensitivity_windows', []):
        c2 = copy.deepcopy(cfg)
        c2['events']['window_days'] = int(W)
        c2['events']['min_days_each_side'] = int(md)
        sub = outdir / 'event_window_sensitivity' / f'window_{int(W)}_days'
        sub.mkdir(parents=True, exist_ok=True)
        _, _, comp = run_events(found, store, audit, sub, c2, {}, with_bins=False)
        take(W, comp)
    write_csv(outdir / '05_events_window_sensitivity.csv',
              ['window_days', 'feature', 'record_id', 'direction', 'channels', 'composite_z', 'p_two_sided',
               'placebo_windows', 'independent_placebo_windows', 'channels_agree_in_sign'], rows)
    S.setdefault('events', {})['window_sensitivity'] = [r for r in rows]
    return rows


# ----------------------------------------------------------------------------
# Module 6: detection limits by synthetic step injection
# ----------------------------------------------------------------------------
def segment_null_stats(series, model_name, params, DT, origin_min_ord, xq=0.0, target_mask_key=None, records=None,
                       origin_mask_key=None):
    """Segment means of standardised forecast errors over the m observation days after each origin.

    Every observation day at or after origin_min_ord is an origin. With target_mask_key, the m target days
    must be flagged in D[target_mask_key] (used to score the holdout exactly as the development null).
    Returns, per (segment bins, days after), the statistics and their weights; when records is a list, one
    entry per statistic is appended to it (direction, bins, days, segment, origin and target days).
    """
    out = {(L, m): ([], []) for L in DT['segment_bins'] for m in DT['days_after']}
    for d, D in series.items():
        if 'states' not in D or D['E'].shape[0] == 0:
            continue
        state = D['states'][model_name]
        E, N, td = D['E'], D['N'], D['ord']
        tmask = D[target_mask_key] if target_mask_key else np.ones(len(td), bool)
        omask = D[origin_mask_key] if origin_mask_key else np.ones(len(td), bool)
        for o in range(len(td)):
            if td[o] < origin_min_ord or not omask[o]:
                continue
            fut = np.where((td > td[o]) & (td <= td[o] + DT['max_days_ahead']) & tmask)[0]
            for m in DT['days_after']:
                if len(fut) < m:
                    continue
                J = fut[:m]
                mu, var = state_forecast(model_name, state, o, td[J], N[J], params, xq)
                with np.errstate(invalid='ignore', divide='ignore'):
                    Z = (E[J] - mu) / np.sqrt(var)
                    Wt = 1.0 / np.sqrt(var)
                valid = np.isfinite(Z) & np.isfinite(Wt)
                zs = np.where(valid, Z, 0.0).sum(axis=0)
                ws = np.where(valid, Wt, 0.0).sum(axis=0)
                cn = valid.sum(axis=0).astype(float)
                for L in DT['segment_bins']:
                    if L > len(zs):
                        continue
                    kern = np.ones(L)
                    zs_s = np.convolve(zs, kern, 'valid')
                    ws_s = np.convolve(ws, kern, 'valid')
                    cn_s = np.convolve(cn, kern, 'valid')
                    need = max(1, int(math.ceil(0.5 * L * m)))
                    ok = cn_s >= need
                    out[(L, m)][0].extend((zs_s[ok] / cn_s[ok]).tolist())
                    out[(L, m)][1].extend((ws_s[ok] / cn_s[ok]).tolist())
                    if records is not None:
                        cor = np.asarray(D['corridor'], bool)
                        for b0 in np.where(ok)[0].tolist():
                            records.append([d, L, m, D['bins'][b0], D['bins'][b0 + L - 1], float(D['chainage'][b0]),
                                            bool(cor[b0:b0 + L].any()), float(zs_s[b0] / cn_s[b0]), o2d(td[o]),
                                            o2d(td[J[0]]), o2d(td[J[-1]])])
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()}


def detection_tables(nulls, DT, key, scope):
    rows, mdc = [], []
    scale = quantity_scale(ind_feature(key))
    qname = quantity_name(ind_feature(key))
    # The step grid spans the same amplitude range for every quantity (twice the log range for power).
    dmax = DT['delta_max'] * scale
    grid = np.round(np.arange(-dmax, dmax + 1e-9, DT['delta_step']), 6)
    thresholds = {}
    for (L, m), (Sv, Wv) in nulls.items():
        if len(Sv) < 20:
            continue
        T = float(np.quantile(np.abs(Sv), 1 - DT['alarm_burden']))
        thresholds[f'{L}|{m}'] = {'threshold': T, 'n_null': int(len(Sv))}
        prob = np.array([np.mean(np.abs(Sv + dl * Wv) > T) for dl in grid])
        for dl, p in zip(grid, prob):
            if abs(round(dl / 0.05) * 0.05 - dl) < 1e-9:
                rows.append([scope, key, qname, L, m, float(dl), pct_change(float(dl)), pct_change(float(dl) / scale),
                             float(p), T, len(Sv)])
        pos, neg = grid > 0, grid < 0
        hit_pos = grid[pos][prob[pos] >= DT['power_target']]
        hit_neg = grid[neg][prob[neg] >= DT['power_target']]
        mpos = float(hit_pos.min()) if len(hit_pos) else float('nan')
        mneg = float(hit_neg.max()) if len(hit_neg) else float('nan')
        fin = np.isfinite
        mdc.append([scope, key, qname, L, m, mpos, pct_change(mpos) if fin(mpos) else float('nan'),
                    pct_change(mpos / scale) if fin(mpos) else float('nan'), mneg,
                    pct_change(mneg) if fin(mneg) else float('nan'),
                    pct_change(mneg / scale) if fin(mneg) else float('nan'), T, len(Sv)])
    return rows, mdc, thresholds


DETECTION_HEADER = ['scope', 'indicator', 'quantity', 'segment_bins', 'days_after', 'step_log', 'step_percent_of_quantity',
                    'step_percent_amplitude_equivalent', 'detection_probability', 'threshold', 'n_null_statistics']
MDC_HEADER = ['scope', 'indicator', 'quantity', 'segment_bins', 'days_after', 'mdc80_increase_log',
              'mdc80_increase_percent_of_quantity', 'mdc80_increase_percent_amplitude', 'mdc80_decrease_log',
              'mdc80_decrease_percent_of_quantity', 'mdc80_decrease_percent_amplitude', 'threshold', 'n_null_statistics']


def run_detection(store, outdir, cfg, S):
    log('Module 6: detection limits by synthetic step injection')
    DT = cfg['detection']
    pm = cfg['primary_model']
    cal_ord = d2o(cfg['calibration_end'])
    rows, mdc = [], []
    keys = [ind_key(*cfg['primary_indicator'])] + [ind_key(*s) for s in cfg['secondary_indicators']]
    for key, st in store.items():
        nulls = segment_null_stats(st['series'], pm, st['fits'][pm]['params'], DT, cal_ord + 1, extra_q(st['fits'], pm))
        a, b, _ = detection_tables(nulls, DT, key, 'development')
        rows += a
        mdc += b
    write_csv(outdir / '06_detection_power.csv', DETECTION_HEADER, rows)
    write_csv(outdir / '06_detection_mdc.csv', MDC_HEADER, mdc)
    S['detection'] = {f'{r[1]}|L{r[3]}|m{r[4]}': {'increase_amplitude_percent': r[7], 'decrease_amplitude_percent': r[10]}
                      for r in mdc if r[1] in keys}
    return rows, mdc


# ----------------------------------------------------------------------------
# Full-development components (priority list and frozen protocol)
# ----------------------------------------------------------------------------
def fit_full_components(found, cfg, indicators):
    FC = cfg['forecast']
    DT = cfg['detection']
    pm = cfg['primary_model']
    comps = OrderedDict()
    for c, f in indicators:
        key = ind_key(c, f)
        log(f'  fitting {key} on all development data')
        model = fit_indicator_model(found.table, c, f, found.dev_last_date, cfg)
        series = build_series(found.table, model, FC)
        fits = add_break_hazard(fit_all_kalman(list(series.values()), FC), list(series.values()), FC)
        for D in series.values():
            compute_states(D, fits, FC)
        first = min(int(D['ord'][0]) for D in series.values() if len(D['ord']))
        nulls = segment_null_stats(series, pm, fits[pm]['params'], DT, first + FC['burn_in_days'], extra_q(fits, pm))
        _, mdc, thresholds = detection_tables(nulls, DT, key, 'all development')
        comps[key] = {'channel': c, 'feature': f, 'model': model, 'series': series, 'fits': fits,
                      'thresholds': thresholds, 'mdc': mdc}
    return comps


def priority_limit(cfg):
    """Review limit on the indicator scale: the amplitude limit, doubled for log power."""
    return cfg['forecast']['priority_limit_log'] * quantity_scale(cfg['primary_indicator'][1])


def priority_rows(comps, cfg):
    FC = cfg['forecast']
    pm = cfg['primary_model']
    key = ind_key(*cfg['primary_indicator'])
    st = comps[key]
    lim = priority_limit(cfg)
    rows = []
    for d, D in st['series'].items():
        if len(D['ord']) == 0:
            continue
        state = D['states'][pm]
        LAST = state[-1]
        target = int(D['ord'][-1]) + FC['priority_horizon_days']
        mu, var = state_forecast(pm, state, len(D['ord']) - 1, target, np.full(len(D['bins']), 2.0), st['fits'][pm]['params'],
                                 extra_q(st['fits'], pm))
        sd = np.sqrt(var)
        prob = 1.0 - stats.norm.cdf((lim - mu) / sd)
        for b in range(len(mu)):
            if np.isfinite(LAST[-1][b]):
                rows.append([d, D['bins'][b], D['chainage'][b], bool(D['corridor'][b]), o2d(LAST[-1][b]),
                             o2d(target), mu[b], sd[b], prob[b]])
    rows.sort(key=lambda r: -r[8])
    return [[i + 1] + r for i, r in enumerate(rows)]


def run_priority(comps, outdir, cfg, S):
    log('Module 7: end-of-development priority list')
    out = priority_rows(comps, cfg)
    write_csv(outdir / '07_priority_list.csv',
              ['rank', 'direction', 'track_bin_id', 'chainage_start_m', 'corridor', 'last_observed', 'target_date',
               'forecast_anomaly_log', 'forecast_sd', 'prob_exceeds_limit'], out)
    FC = cfg['forecast']
    S['priority'] = {'indicator': ind_key(*cfg['primary_indicator']), 'model': cfg['primary_model'],
                     'limit_log': priority_limit(cfg), 'limit_percent': pct_change(FC['priority_limit_log']),
                     'top5': [{'direction': r[1], 'bin': r[2], 'corridor': r[4], 'anomaly': r[7], 'prob': r[9]} for r in out[:5]],
                     'bins_with_prob_above_0_5': int(sum(1 for r in out if r[9] > 0.5))}
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def get_plt():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        style = {
            'font.family': 'sans-serif', 'font.size': 8.5, 'axes.titlesize': 9.0, 'axes.labelsize': 8.5,
            'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
            'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
            'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
            'grid.linestyle': '-', 'axes.spines.top': False, 'axes.spines.right': False,
            'legend.frameon': False, 'legend.fontsize': 8, 'figure.dpi': 110, 'savefig.dpi': 200,
            'figure.facecolor': 'white', 'axes.facecolor': 'white', 'lines.linewidth': 1.5,
        }
        for k, v in style.items():
            try:  # older Matplotlib versions lack some keys; skip those rather than lose all figures
                plt.rcParams[k] = v
            except (KeyError, ValueError):
                pass
        return plt
    except Exception as exc:  # pragma: no cover
        log(f'Matplotlib unavailable ({exc}); figures skipped.')
        return None


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list('blue_grey_red', ['#0d366b', '#2a78d6', '#9ec5f4', NEUTRAL,
                                                             '#f1a9a8', '#e34948', '#8c1d1d'])
    cm.set_bad('white')
    return cm


def date_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))


def ord_to_dt(o):
    return datetime.fromordinal(int(o))


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def fig_levels(plt, audit, spans, path):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 8.8), sharex=True)
    for ax, c in zip(axes, CHANNELS):
        arr = audit['levels'][c]
        o = np.array([a[0] for a in arr])
        v = np.array([a[1] for a in arr])
        ok = np.array([a[2] for a in arr])
        for d in DIRECTIONS:
            for a, b, _, _ in spans.get(d, []):
                ax.axvspan(ord_to_dt(a), ord_to_dt(b + 1), color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.scatter([ord_to_dt(x) for x in o[~ok]], v[~ok], s=9, facecolors='none', edgecolors=MUTED,
                   linewidths=0.6, zorder=2)
        ax.scatter([ord_to_dt(x) for x in o[ok]], v[ok], s=9, color=PAL['blue'], lw=0, zorder=3)
        for a, b, lvl in audit['segments'][c]:
            ax.plot([ord_to_dt(a), ord_to_dt(b)], [lvl, lvl], color=PAL['orange'], lw=2, zorder=4,
                    solid_capstyle='round')
        ax.set_ylabel(f's{c} level\n(log RMS)')
        ax.set_title(f'Channel {c}: {CHANNEL_NAMES[c]}', loc='left')
    handles = [Line2D([], [], marker='o', ls='', color=PAL['blue'], ms=4, label='Passage, QC pass'),
               Line2D([], [], marker='o', ls='', mfc='none', mec=MUTED, ms=4, label='Passage, QC fail'),
               Line2D([], [], color=PAL['orange'], lw=2, label='Working regime median'),
               Patch(color=GRID, label='Scheduled corridor work')]
    fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.01))
    date_axis(axes[-1])
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_fig(fig, path)


def fig_anomaly_map(plt, store, key, spans, audit, cfg, path):
    import matplotlib.dates as mdates
    st = store[key]
    ch_num = st['channel']
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    cmap = diverging_cmap()
    im = None
    for ax, d in zip(axes, DIRECTIONS):
        D = st['series'][d]
        if D['E'].shape[0] == 0:
            continue
        ords, ch = D['ord'], D['chainage']
        o0, o1 = int(ords.min()) - 3, int(ords.max()) + 3
        grid = np.full((len(ch), o1 - o0 + 1), np.nan)
        for k, o in enumerate(ords):
            lo = max(int(o) - 2 - o0, 0)
            hi = min(int(o) + 2 - o0, grid.shape[1] - 1)
            grid[:, lo:hi + 1] = D['E'][k][:, None]
        x0, x1 = mdates.date2num(ord_to_dt(o0)), mdates.date2num(ord_to_dt(o1 + 1))
        vlim = 0.5 * quantity_scale(st['feature'])
        im = ax.imshow(np.ma.masked_invalid(grid), aspect='auto', origin='lower', cmap=cmap, vmin=-vlim, vmax=vlim,
                       interpolation='nearest', extent=[x0, x1, ch[0], ch[-1] + 100])
        ax.xaxis_date()
        cor = D['corridor']
        if cor.any():
            for yv in (ch[cor].min(), ch[cor].max() + 100):
                ax.axhline(yv, color=INK, lw=0.8)
        ax.axvline(mdates.date2num(ord_to_dt(d2o(cfg['calibration_end']) + 1)), color=INK, lw=0.8)
        for a, b, sh in channel_boundaries(audit, ch_num):
            ax.axvline(mdates.date2num(ord_to_dt(b)), color=INK2, lw=0.6)
        for a, b, rid, act in spans.get(d, []):
            xa = mdates.date2num(ord_to_dt(a))
            ax.plot([xa, xa], [1.0, 1.035], transform=ax.get_xaxis_transform(), color=INK, lw=0.9, clip_on=False)
            if rid in cfg['events']['key_events']:
                ax.text(xa, 1.04, rid, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7, color=INK2)
        ax.set_title(f'{d.capitalize()}', loc='left', pad=14)
        ax.set_ylabel('Chainage (m)')
        ax.grid(False)
    date_axis(axes[-1])
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
        cb.set_label('Indicator anomaly (log units)', color=INK2)
        cb.outline.set_visible(False)
    fig.text(0.01, 0.005, 'Horizontal lines: provisional corridor. Vertical black line: end of calibration. Thin grey lines: '
             'working-regime changes of this channel. Ticks above: scheduled corridor work.', fontsize=7, color=INK2)
    return save_fig(fig, path)


def fig_classification(plt, res_rows, pl_rows, path):
    dirs = sorted({r[0] for r in res_rows})
    if not dirs:
        return None
    fig, axes = plt.subplots(1, len(dirs), figsize=(7.6, 4.8), sharex=True, squeeze=False)
    clfs = sorted({r[2] for r in res_rows}, key=lambda c: ['knn', 'lda', 'rf'].index(c))
    lab = {'knn': 'kNN', 'lda': 'Shrinkage LDA', 'rf': 'Random forest'}
    combos = [('absolute_110', 'random'), ('absolute_110', 'grouped_by_date'), ('absolute_110', 'blocked_in_time'),
              ('relative_110', 'random'), ('relative_110', 'grouped_by_date'), ('relative_110', 'blocked_in_time'),
              ('single_s1_rms', 'random')]
    names = {('absolute_110', 'random'): 'Absolute features, random CV',
             ('absolute_110', 'grouped_by_date'): 'Absolute, grouped by date',
             ('absolute_110', 'blocked_in_time'): 'Absolute, blocked in time',
             ('relative_110', 'random'): 'Relative features, random CV',
             ('relative_110', 'grouped_by_date'): 'Relative, grouped by date',
             ('relative_110', 'blocked_in_time'): 'Relative, blocked in time',
             ('single_s1_rms', 'random'): 'One feature (s1 level), random CV'}
    pl_names = [(v, n) for v, n in (('shifted', 'Placebo, real labels shifted in time'),
                                    ('scattered', 'Placebo blocks, scattered'), ('adjacent', 'Placebo blocks, adjacent'))
                if any(r[1] == v for r in pl_rows)]
    nrow = len(combos) + len(pl_names)
    for ax, d in zip(axes[0], dirs):
        ys = np.arange(nrow)[::-1]
        for j, clf in enumerate(clfs):
            xs = []
            for (fs, sc) in combos:
                v = [r[4] for r in res_rows if r[0] == d and r[1] == fs and r[3] == sc and r[2] == clf]
                xs.append(v[0] if v else np.nan)
            ax.scatter(xs, ys[:len(combos)] + (1 - j) * 0.2, s=28, color=SERIES[j], label=lab[clf], zorder=3,
                       edgecolors='white', linewidths=0.8)
        for i, (variant, nm) in enumerate(pl_names):
            yy = ys[len(combos) + i]
            for j, clf in enumerate(('knn', 'lda')):
                pv = np.array([r[5] for r in pl_rows if r[0] == d and r[1] == variant and r[3] == 'absolute_110' and r[4] == clf])
                if len(pv):
                    lo, med, hi = np.percentile(pv, [5, 50, 95])
                    off = (1 - j) * 0.2
                    ax.plot([lo, hi], [yy + off, yy + off], color=SERIES[j], lw=2, solid_capstyle='round', zorder=2)
                    ax.scatter([med], [yy + off], s=28, color=SERIES[j], zorder=3, edgecolors='white', linewidths=0.8)
        chance = [r[9] for r in res_rows if r[0] == d][0]
        ax.axvline(chance, color=INK2, lw=0.8)
        ax.text(chance + 0.01, -0.9, 'chance', color=INK2, fontsize=7, va='bottom')
        ax.axhline(ys[len(combos) - 1] - 0.5, color=AXIS, lw=0.8)
        ax.set_yticks(ys)
        if ax is axes[0][0]:
            ax.set_yticklabels([names[cmb] for cmb in combos] + [n for _, n in pl_names])
        else:
            ax.set_yticklabels([''] * nrow)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-1.0, nrow - 0.4)
        ax.set_xlabel('Balanced accuracy')
        ax.set_title(f'{d.capitalize()} passages', loc='left')
    axes[0][0].legend(loc='upper center', bbox_to_anchor=(1.0, -0.14), ncol=3)
    return save_fig(fig, path)


def short_h(h):
    return h.replace('next observation day', 'next day').replace('at least ', '').replace(' days ahead', ' days ahead')


def fig_forecast(plt, mrows, keys, path, scope='development'):
    keys = [k for k in keys if any(r[1] == k and r[0] == scope for r in mrows)]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(7.6, 3.3), squeeze=False)
    for i, (ax, key) in enumerate(zip(axes[0], keys)):
        hls = []
        for r in mrows:
            if r[1] == key and r[0] == scope and r[2] not in hls:
                hls.append(r[2])
        x = np.arange(len(hls))
        for j, m in enumerate(MODELS):
            v = [next((r[8] for r in mrows if r[0] == scope and r[1] == key and r[2] == h and r[3] == m), np.nan) for h in hls]
            ax.plot(x + (j - (len(MODELS) - 1) / 2.0) * 0.1, v, marker='o', ms=4.5, lw=0, color=SERIES[j],
                    label=MODEL_LABELS[m], markeredgecolor='white', markeredgewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_h(h).replace(' days ahead', ' d').replace('next day', 'next\nday') for h in hls])
        ax.set_xlim(-0.6, len(hls) - 0.4)
        ax.set_title(key, loc='left')
        if i == 0:
            ax.set_ylabel('Mean absolute error (log units)')
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout()
    return save_fig(fig, path)


def fig_cross_sensor(plt, rows, path):
    if not rows:
        return None
    pairs = []
    for r in rows:
        if (r[0], r[1]) not in pairs:
            pairs.append((r[0], r[1]))
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(pairs) + 1.2))
    ys = np.arange(len(pairs))[::-1]
    for j, kind in enumerate(('daily', 'filtered')):
        for y, (a, b) in zip(ys, pairs):
            rr = [r for r in rows if r[0] == a and r[1] == b and r[2] == kind]
            if not rr:
                continue
            r = rr[0]
            off = 0.15 if j == 0 else -0.15
            ax.plot([r[6], r[7]], [y + off, y + off], color=SERIES[j], lw=2, solid_capstyle='round')
            ax.scatter([r[5]], [y + off], s=28, color=SERIES[j], zorder=3, edgecolors='white', linewidths=0.8,
                       label=('Daily anomaly' if kind == 'daily' else 'Filtered level') if y == ys[0] else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{a} and {b}' for a, b in pairs])
    ax.set_xlabel('Mean Spearman correlation across bins (95% episode interval)')
    ax.set_title('Agreement of anomaly profiles between sensors, evaluation period', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    return save_fig(fig, path)


def fig_events(plt, bin_rows, store, key, cfg, path, sum_rows=()):
    evs = cfg['events']['key_events']
    info = {(r[0], r[1], r[2]): r for r in sum_rows}
    sel = [(rid, d) for rid in evs for d in DIRECTIONS
           if any(r[0] == key and r[1] == rid and r[2] == d for r in bin_rows)]
    if not sel:
        return None
    fig, axes = plt.subplots(len(sel), 1, figsize=(7.2, 2.1 * len(sel) + 0.6), sharex=True, squeeze=False)
    for ax, (rid, d) in zip(axes[:, 0], sel):
        D = store[key]['series'][d]
        ch = D['chainage']
        cor = D['corridor']
        rr = sorted([r for r in bin_rows if r[0] == key and r[1] == rid and r[2] == d], key=lambda r: r[EBH['chainage_start_m']])
        x = np.array([r[EBH['chainage_start_m']] for r in rr], float) + 50
        y = np.array([r[EBH['change']] for r in rr], float)
        sd = np.array([np.nan if r[EBH['null_sd_calibrated']] is None else r[EBH['null_sd_calibrated']] for r in rr], float)
        sg = np.array([r[EBH['fdr_significant']] for r in rr], bool)
        ax.fill_between(x, -1.645 * sd, 1.645 * sd, color=GRID, lw=0, label='90% range expected without an effect')
        if cor.any():
            ax.axvspan(ch[cor].min(), ch[cor].max() + 100, color=PAL['blue'], alpha=0.07, lw=0, label='Provisional corridor')
        ax.plot(x, y, color=PAL['blue'], lw=1.5, marker='o', ms=3, label='Change, after minus before')
        if sg.any():
            ax.scatter(x[sg], y[sg], s=36, color=PAL['orange'], zorder=4, label='FDR-significant bin',
                       edgecolors='white', linewidths=0.8)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel('Change (log units)')
        r = info.get((key, rid, d))
        extra = ''
        if r is not None and r[EH['z']] is not None and np.isfinite(r[EH['z']]):
            extra = (f': corridor minus control {r[EH["corridor_minus_control_change"]]:+.3f}, z {r[EH["z"]]:+.2f}, '
                     f'p {r[EH["p_two_sided"]]:.3f}')
            if r[EH['regime_change_in_window']]:
                extra += '; window contains an instrument regime change'
        ax.set_title(f'{rid} ({d}){extra}', loc='left')
    axes[-1, 0].set_xlabel('Chainage (m)')
    handles, labels = [], []
    for ax in axes[:, 0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_fig(fig, path)


def fig_detection(plt, rows, key, path, m=3, scope='development'):
    sel = [r for r in rows if r[0] == scope and r[1] == key and r[4] == m]
    if not sel:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    Ls = sorted({r[3] for r in sel})
    for j, L in enumerate(Ls):
        rr = sorted([r for r in sel if r[3] == L], key=lambda r: r[5])
        x = np.array([r[7] for r in rr])
        y = np.array([r[8] for r in rr])
        keep = (x >= -60) & (x <= 150)
        ax.plot(x[keep], y[keep], color=SERIES[j], lw=2, label=f'{int(L) * 100} m segment')
    ax.axhline(0.8, color=INK2, lw=0.8)
    ax.text(148, 0.78, '80% detection', color=INK2, fontsize=7, va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0))
    ax.set_xlim(-60, 150)
    ax.set_xlabel('Injected persistent change, amplitude equivalent (%)')
    ax.set_ylabel('Detection probability')
    ax.set_title(f'{key}: 5% alarm burden on unmodified data, {m} observation days after the change', loc='left')
    ax.legend(loc='lower right')
    return save_fig(fig, path)


def fig_priority(plt, comps, cfg, path):
    key = ind_key(*cfg['primary_indicator'])
    st = comps[key]
    pm = cfg['primary_model']
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    for ax, d in zip(axes, DIRECTIONS):
        D = st['series'][d]
        if D['E'].shape[0] == 0:
            continue
        target = int(D['ord'][-1]) + cfg['forecast']['priority_horizon_days']
        mu, var = state_forecast(pm, D['states'][pm], len(D['ord']) - 1, target, np.full(len(D['bins']), 2.0),
                                 st['fits'][pm]['params'], extra_q(st['fits'], pm))
        sd = np.sqrt(var)
        x = D['chainage'] + 50
        cor = D['corridor']
        if cor.any():
            ax.axvspan(D['chainage'][cor].min(), D['chainage'][cor].max() + 100, color=GRID, alpha=0.7, lw=0,
                       label='Provisional corridor')
        ax.fill_between(x, mu - 1.645 * sd, mu + 1.645 * sd, color=PAL['blue'], alpha=0.12, lw=0, label='90% interval')
        ax.plot(x, mu, color=PAL['blue'], lw=2, label='Forecast anomaly')
        lim = priority_limit(cfg)
        ax.axhline(lim, color=PAL['red'], lw=0.9)
        ax.text(0.995, lim, f'review limit (+{pct_change(cfg["forecast"]["priority_limit_log"]):.0f}% amplitude) ',
                color=INK2, fontsize=7, va='bottom', ha='right', transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0))
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel('Anomaly (log)')
        ax.set_title(f'{d.capitalize()}: forecast for {o2d(target)} from the last development day', loc='left')
    axes[-1].set_xlabel('Chainage (m)')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, path)


# ----------------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------------
def html_escape(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def html_table(header, rows, max_rows=40, caption=None):
    out = []
    if caption:
        out.append(f'<p class="cap">{html_escape(caption)}</p>')
    out.append('<div class="tw"><table><thead><tr>' + ''.join(f'<th>{html_escape(h)}</th>' for h in header) + '</tr></thead><tbody>')
    for r in rows[:max_rows]:
        cells = []
        for v in r:
            if isinstance(v, bool) or isinstance(v, np.bool_):
                cells.append(f'<td>{"yes" if v else "no"}</td>')
            elif isinstance(v, (float, np.floating)):
                cells.append(f'<td class="n">{fnum(v, 3)}</td>')
            elif isinstance(v, (int, np.integer)):
                cells.append(f'<td class="n">{int(v)}</td>')
            else:
                cells.append(f'<td>{html_escape(fmt_cell(v))}</td>')
        out.append('<tr>' + ''.join(cells) + '</tr>')
    out.append('</tbody></table></div>')
    if len(rows) > max_rows:
        out.append(f'<p class="note">Showing {max_rows} of {len(rows)} rows; the CSV file holds all rows.</p>')
    return '\n'.join(out)


def html_figure(path, caption):
    if path is None or not Path(path).is_file():
        return ''
    b64 = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html_escape(caption)}"><figcaption>{html_escape(caption)}</figcaption></figure>'


def html_page(title, blocks):
    css = """
    body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#0b0b0b;background:#f9f9f7;margin:0}
    main{max-width:1000px;margin:0 auto;padding:24px 16px 64px;background:#fcfcfb}
    h1{font-size:1.6rem;margin:0.2em 0 0.3em} h2{font-size:1.2rem;margin-top:2em;border-top:1px solid #e1e0d9;padding-top:1em}
    h3{font-size:1rem;margin-top:1.4em} p,li{line-height:1.5;color:#2b2a28} .note{color:#52514e;font-size:0.9rem}
    .cap{font-weight:600;color:#2b2a28;margin:1.2em 0 0.2em;font-size:0.92rem}
    .tw{overflow-x:auto;margin:0.4em 0} table{border-collapse:collapse;font-size:0.82rem}
    th,td{border-bottom:1px solid #e1e0d9;padding:4px 8px;text-align:left;white-space:nowrap}
    th{color:#52514e;font-weight:600} td.n{text-align:right;font-variant-numeric:tabular-nums}
    figure{margin:1em 0} img{max-width:100%;height:auto;border:1px solid #e1e0d9} figcaption{font-size:0.85rem;color:#52514e}
    .key{background:#f0efec;padding:12px 16px;border-radius:6px} code{background:#f0efec;padding:1px 4px;border-radius:3px}
    """
    return ('<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html_escape(title)}</title><style>{css}</style></head><body><main>' + '\n'.join(blocks) + '</main></body></html>')


def read_rows(path):
    """Read one of our CSV outputs back with numeric conversion."""
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        next(r)
        out = []
        for row in r:
            conv = []
            for v in row:
                if v == '':
                    conv.append(None)
                elif v in ('yes', 'no'):
                    conv.append(v == 'yes')
                else:
                    try:
                        conv.append(float(v))
                    except ValueError:
                        conv.append(v)
            out.append(conv)
    return out


def iv(v):
    return int(v) if v is not None else None


def report_tables(out, S, cfg):
    t = {}
    keys = [ind_key(*cfg['primary_indicator'])] + [ind_key(*s) for s in cfg['secondary_indicators']]
    sh = read_rows(out / '01_audit_level_shifts.csv')
    t['shifts'] = (['Channel', 'Shift', 'Last day before', 'First day after', 'Shift (log)', 'Factor'],
                   [[iv(r[0]), iv(r[1]), r[2], r[3], r[4], r[5]] for r in sh])
    dd = read_rows(out / '01_audit_dead_periods.csv')
    t['dead'] = (['Channel', 'First day', 'Last day', 'Days', 'Passages', 'Fail fraction', 'Median level (log)'],
                 [[iv(r[0]), r[1], r[2], iv(r[3]), iv(r[4]), r[5], r[6]] for r in dd])
    ch = S['audit']['channels']
    fs = S['audit']['fingerprint_stability']
    t['channels'] = (['Channel', 'Name', 'Working regimes', 'Largest factor', 'Dead periods', 'Dead days', 'QC pass',
                      'Fingerprint stability (median rho)'],
                     [[c, CHANNEL_NAMES[c], v['working_regimes'], v['largest_shift_factor'], v['dead_periods'],
                       v['dead_period_days'], v['qc_pass_fraction'], (fs.get(c) or {}).get('median')]
                      for c, v in ch.items()])
    cr = read_rows(out / '02_classification_results.csv')
    t['classification'] = (['Direction', 'Feature set', 'Classifier', 'Validation', 'Balanced accuracy', 'SD', 'Repeats', 'Passages', 'Chance'],
                           [[r[0], r[1], r[2], r[3], r[4], r[5], iv(r[6]), iv(r[7]), r[9]] for r in cr])
    pl = []
    for d, v in S.get('classification', {}).get('by_direction', {}).items():
        for k, w in v.items():
            if isinstance(w, dict) and 'placebo_median' in w:
                pl.append([d] + k.split('|') + [w['random_cv'], w['placebo_median'], w['placebo_p05'], w['placebo_p95'],
                                                w['real_percentile_in_placebo'], w['placebo_n'], w.get('placebo_sd'),
                                                w.get('repeat_noise_sd')])
    t['placebo'] = (['Direction', 'Placebo', 'Feature set', 'Classifier', 'Real labels, random CV', 'Placebo median',
                     'Placebo 5th pct', 'Placebo 95th pct', 'Real percentile among placebos', 'Placebos',
                     'Placebo SD', 'Repeat-noise SD'], pl)
    fm = read_rows(out / '03_forecast_metrics.csv')
    t['forecast'] = (['Indicator', 'Horizon', 'Model', 'Bin-days', 'Episodes', 'MAE', 'Skill vs persistence', 'Low', 'High',
                      'Episodes better than persistence', 'Sign test p', 'Skill vs fingerprint', 'Low', 'High',
                      'Episodes better than fingerprint', 'Sign test p', 'Coverage 80%'],
                     [[r[1], short_h(r[2]), r[3], iv(r[4]), iv(r[5]), r[8], r[10], r[11], r[12], f'{iv(r[22])} of {iv(r[23])}',
                       r[24], r[13], r[14], r[15], f'{iv(r[19])} of {iv(r[20])}', r[21], r[17]]
                      for r in fm if r[1] in keys])
    fam = []
    for r in fm:
        if r[3] == cfg['primary_model'] and r[2] == horizon_label(30):
            fam.append([r[1], iv(r[4]), r[8], r[10], r[11], r[12], r[13], r[14], r[15], r[17]])
    t['family'] = (['Indicator', 'Bin-days', 'MAE', 'Skill vs persistence', 'Low', 'High', 'Skill vs fingerprint', 'Low', 'High',
                    'Coverage 80%'], fam)
    rk = read_rows(out / '03_forecast_ranking.csv')
    t['ranking'] = (['Indicator', 'Horizon', 'Model', 'Days', 'Mean Spearman', 'Low', 'High', 'Top-decile precision', 'Low', 'High',
                     'Episodes positive', 'Sign test p'],
                    [[r[1], short_h(r[2]), r[3], iv(r[4]), r[6], r[7], r[8], r[9], r[10], r[11], f'{iv(r[12])} of {iv(r[13])}', r[14]]
                     for r in rk if r[1] in keys])
    kp = read_rows(out / '03_kalman_parameters.csv')
    t['params'] = (['Indicator', 'Model', 'Drift variance per day', 'Reversion time (days)', 'Day noise variance',
                    'Passage noise variance', 'Level change SD over 90 days', 'Single-passage noise SD',
                    'Change to noise (90 days)', 'Speed slope', 'Regime breaks per day'],
                   [[r[0], r[1], r[2], r[4], r[6], r[7], r[10], r[11], r[12], r[13], r[14]] for r in kp])
    qc = read_rows(out / '03_indicator_qc.csv')
    t['qc'] = (['Indicator', 'Direction', 'Calibration passages', 'Passing QC', 'QC pass (all)', 'Days', 'Evaluation days',
                'Anomaly-speed correlation (evaluation)'],
               [[r[0], r[1], iv(r[2]), iv(r[3]), r[4], iv(r[5]), iv(r[6]), r[7]] for r in qc])
    pp = read_rows(out / '03_pseudo_prospective.csv')
    t['pseudo'] = (['Scope', 'Indicator', 'Mode', 'Model', 'Bin-days', 'Episodes', 'MAE', 'Skill vs persistence',
                    'Better than persistence', 'Sign test p', 'Skill vs fingerprint', 'Low', 'High', 'Better than fingerprint',
                    'Sign test p', 'Coverage 80%'],
                   [[r[0].replace('pseudo-prospective ', ''), r[1], r[2], r[3], iv(r[4]), iv(r[5]), r[8], r[10],
                     f'{iv(r[22])} of {iv(r[23])}', r[24], r[13], r[14], r[15], f'{iv(r[19])} of {iv(r[20])}', r[21], r[17]]
                    for r in pp
                    if r[3] in ('fingerprint', 'persistence', 'moving_mean', 'local_level', 'mean_reverting')
                    and (r[0] == 'pseudo-prospective pooled cuts' or r[1] == keys[0])])
    pe = read_rows(out / '03_pseudo_prospective_episodes.csv')
    t['pseudo_episodes'] = (['Scope', 'Mode', 'Model', 'Episode', 'Last day', 'Days', 'MAE', 'Skill vs fingerprint',
                             'Skill vs persistence', 'Coverage 80%', 'Mean Spearman'],
                            [[r[0].replace('pseudo-prospective ', ''), r[2], r[3], r[4], r[6], iv(r[7]), r[9], r[10], r[11],
                              r[12], r[13]] for r in pe if r[3] == cfg['primary_model']])
    brk = read_rows(out / '03_causal_regime_breaks.csv')
    t['breaks'] = (['Indicator', 'First day of new regime', 'Confirmed on'], [[r[0], r[1], r[2]] for r in brk])
    xs = read_rows(out / '04_cross_sensor_agreement.csv')
    t['cross'] = (['Indicator A', 'Indicator B', 'Anomaly', 'Days', 'Episodes', 'Mean Spearman', 'Low', 'High'],
                  [[r[0], r[1], r[2], iv(r[3]), iv(r[4]), r[5], r[6], r[7]] for r in xs])
    ev = read_rows(out / '05_events_summary.csv')
    t['events'] = (['Indicator', 'Record', 'Direction', 'Activity', 'Start', 'Days before', 'Days after',
                    'Corridor minus control', 'Model SD', 'z', 'p', 'q (BH)', 'Placebo windows', 'Independent',
                    'Placebo RMS z', 'Regime change in window', 'FDR bins corridor', 'FDR bins control'],
                   [[r[0], r[1], r[2], r[3], r[4], iv(r[EH['days_before']]), iv(r[EH['days_after']]),
                     r[EH['corridor_minus_control_change']], r[EH['model_sd']], r[EH['z']], r[EH['p_two_sided']],
                     r[EH['q_value_bh']], iv(r[EH['placebo_windows']]), iv(r[EH['independent_placebo_windows']]),
                     r[EH['placebo_rms_z']], r[EH['regime_change_in_window']], iv(r[EH['fdr_bins_corridor']]),
                     iv(r[EH['fdr_bins_control']])]
                    for r in ev if r[EH['z_model']] is not None])
    cp = read_rows(out / '05_events_composite.csv')
    t['composite'] = (['Feature', 'Record', 'Direction', 'Activity', 'Start', 'Channels', 'Change (scaled)', 'Composite z', 'p',
                       'q (BH)', 'Placebo windows', 'Independent', 'Placebo RMS z', 'Channel z', 'Signs agree'],
                      [[r[0], r[1], r[2], r[3], r[4], r[CH['channels']], r[CH['composite_change']], r[CH['composite_z']],
                        r[CH['p_two_sided']], r[CH['q_value_bh']], iv(r[CH['placebo_windows']]),
                        iv(r[CH['independent_placebo_windows']]), r[CH['placebo_rms_z']], r[CH['channel_z']],
                        r[CH['channels_agree_in_sign']]]
                       for r in cp if r[CH['composite_z']] is not None])
    nm = read_rows(out / '05_events_null_models.csv')
    t['nulls'] = (['Indicator', 'Direction', 'Days', 'Days outside work windows', 'Isolated days removed',
                   'Level change SD over 90 days', 'Noise SD'],
                  [[r[0], r[1], iv(r[2]), iv(r[3]), iv(r[5]), r[8], r[9]] for r in nm if r[0] in keys])
    ws = read_rows(out / '05_events_window_sensitivity.csv')
    t['window'] = (['Window (days)', 'Feature', 'Record', 'Direction', 'Channels', 'Composite z', 'p', 'Placebo windows',
                    'Independent', 'Signs agree'],
                   [[iv(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], iv(r[7]), iv(r[8]), r[9]] for r in ws])
    sm = read_rows(out / '01_audit_simultaneous_changes.csv')
    t['simultaneous'] = (['Last day before', 'First day after', 'Channels', 'Factor by channel'],
                         [[r[0], r[1], iv(r[2]), r[3]] for r in sm])
    md = read_rows(out / '06_detection_mdc.csv')
    t['mdc'] = (['Indicator', 'Quantity', 'Segment bins', 'Days after', 'Increase for 80% (amplitude %)',
                 'Decrease for 80% (amplitude %)', 'Threshold'],
                [[r[1], r[2], iv(r[3]), iv(r[4]), r[7], r[10], r[11]] for r in md if r[1] in keys])
    pr = read_rows(out / '07_priority_list.csv')
    t['priority'] = (['Rank', 'Direction', 'Bin', 'Chainage', 'Corridor', 'Last observed', 'Anomaly', 'SD', 'P(exceeds)'],
                     [[iv(r[0]), r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[9]] for r in pr])
    return t


def development_report(outdir, S, figs, tables, cfg, found):
    EV = cfg['events']
    blocks = ['<h1>DR-Train Region 5 study: development results</h1>',
              f'<p class="note">Pipeline version {VERSION}, generated {S["generated_utc"]}, run time {S.get("elapsed_seconds")} s. '
              'Development data only; the 89 holdout passages were not read. '
              f'Feature table SHA-256 <code>{found.sha256["features"][:16]}</code>; configuration SHA-256 <code>{S["config_sha256"][:16]}</code>.</p>']
    blocks.append('<div class="key"><h3>Key results</h3><ul>' + ''.join(f'<li>{html_escape(li)}</li>' for li in S['key_findings']) + '</ul></div>')
    blocks.append('<h2>1. Data audit</h2><p>Each passage level is the median log RMS of a channel after removing the direction fingerprint. '
                  'A passage fails quality control when its spatial profile correlates with the fingerprint below the threshold, which marks '
                  'dead or abnormal channels. Working regimes are found by penalised segmentation of daily median levels of passing passages; '
                  'neighbouring regimes differing by less than a factor of 1.5 are merged.</p>')
    blocks.append(html_figure(figs.get('levels'), 'Figure 1. Channel levels over time with working regimes, failed passages and scheduled corridor work.'))
    blocks.append(html_table(*tables['channels'], caption='Table 1. Channel summary'))
    blocks.append(html_table(*tables['shifts'], caption='Table 2. Level changes between working regimes'))
    blocks.append(html_table(*tables['simultaneous'], caption='Table 2b. Level changes grouped across channels (within 7 days)'))
    blocks.append(html_table(*tables['dead'], caption='Table 3. Dead or abnormal periods'))
    blocks.append('<h2>2. Re-analysis of maintenance classification</h2><p>Classes follow the published design: passes in date windows before '
                  'scheduled tamping, tamping with surfacing, surfacing, and no maintenance. Features are corridor medians of the 110 foundation '
                  'features (absolute) or corridor minus control medians within each passage (relative). Three kinds of placebo labels are '
                  'used. Shifted placebos move the real label pattern (an isolated block and three adjacent blocks) by every possible offset in '
                  'observation days, wrapping round the record, so only the dates change; they are the exact comparison. Scattered and adjacent '
                  'placebos are contiguous blocks of the real class sizes at random positions. If maintenance carried information beyond the '
                  'recording period, the real labels should be classified better than their shifted copies. Placebo values average '
                  f'{cfg["classification"]["placebo_repeats"]} cross-validation repeats against {cfg["classification"]["repeats"]} for the real '
                  'labels; the repeat-noise column shows the extra spread this alone causes.</p>')
    blocks.append(html_figure(figs.get('classification'), 'Figure 2. Balanced accuracy by validation scheme and feature set, and placebo ranges (5th, 50th and 95th percentiles).'))
    blocks.append(html_table(*tables['placebo'], caption='Table 4. Real labels against placebo labels (random cross-validation)'))
    blocks.append(html_table(*tables['classification'], max_rows=60, caption='Table 5. All classification results'))
    blocks.append('<h2>3. Regime-robust indicator and forecasting</h2><p>The indicator is the log feature in each 100 m bin after subtracting the '
                  'passage level (which cancels gain changes), the calibration fingerprint and a speed term. Daily medians are forecast with '
                  'eight models: four baselines and four state-space models (local level, mean-reverting level, regime-aware local level and '
                  'local linear trend). The regime-aware model resets its state when a causal detector confirms a jump in passage levels on '
                  'two consecutive observation days, and its forecasts include the variance of future resets at the calibration break rate. '
                  f'Calibration ends {cfg["calibration_end"]}; all later days are scored with rolling origins. Skill is one minus the ratio of '
                  'mean absolute errors. Calendar episodes (observation days of both directions separated by more than seven days) are the '
                  'independent units: 95% intervals resample episodes, and the sign test counts episodes in which the model beats the '
                  f'baseline. Primary model: {cfg["primary_model"]}.</p>')
    blocks.append(html_figure(figs.get('anomaly'), 'Figure 3. Daily anomaly map of the primary indicator.'))
    blocks.append(html_figure(figs.get('forecast'), 'Figure 4. Forecast errors by model and horizon.'))
    blocks.append(html_table(*tables['forecast'], max_rows=100, caption='Table 6. Forecast accuracy for the primary and secondary indicators'))
    blocks.append(html_table(*tables['family'], caption=f'Table 7. Whole indicator family, {cfg["primary_model"]}, at least 30 days ahead'))
    blocks.append(html_table(*tables['ranking'], max_rows=60, caption='Table 8. Ranking of bins'))
    blocks.append('<p>The holdout-like check below repeats the frozen-protocol design inside development data: for each cut date, '
                  'targets lie 8 to 134 days after the cut, forecast either once from the cut (fixed origin) or from the previous '
                  'observation day (sequential), with models fitted only on the calibration period. The sign-test columns show how the '
                  'frozen decision rule would have decided after each cut.</p>')
    blocks.append(html_table(*tables['pseudo'], max_rows=140, caption='Table 8b. Holdout-like evaluation inside development data '
                             '(pooled over cut dates, and each cut for the primary indicator)'))
    blocks.append(html_table(*tables['pseudo_episodes'], max_rows=80, caption=f'Table 8c. Holdout-like evaluation by calendar episode '
                             f'(primary indicator, {cfg["primary_model"]})'))
    blocks.append(html_table(*tables['params'], max_rows=40, caption='Table 9. Fitted state-space parameters'))
    blocks.append(html_table(*tables['qc'], max_rows=40, caption='Table 9b. Indicator quality control and the speed check (a '
                             'correlation near zero shows that the pooled speed slope removes the speed effect)'))
    blocks.append(html_table(*tables['breaks'], max_rows=60, caption='Table 10. Confirmed causal regime breaks'))
    blocks.append('<h2>4. Agreement between independent sensors</h2><p>Spearman correlation across bins between the anomaly profiles of two '
                  'indicators on the same day. Shared structure between bogie and cabin sensors, which are separate hardware, points to a track '
                  'or operating origin; it does not separate the two.</p>')
    blocks.append(html_figure(figs.get('cross'), 'Figure 5. Agreement of anomaly profiles between sensors.'))
    blocks.append(html_table(*tables['cross'], caption='Table 11. Cross-sensor agreement'))
    blocks.append('<h2>5. Scheduled work: corridor against control bins</h2><p>For each scheduled record the statistic is the change in the '
                  'daily corridor-minus-control contrast: its mean over '
                  f'{EV["window_days"]} days after the work minus its mean over {EV["window_days"]} days before. The null model is a random '
                  'walk plus noise for the contrast, fitted by maximum likelihood on days more than '
                  f'{EV["window_days"]} days from any scheduled work in that direction and restarted at working-regime changes; it gives the '
                  'exact variance of the statistic for the observation days of each window. Isolated days further than '
                  f'{EV["outlier_robust_sd"]:.0f} robust standard deviations from the running median of {2 * EV["outlier_half_window"] + 1} '
                  'observation days are removed first. The model is checked on clean placebo windows of the same duration (no scheduled '
                  'work, no regime change); where their root-mean-square z exceeds one, the standard error is widened by that factor. '
                  'Overlapping placebo windows are not independent, so the number of non-overlapping windows is reported. Records whose '
                  'window contains a regime change are not valid tests. The composite averages the contrasts of exactly the channels that '
                  'give a valid test for that record, each scaled by its noise, and is modelled in the same way.</p>')
    blocks.append(html_figure(figs.get('events'), 'Figure 6. Bin-level change for the best-covered tamping records (primary indicator), with the 90% range expected without an effect.'))
    blocks.append(html_table(*tables['composite'], max_rows=60, caption='Table 12. Multi-channel composite tests'))
    blocks.append(html_table(*tables['window'], max_rows=40, caption='Table 12b. Composite tests for the best-covered records '
                             'with other window lengths (rows absent where too few days or an instrument regime change in the window)'))
    blocks.append(html_table(*tables['events'], max_rows=120, caption='Table 13. Single-indicator tests'))
    blocks.append(html_table(*tables['nulls'], max_rows=40, caption='Table 13b. Null models of the contrast (primary and secondary indicators)'))
    blocks.append('<h2>6. Detection limits</h2><p>Persistent multiplicative steps are added to the forecast errors after each origin. Thresholds '
                  'give a 5% alarm rate on unmodified development data, which is an alarm burden, not a verified false-positive rate. For band power, '
                  'steps are converted to the equivalent amplitude change.</p>')
    blocks.append(html_figure(figs.get('detection'), 'Figure 7. Detection probability against step size.'))
    blocks.append(html_table(*tables['mdc'], max_rows=40, caption='Table 14. Minimum detectable change for 80% detection'))
    blocks.append('<h2>7. End-of-development priority list</h2><p>Models refitted on all development data. Bins are ranked by the probability '
                  'that the forecast anomaly 30 days after the last development day exceeds the review limit. This is a list for engineering '
                  'review, not a maintenance prediction.</p>')
    blocks.append(html_figure(figs.get('priority'), 'Figure 8. Forecast anomaly along the route.'))
    blocks.append(html_table(*tables['priority'], max_rows=15, caption='Table 15. Highest-ranked bins'))
    blocks.append('<h2>Limitations</h2><ul>'
                  '<li>Maintenance records are schedules, not confirmed work, and the corridor is provisional, not the work footprint.</li>'
                  '<li>Absolute GPS and accelerometer synchronisation is unverified; 100 m localisation is therefore approximate.</li>'
                  '<li>Instrument regime changes alter levels and, for some channels, spatial profiles; relative indicators reduce but do not remove this.</li>'
                  '<li>Windows and passages from the same day are dependent; inference uses calendar episodes as units, and few episodes give wide intervals.</li>'
                  '<li>The scheduled-work null assumes a random walk plus noise for the contrast; the placebo check of it rests on few independent windows.</li>'
                  '<li>Synthetic steps are changes in a feature, not simulated track defects.</li>'
                  '<li>All results here are development results; only the frozen holdout evaluation is confirmatory.</li></ul>')
    blocks.append('<h2>Next commands</h2><p>After reading this report: <code>freeze</code> (then register the protocol SHA-256 '
                  'externally), <code>verify-extraction</code>, <code>extract-holdout --confirm-single-use</code> and '
                  '<code>score-holdout</code>. Every step is recorded in <code>PROTOCOL_LEDGER.jsonl</code>.</p>')
    write_text(outdir / 'DEVELOPMENT_REPORT.html', html_page('DR-Train study development results', blocks))


# ----------------------------------------------------------------------------
# Command: develop
# ----------------------------------------------------------------------------
def build_key_findings(S, cfg):
    out = []
    A = S['audit']['channels']
    parts = []
    for c, v in A.items():
        if v['largest_shift_between']:
            parts.append(f's{c} factor {v["largest_shift_factor"]:.1f} ({v["largest_shift_between"][0]} to {v["largest_shift_between"][1]})')
    if parts:
        out.append('Instrument level regimes (largest change between working regimes): ' + '; '.join(parts) + '.')
    for r in S['audit'].get('simultaneous_changes', []):
        out.append(f'System-wide change: between {r[0]} and {r[1]}, {r[2]} channels changed level together ({r[3]}; factor = after over before).')
    out.append('Dead or abnormal periods and quality-control pass rate: ' + ', '.join(
        f's{c} {v["dead_period_days"]} days, {100 * v["qc_pass_fraction"]:.0f}% pass' for c, v in A.items()) + '.')
    fs = S['audit']['fingerprint_stability']
    out.append('Fingerprint stability across working regimes (median Spearman): ' + ', '.join(
        f's{c} {v["median"]:.2f}' for c, v in fs.items() if v['median'] is not None) + '.')
    cl = S.get('classification', {}).get('by_direction', {})
    for d, v in cl.items():
        sh = v.get('shifted|absolute_110|knn')
        sc = v.get('scattered|absolute_110|knn')
        ad = v.get('adjacent|absolute_110|knn')
        base = sh or sc or ad
        if not base:
            continue
        txt = (f'Classification ({d}, absolute features, kNN): random CV {base["random_cv"]:.2f}, grouped by date '
               f'{base["grouped_cv"]:.2f}, blocked in time {base["blocked_cv"]:.2f}.')
        if sh:
            txt += (f' Time-shifted copies of the real labels: median {sh["placebo_median"]:.2f} over {sh["placebo_n"]} shifts '
                    f'(real labels at the {ordinal(100 * sh["real_percentile_in_placebo"])} percentile).')
        extra = []
        if sc:
            extra.append(f'scattered median {sc["placebo_median"]:.2f}')
        if ad:
            extra.append(f'adjacent median {ad["placebo_median"]:.2f}')
        if extra:
            txt += ' Placebo blocks of the real sizes: ' + ', '.join(extra) + '.'
        out.append(txt + f' Chance {v["chance"]:.2f}.')
        rel = v.get('shifted|relative_110|lda')
        if rel and rel.get('real_percentile_in_placebo') is not None:
            out.append(f'Classification ({d}, regime-robust relative features, shrinkage LDA): random CV {rel["random_cv"]:.2f}, '
                       f'grouped by date {rel["grouped_cv"]:.2f}, blocked in time {rel["blocked_cv"]:.2f}; time-shifted copies '
                       f'median {rel["placebo_median"]:.2f} (real labels at the {ordinal(100 * rel["real_percentile_in_placebo"])} '
                       f'percentile, random cross-validation only).')
    fc = S.get('forecast', {})
    prim = fc.get('primary')
    pm = cfg['primary_model']
    if prim and prim in fc.get('key_indicators', {}):
        for hl, mv in fc['key_indicators'][prim].items():
            ll = mv.get(pm)
            if ll:
                sk, sf = ll['skill_vs_persistence'], ll['skill_vs_fingerprint']
                eb = ll.get('episodes_better_than_fingerprint') or [None, None, None]
                ebp = ll.get('episodes_better_than_persistence') or [None, None, None]
                ep_txt = (f'; better than persistence in {int(ebp[0])} and than the fingerprint in {int(eb[0])} of '
                          f'{int(eb[1])} calendar episodes' if eb[0] is not None and eb[1] and ebp[0] is not None else '')
                out.append(f'Forecasting {prim} with {pm}, {hl}: skill against persistence {sk[0]:.2f} ({sk[1]:.2f} to {sk[2]:.2f}), '
                           f'against the static fingerprint {sf[0]:.2f} ({sf[1]:.2f} to {sf[2]:.2f}){ep_txt}; '
                           f'80% interval coverage {ll["coverage80"]:.2f}.')
    rk = fc.get('ranking', {})
    k30 = f'{prim}|{horizon_label(30)}|{pm}'
    if k30 in rk:
        out.append(f'Ranking of bins 30 or more days ahead ({prim}, {pm}): mean Spearman {rk[k30][0]:.2f} '
                   f'({rk[k30][1]:.2f} to {rk[k30][2]:.2f}); top-decile precision {rk[k30][3]:.2f} against 0.10 by chance.')
    ppk = fc.get('pseudo_prospective', {})
    for mode in ('sequential', 'fixed origin'):
        k = f'{prim}|{mode}|{pm}'
        if k in ppk:
            v = ppk[k]
            sf = v['skill_vs_fingerprint']
            eb = v.get('episodes_better_than_fingerprint') or [None, None, None]
            ep_txt = f'; better in {int(eb[0])} of {int(eb[1])} calendar episodes' if eb[0] is not None and eb[1] else ''
            out.append(f'Holdout-like check inside development data ({prim}, {pm}, {mode}, pooled over four cut dates): '
                       f'skill against the static fingerprint {sf[0]:.2f} ({sf[1]:.2f} to {sf[2]:.2f}){ep_txt}; '
                       f'80% interval coverage {v["coverage80"]:.2f}.')
    rule = [x for x in fc.get('pseudo_prospective_rule', []) if x['mode'] == 'sequential' and x.get('episodes')]
    if rule:
        out.append('Holdout-like check by cut date (sequential forecasts, calendar episodes after each cut): '
                   + '; '.join(f'{x["cut"]}: better than persistence in {int(x["vs_persistence"]["episodes_better"])} of '
                               f'{int(x["episodes"])} episodes and than the fingerprint in {int(x["vs_fingerprint"]["episodes_better"])}'
                               for x in rule)
                   + '. A cut with fewer than five episodes cannot reach p below 0.05 by construction.')
    kv = fc.get('key_indicators', {}).get(prim, {}).get(horizon_label('next'), {}).get(pm) if prim else None
    if kv and kv.get('episodes_better_than_persistence') and kv.get('episodes_better_than_fingerprint'):
        kP, nP = int(kv['episodes_better_than_persistence'][0]), int(kv['episodes_better_than_persistence'][1])
        kF, nF = int(kv['episodes_better_than_fingerprint'][0]), int(kv['episodes_better_than_fingerprint'][1])
        pP, pF = (kP + 0.5) / (nP + 1), (kF + 0.5) / (nF + 1)
        hP, hF = ('H1', 'H2') if comparison_order(cfg)[0] == 'persistence' else ('H2', 'H1')
        out.append(f'Estimated power of the holdout decision rule with five episodes (all five must favour the model), from the '
                   f'rolling-origin development episodes: {hP} against persistence about {pP ** 5:.2f} (model better in {kP} of '
                   f'{nP} episodes); {hF} against the static fingerprint about {pF ** 5:.2f} (better in {kF} of {nF}).')
        S.setdefault('forecast', {})['holdout_power'] = {'vs_persistence': pP ** 5, 'vs_fingerprint': pF ** 5,
                                                         'episodes_vs_persistence': [kP, nP], 'episodes_vs_fingerprint': [kF, nF]}
    xs = S.get('cross_sensor', {})
    if xs:
        out.append('Cross-sensor agreement of filtered anomalies (mean Spearman): ' + ', '.join(
            f'{k.split("|")[0]} with {k.split("|")[1]} {v[0]:.2f}' for k, v in xs.items() if k.endswith('filtered')) + '.')
    ev = S.get('events', {})
    if ev:
        n = ev['valid_tests_all']
        out.append(f'Scheduled work: {n} valid single-indicator tests across the indicator family (no regime change in the '
                   f'window); {ev["valid_q_below_fdr_all"]} with a Benjamini-Hochberg q of {cfg["events"]["fdr_q"]:.2f} or less, '
                   f'{ev["valid_p_below_0_05_all"]} with p below 0.05 (about {0.05 * n:.1f} expected by chance), '
                   f'{ev["valid_negative_all"]} with a relative decrease in the corridor.')
        if ev.get('placebo_rms_median') is not None:
            out.append(f'Check of the scheduled-work null on clean placebo windows: median root-mean-square z '
                       f'{ev["placebo_rms_median"]:.2f}; the standard error was widened for '
                       f'{100 * ev["placebo_rms_above_one_fraction"]:.0f}% of the tests.')
        comp = ev.get('composite', {})
        if comp:
            out.append('Multi-channel composite tests for the best-covered records: ' + '; '.join(
                f'{k.replace("|", " ")} ({v["channels"]}) z {v["z"]:+.2f}, p {v["p"]:.3f}, q {v["q"]:.3f}, channels '
                f'{"agree in sign" if v["signs_agree"] else "disagree in sign"}' for k, v in comp.items()) + '.')
        wsr = ev.get('window_sensitivity', [])
        lowband = [r for r in wsr if r[1] == 'bandpower_0p5_5hz']
        if lowband:
            out.append('Window sensitivity, 0.5 to 5 Hz composite: ' + '; '.join(
                f'{r[2]} {r[3]} {int(r[0])} days z {r[5]:+.2f} (p {r[6]:.3f}, signs {"agree" if r[9] else "mixed"})'
                for r in sorted(lowband, key=lambda r: (r[2], r[3], r[0]))) + '.')
    det = S.get('detection', {})
    key = f'{ind_key(*cfg["primary_indicator"])}|L5|m3'
    if key in det:
        inc, dec = det[key]['increase_amplitude_percent'], det[key]['decrease_amplitude_percent']
        if inc is not None and dec is not None and np.isfinite(inc) and np.isfinite(dec):
            out.append(f'Detection limit ({ind_key(*cfg["primary_indicator"])}, 500 m, 3 days after the change, 5% alarm burden): '
                       f'about +{inc:.0f}% or {dec:.0f}% in amplitude for 80% detection.')
    pr = S.get('priority')
    if pr:
        out.append(f'Priority list: {pr["bins_with_prob_above_0_5"]} bins have a probability above 0.5 of exceeding +{pr["limit_percent"]:.0f}% '
                   f'in amplitude 30 days after the last development day.')
    return out


def cmd_develop(args, cfg=None):
    t0 = time.monotonic()
    cfg = copy.deepcopy(cfg or CONFIG)
    if getattr(args, 'quick', False):
        cfg = apply_quick(cfg)
    validate_config(cfg)
    out = Path(args.output_dir) / 'develop'
    if out.exists():
        if not args.replace:
            raise RuntimeError(f'{out} exists; add --replace to overwrite the development results only.')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = np.random.default_rng(cfg['seed'])
    found = Foundation(args.foundation_dir, allow_count_mismatch=getattr(args, 'allow_count_mismatch', False))
    S = OrderedDict()
    S['generated_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    S['pipeline_version'] = VERSION
    S['pipeline_sha256'] = this_file_sha256()
    S['config_sha256'] = config_hash(cfg)
    S['inputs_sha256'] = found.sha256
    S['python'] = platform.python_version()
    S['numpy'] = np.__version__
    timing = OrderedDict()

    def timed(name, fn):
        t = time.monotonic()
        r = fn()
        timing[name] = round(time.monotonic() - t, 1)
        return r

    audit = timed('audit', lambda: run_audit(found, out, cfg, S))
    res_rows, pl_rows = timed('classification', lambda: run_classification(found, out, cfg, S, audit, rng))
    store = timed('forecasting', lambda: run_forecasting(found, out, cfg, S, rng))
    cross_rows = timed('cross_sensor', lambda: run_cross_sensor(store, out, cfg, S, rng))
    ev_sum, ev_bins, ev_comp = timed('events', lambda: run_events(found, store, audit, out, cfg, S))
    timed('event_window_sensitivity', lambda: run_event_window_sensitivity(found, store, audit, out, cfg, S, ev_comp))
    det_rows, mdc_rows = timed('detection', lambda: run_detection(store, out, cfg, S))
    inds = [cfg['primary_indicator']] + cfg['secondary_indicators']
    comps = timed('full_development_fit', lambda: fit_full_components(found, cfg, inds))
    timed('priority', lambda: run_priority(comps, out, cfg, S))
    S['timing_seconds'] = timing
    S['key_findings'] = build_key_findings(S, cfg)
    figs = {}
    plt = get_plt()
    prim = ind_key(*cfg['primary_indicator'])
    if plt is not None:
        spans = found.schedule_spans()
        fdir = out / 'figures'
        for name, fn in [
            ('levels', lambda: fig_levels(plt, audit, spans, fdir / 'fig1_channel_levels.png')),
            ('classification', lambda: fig_classification(plt, res_rows, pl_rows, fdir / 'fig2_classification.png')),
            ('anomaly', lambda: fig_anomaly_map(plt, store, prim, spans, audit, cfg, fdir / 'fig3_anomaly_map.png')),
            ('forecast', lambda: fig_forecast(plt, read_rows(out / '03_forecast_metrics.csv'), [ind_key(*i) for i in inds],
                                              fdir / 'fig4_forecast_errors.png')),
            ('cross', lambda: fig_cross_sensor(plt, cross_rows, fdir / 'fig5_cross_sensor.png')),
            ('events', lambda: fig_events(plt, ev_bins, store, prim, cfg, fdir / 'fig6_events.png', ev_sum)),
            ('detection', lambda: fig_detection(plt, det_rows, prim, fdir / 'fig7_detection.png')),
            ('priority', lambda: fig_priority(plt, comps, cfg, fdir / 'fig8_priority.png')),
        ]:
            try:
                figs[name] = fn()
            except Exception as exc:
                log(f'Figure {name} failed: {type(exc).__name__}: {exc}')
    tables = report_tables(out, S, cfg)
    S['elapsed_seconds'] = round(time.monotonic() - t0, 1)
    write_json(out / 'study_summary.json', {'summary': S, 'config': cfg})
    development_report(out, S, figs, tables, cfg, found)
    for li in S['key_findings']:
        log('KEY ' + li)
    log(f'Module timing (s): {dict(timing)}')
    log(f'Development stage complete in {S["elapsed_seconds"]} s. Report: {out / "DEVELOPMENT_REPORT.html"}')
    write_text(out / 'run_log.txt', '\n'.join(LOG_LINES) + '\n')
    ledger_append(args.output_dir, 'develop', quick_mode=bool(cfg.get('quick_mode')), config_sha256=S['config_sha256'],
                  inputs_sha256=found.sha256, summary_sha256=sha256_file(out / 'study_summary.json'))
    return S


# ----------------------------------------------------------------------------
# Command: freeze
# ----------------------------------------------------------------------------
def protocol_paths(output_dir):
    out = Path(output_dir)
    return out / 'FROZEN_PROTOCOL.json', out / 'FROZEN_PROTOCOL.sha256'


def holdout_paths(output_dir):
    out = Path(output_dir)
    return {'features': out / 'holdout' / 'holdout_vibration_features.csv.gz',
            'partial': out / 'holdout' / 'holdout_vibration_features.csv.gz.partial',
            'log': out / 'holdout' / 'holdout_extraction_log.json',
            'results': out / 'holdout_results'}


def ledger_has(output_dir, event):
    return any(r.get('event') == event for r in ledger_read(output_dir))


def cmd_freeze(args, cfg=None):
    cfg = copy.deepcopy(cfg or CONFIG)
    validate_config(cfg)
    out = Path(args.output_dir)
    summ_path = out / 'develop' / 'study_summary.json'
    if not summ_path.is_file():
        raise RuntimeError('Run the develop command first and read its report.')
    hp = holdout_paths(out)
    if (hp['features'].exists() or hp['partial'].exists() or hp['log'].exists()
            or ledger_has(out, 'extract-holdout started') or ledger_has(out, 'extract-holdout completed')):
        raise RuntimeError('Holdout extraction has started or finished for this output folder. '
                           'The protocol cannot be frozen again after holdout extraction.')
    pf, ph = protocol_paths(out)
    replacing = pf.exists()
    if replacing and not args.replace:
        raise RuntimeError(f'{pf} exists; add --replace to freeze again (allowed only before holdout extraction).')
    dev = json.loads(summ_path.read_text(encoding='utf-8'))
    dev_cfg = dev['config']
    allow_quick = getattr(args, 'allow_quick', False)
    if dev_cfg.get('quick_mode') and not allow_quick:
        raise RuntimeError('The development results were produced in quick mode. Re-run develop without --quick before freezing.')
    base_cfg = copy.deepcopy(dev_cfg)
    base_cfg.pop('quick_mode', None)
    cmp_cfg = copy.deepcopy(cfg)
    cmp_cfg.pop('quick_mode', None)
    if config_hash(base_cfg) != config_hash(cmp_cfg):
        raise RuntimeError('Current settings differ from those used for the development results. Re-run develop first.')
    if dev['summary'].get('pipeline_sha256') != this_file_sha256():
        raise RuntimeError('This pipeline file differs from the one that produced the development results. Re-run develop first.')
    found = Foundation(args.foundation_dir, allow_count_mismatch=getattr(args, 'allow_count_mismatch', False))
    if dev['summary'].get('inputs_sha256') != found.sha256:
        raise RuntimeError('Foundation files differ from those used for the development results.')
    log('Freezing: fitting the final models on all development data')
    inds = [cfg['primary_indicator']] + cfg['secondary_indicators']
    comps = fit_full_components(found, cfg, inds)
    protocol = OrderedDict()
    protocol['protocol'] = 'DR-Train LRV4306 Region 5 prospective evaluation'
    protocol['created_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    protocol['pipeline_version'] = VERSION
    protocol['pipeline_sha256'] = this_file_sha256()
    protocol['expected_builder_sha256'] = EXPECTED_BUILDER_SHA256
    protocol['inputs_sha256'] = found.sha256
    protocol['development_summary_sha256'] = sha256_file(summ_path)
    protocol['config'] = cfg
    protocol['config_sha256'] = config_hash(cfg)
    protocol['development_last_date'] = found.dev_last_date
    protocol['embargo_end'] = found.embargo_end
    protocol['holdout_passage_ids'] = found.holdout_ids
    protocol['holdout_id_sha256'] = found.holdout_hash
    protocol['primary_indicator'] = ind_key(*cfg['primary_indicator'])
    protocol['primary_model'] = cfg['primary_model']
    protocol['secondary_indicators'] = [ind_key(*s) for s in cfg['secondary_indicators']]
    protocol['decision_alpha'] = cfg['forecast']['decision_alpha']
    protocol['plan'] = cfg['holdout_plan']
    protocol['development_key_findings'] = dev['summary'].get('key_findings')
    protocol['indicators'] = OrderedDict()
    for key, st in comps.items():
        D0 = next(iter(st['series'].values()))
        protocol['indicators'][key] = {'model': st['model'], 'fits': st['fits'], 'alarm_thresholds': st['thresholds'],
                                       'development_break_days': [o2d(b) for b in D0['break_days']],
                                       'development_break_confirm_days': [o2d(b) for b in D0['break_confirm_days']]}
    text = json.dumps(to_jsonable(protocol), indent=2, allow_nan=False) + '\n'
    write_text(pf, text)
    digest = sha256_text(text)
    write_text(ph, digest + '  FROZEN_PROTOCOL.json\n')
    ledger_append(out, 'freeze', protocol_sha256=digest, config_sha256=protocol['config_sha256'],
                  development_summary_sha256=protocol['development_summary_sha256'], replaced_previous_protocol=replacing)
    log(f'Protocol frozen: {pf} (SHA-256 {digest}).')
    log('Before extraction, register this SHA-256 outside this folder (for example an OSF or Zenodo registration, or an '
        'e-mail to a co-author), and pass the reference to extract-holdout with --registration.')
    log('Next: verify-extraction, then extract-holdout --confirm-single-use, then score-holdout.')
    write_text(out / 'freeze_log.txt', '\n'.join(LOG_LINES) + '\n')
    return protocol


def load_protocol(output_dir):
    pf, ph = protocol_paths(output_dir)
    if not pf.is_file() or not ph.is_file():
        raise RuntimeError('No frozen protocol found. Run develop and freeze first.')
    text = pf.read_text(encoding='utf-8')
    expected = ph.read_text(encoding='utf-8').split()[0]
    if sha256_text(text) != expected:
        raise RuntimeError('FROZEN_PROTOCOL.json does not match its recorded SHA-256. It has been altered; stop.')
    return json.loads(text), expected


# ----------------------------------------------------------------------------
# Raw extraction through the foundation builder (verification and holdout)
# ----------------------------------------------------------------------------
def load_builder(path, allow_other=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Foundation builder not found: {path}')
    sha = sha256_file(path)
    if sha != EXPECTED_BUILDER_SHA256 and not allow_other:
        raise RuntimeError(f'Builder SHA-256 {sha} differs from the verified version {EXPECTED_BUILDER_SHA256}.')
    spec = importlib.util.spec_from_file_location('pm_foundation_builder', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


def extract_passage(B, rec, data_root, tracks):
    """Same per-passage processing as the foundation builder (without duplicate screening)."""
    pid = rec['passage_id']
    direction = rec['direction']
    g = B.read_gps(data_root / rec['vehicle'] / 'gps_data' / rec['gps_file'])
    apaths = [data_root / rec['vehicle'] / 'accelerometer_data' / rec[f'acc_sensor_{j}_file'] for j in range(1, 6)]
    lens = [B.acc_length(p) for p in apaths]
    windows, gps_dt = B.build_windows(rec, g, tracks[direction], lens)
    rows = []
    failures = 0
    for v in windows:
        r = {k: v.get(k, '') for k in B.META}
        r.update(passage_id=pid, vehicle='LRV4306', region=5, direction=direction, metadata_utc=rec['metadata_utc'],
                 local_date=rec['local_date'], gps_metadata_start_difference_s=gps_dt, n_sensors_with_features=0,
                 raw_acc_time_alignment='assumed_common_first_last_samples_unverified',
                 analysis_screening_flag='speed_screen_pass' if v['estimated_speed_mps'] >= B.MIN_SPEED_MPS else 'review_low_speed')
        rows.append(r)
    for sensor_id, ap in enumerate(apaths, 1):
        x = B.load_acc(ap)
        for v, r in zip(windows, rows):
            a, b = B.sensor_window(v['ia'], v['ib'], lens[0], len(x))
            try:
                vals = B.feature_vector(x[a:b])
                for name, val in zip(B.FEATURES, vals):
                    r[f's{sensor_id}_{name}'] = round(float(val), 9)
                r['n_sensors_with_features'] += 1
            except (ValueError, OverflowError, FloatingPointError):
                failures += 1
        del x
    complete = [r for r in rows if r['n_sensors_with_features'] == 5]
    return complete, g, failures


def acc_digest(B, rec, data_root):
    h = hashlib.sha256()
    for ch in range(1, 6):
        p = data_root / rec['vehicle'] / 'accelerometer_data' / rec[f'acc_sensor_{ch}_file']
        h.update(B.array_digest(B.load_acc(p)).encode('ascii'))
    return h.hexdigest()


def cmd_verify_extraction(args):
    out = Path(args.output_dir)
    B, bsha = load_builder(args.builder, args.allow_builder_mismatch)
    found = Foundation(args.foundation_dir, allow_count_mismatch=getattr(args, 'allow_count_mismatch', False))
    data_root = Path(args.data_root)
    cat = B.catalogue(data_root / 'LRV4306' / 'LRV4306_metadata.csv', data_root)
    recs = {r['passage_id']: r for r in cat}
    t = found.table
    first = {}
    for p, o, d in zip(t.s['passage_id'].tolist(), t.ord.tolist(), t.s['direction'].tolist()):
        first.setdefault(p, (o, d))
    chosen = []
    for d in DIRECTIONS:
        ps = sorted([p for p, v in first.items() if v[1] == d], key=lambda p: (first[p][0], p))
        if not ps:
            continue
        k = max(1, args.n_passages // 2)
        idx = np.unique(np.linspace(0, len(ps) - 1, k).round().astype(int))
        chosen += [ps[i] for i in idx]
    tracks = {d: B.make_track(data_root, d) for d in DIRECTIONS}
    report = []
    all_ok = True
    num_cols = [c for c in B.FIELDS if c not in STRING_COLUMNS and c in t.x]
    for pid in chosen:
        if pid not in recs:
            report.append({'passage_id': pid, 'status': 'not in catalogue'})
            all_ok = False
            continue
        rows, _, failures = extract_passage(B, recs[pid], data_root, tracks)
        m = t.s['passage_id'] == pid
        ref_bins = {b: i for b, i in zip(t.s['track_bin_id'][m].tolist(), np.where(m)[0].tolist())}
        new_bins = {r['track_bin_id']: r for r in rows}
        missing = sorted(set(ref_bins) - set(new_bins))
        extra = sorted(set(new_bins) - set(ref_bins))
        max_diff, n_diff, flag_diff = 0.0, 0, 0
        for b in set(ref_bins) & set(new_bins):
            i = ref_bins[b]
            r = new_bins[b]
            for c in num_cols:
                v = r.get(c, '')
                try:
                    nv = float(v) if v not in ('', None) else float('nan')
                except (TypeError, ValueError):
                    nv = float('nan')
                rv = float(t.x[c][i])
                if not (np.isnan(nv) and np.isnan(rv)):
                    dd = abs(nv - rv)
                    if not np.isfinite(dd) or dd > 0:
                        n_diff += 1
                        max_diff = max(max_diff, dd if np.isfinite(dd) else float('inf'))
            if r['analysis_screening_flag'] != t.s['analysis_screening_flag'][i] or r['corridor_candidate_bin'] != t.s['corridor_candidate_bin'][i]:
                flag_diff += 1
        ok = not missing and not extra and n_diff == 0 and flag_diff == 0
        all_ok &= ok
        report.append({'passage_id': pid, 'direction': recs[pid]['direction'], 'local_date': recs[pid]['local_date'],
                       'windows_reference': len(ref_bins), 'windows_new': len(new_bins), 'missing_bins': missing,
                       'extra_bins': extra, 'values_differing': n_diff, 'max_abs_difference': max_diff,
                       'flag_differences': flag_diff, 'feature_failures': failures, 'status': 'PASS' if ok else 'FAIL'})
        log(f'  {pid}: {"PASS" if ok else "FAIL"} ({len(new_bins)} windows, {n_diff} differing values)')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'builder_sha256': bsha,
              'pipeline_sha256': this_file_sha256(), 'passages': report, 'overall': 'PASS' if all_ok else 'FAIL'}
    write_json(out / 'verification' / 'extraction_check.json', result)
    ledger_append(out, 'verify-extraction', overall=result['overall'], builder_sha256=bsha,
                  passages=[r.get('passage_id') for r in report], allow_builder_mismatch=bool(args.allow_builder_mismatch))
    log(f'Extraction verification {result["overall"]}: {out / "verification" / "extraction_check.json"}')
    if not all_ok:
        raise SystemExit(3)
    return result


def write_extraction_record(output_dir, info):
    """Write the extraction log and its ledger entry; scoring checks both against the files."""
    hp = holdout_paths(output_dir)
    write_json(hp['log'], info)
    ledger_append(output_dir, 'extract-holdout completed', protocol_sha256=info['protocol_sha256'],
                  output_sha256=info['output_sha256'], accepted_passages=len(info['accepted_passage_ids']),
                  accepted_ids_sha256=info['accepted_ids_sha256'], extraction_log_sha256=sha256_file(hp['log']),
                  flags=info.get('flags'), protocol_amendment=info.get('protocol_amendment'),
                  registration=info.get('registration'))


def cmd_extract_holdout(args):
    if not args.confirm_single_use:
        raise RuntimeError('Holdout extraction is single use. Add --confirm-single-use once the protocol is frozen.')
    out = Path(args.output_dir)
    protocol, psha = load_protocol(out)
    hp = holdout_paths(out)
    final = hp['features']
    if final.exists() or hp['log'].exists() or ledger_has(out, 'extract-holdout completed'):
        raise RuntimeError(f'Holdout features were already extracted for {out}. Holdout extraction is not repeated.')
    cur = this_file_sha256()
    amendment = getattr(args, 'protocol_amendment', None)
    if cur != protocol['pipeline_sha256'] and not amendment:
        raise RuntimeError('This pipeline file differs from the frozen one. Restore it, or state a documented '
                           '--protocol-amendment "reason" (recorded in the ledger and the results).')
    flags = OrderedDict([('skip_verification_check', bool(args.skip_verification_check)),
                         ('allow_holdout_manifest_mismatch', bool(args.allow_holdout_manifest_mismatch)),
                         ('allow_builder_mismatch', bool(args.allow_builder_mismatch)),
                         ('allow_count_mismatch', bool(getattr(args, 'allow_count_mismatch', False)))])
    ver = out / 'verification' / 'extraction_check.json'
    ver_status = json.loads(ver.read_text(encoding='utf-8')).get('overall') if ver.is_file() else 'missing'
    if ver_status != 'PASS' and not args.skip_verification_check:
        raise RuntimeError('Run verify-extraction first; it must report PASS.')
    B, bsha = load_builder(args.builder, args.allow_builder_mismatch)
    found = Foundation(args.foundation_dir, allow_count_mismatch=getattr(args, 'allow_count_mismatch', False))
    if found.sha256 != protocol['inputs_sha256']:
        raise RuntimeError('Foundation files differ from those recorded in the frozen protocol.')
    data_root = Path(args.data_root)
    cat = B.catalogue(data_root / 'LRV4306' / 'LRV4306_metadata.csv', data_root)
    complete = [r for r in cat if r['complete_file_set']]
    cutoff = date.fromisoformat(protocol['development_last_date'])
    embargo = date.fromisoformat(protocol['embargo_end'])
    frozen = [r for r in complete if date.fromisoformat(r['local_date']) > embargo]
    ids = sorted(r['passage_id'] for r in frozen)
    h = sha256_text('\n'.join(ids))
    if h != protocol['holdout_id_sha256'] or ids != sorted(protocol['holdout_passage_ids']):
        raise RuntimeError(f'Catalogue holdout set ({len(ids)} identifiers, {h}) differs from the frozen protocol. No signals read.')
    if not args.allow_holdout_manifest_mismatch and getattr(B, 'EXPECTED_HOLDOUT_ID_SHA256', h) != h:
        raise RuntimeError('Holdout digest differs from the foundation builder constant.')
    ledger_append(out, 'extract-holdout started', protocol_sha256=psha, flags=flags, verification=ver_status,
                  protocol_amendment=amendment, registration=getattr(args, 'registration', None), builder_sha256=bsha)
    dev_cands = [r for r in complete if date.fromisoformat(r['local_date']) <= cutoff]
    log(f'Holdout extraction: {len(frozen)} frozen passages; screening GPS duplicates against {len(dev_cands)} development candidates')
    seen = {}
    for r in dev_cands:
        try:
            g = B.read_gps(data_root / r['vehicle'] / 'gps_data' / r['gps_file'])
        except (OSError, ValueError, KeyError) as exc:
            log(f'  development GPS unreadable for {r["passage_id"]}: {exc}')
            continue
        seen.setdefault((r['direction'], B.array_digest(g)), ('development', r))
    tracks = {d: B.make_track(data_root, d) for d in DIRECTIONS}
    tmp = hp['partial']
    final.parent.mkdir(parents=True, exist_ok=True)
    reasons = defaultdict(int)
    excluded = []
    accepted_ids = []
    n_rows = 0
    frozen.sort(key=lambda r: (r['metadata_utc'], r['direction'], r['passage_id']))
    try:
        with gzip.open(tmp, 'wt', newline='', encoding='utf-8', compresslevel=5) as f:
            w = csv.DictWriter(f, fieldnames=B.FIELDS)
            w.writeheader()
            for i, r in enumerate(frozen, 1):
                pid = r['passage_id']
                try:
                    g = B.read_gps(data_root / r['vehicle'] / 'gps_data' / r['gps_file'])
                    key = (r['direction'], B.array_digest(g))
                    if key in seen:
                        src, other = seen[key]
                        same = acc_digest(B, r, data_root) == acc_digest(B, other, data_root)
                        why = (f'confirmed_duplicate_of_{src}' if same else f'same_gps_as_{src}_different_sensors_review')
                        reasons[why] += 1
                        excluded.append({'passage_id': pid, 'reason': why, 'other': other['passage_id']})
                        continue
                    seen[key] = ('holdout', r)
                    rows, _, failures = extract_passage(B, r, data_root, tracks)
                    for row in rows:
                        w.writerow(row)
                    n_rows += len(rows)
                    if rows:
                        accepted_ids.append(pid)
                    else:
                        reasons['no_complete_windows'] += 1
                        excluded.append({'passage_id': pid, 'reason': 'no_complete_windows'})
                except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
                    why = type(exc).__name__ + ':' + str(exc)[:90]
                    reasons[why] += 1
                    excluded.append({'passage_id': pid, 'reason': why})
                if i % 10 == 0 or i == len(frozen):
                    log(f'  {i}/{len(frozen)} holdout passages processed; accepted {len(accepted_ids)}, windows {n_rows}')
        os.replace(tmp, final)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    accepted_ids = sorted(accepted_ids)
    info = OrderedDict([
        ('created_utc', datetime.now(timezone.utc).isoformat(timespec='seconds')), ('protocol_sha256', psha),
        ('pipeline_sha256', cur), ('frozen_pipeline_sha256', protocol['pipeline_sha256']),
        ('protocol_amendment', amendment), ('registration', getattr(args, 'registration', None)),
        ('flags', flags), ('verification', ver_status), ('builder_sha256', bsha), ('frozen_passages', len(frozen)),
        ('accepted_passages', len(accepted_ids)), ('accepted_passage_ids', accepted_ids),
        ('accepted_ids_sha256', sha256_text('\n'.join(accepted_ids))), ('windows', n_rows),
        ('exclusions', dict(reasons)), ('excluded', excluded), ('output_sha256', sha256_file(final))])
    write_extraction_record(out, info)
    log(f'Holdout features written: {final} ({len(accepted_ids)} passages, {n_rows} windows).')
    return info


# ----------------------------------------------------------------------------
# Command: score-holdout
# ----------------------------------------------------------------------------
def combine_series(Ddev, Dhold):
    out = {}
    for k in ('E', 'N'):
        out[k] = np.vstack([Ddev[k], Dhold[k]]) if Dhold[k].shape[0] else Ddev[k].copy()
    out['ord'] = np.concatenate([Ddev['ord'], Dhold['ord']]).astype(int)
    out['offset'] = np.concatenate([Ddev['offset'], Dhold['offset']])
    out['n_passages'] = np.concatenate([Ddev['n_passages'], Dhold['n_passages']])
    out['is_hold'] = np.concatenate([np.zeros(len(Ddev['ord']), bool), np.ones(len(Dhold['ord']), bool)])
    for k in ('chainage', 'corridor', 'bins'):
        out[k] = Ddev[k]
    return out


def check_extraction_record(out, psha):
    """The holdout file, the extraction log and the ledger must all agree with the frozen protocol."""
    hp = holdout_paths(out)
    if not hp['log'].is_file():
        raise RuntimeError('No holdout extraction record (holdout/holdout_extraction_log.json). Run extract-holdout first.')
    if not hp['features'].is_file():
        raise RuntimeError('Holdout feature file not found.')
    xlog = json.loads(hp['log'].read_text(encoding='utf-8'))
    if xlog.get('protocol_sha256') != psha:
        raise RuntimeError('The frozen protocol differs from the one recorded at holdout extraction. Stop.')
    fsha = sha256_file(hp['features'])
    if xlog.get('output_sha256') != fsha:
        raise RuntimeError('The holdout feature file differs from the one written at extraction. Stop.')
    ids = sorted(xlog.get('accepted_passage_ids', []))
    if xlog.get('accepted_ids_sha256') != sha256_text('\n'.join(ids)):
        raise RuntimeError('The accepted passage list in the extraction record does not match its digest. Stop.')
    entries = [r for r in ledger_read(out) if r.get('event') == 'extract-holdout completed']
    if not entries:
        raise RuntimeError('The protocol ledger has no completed extraction. Stop.')
    last = entries[-1]
    if (last.get('protocol_sha256') != psha or last.get('output_sha256') != fsha
            or last.get('extraction_log_sha256') != sha256_file(hp['log'])):
        raise RuntimeError('The protocol ledger does not match the extraction record or the holdout file. Stop.')
    return xlog, fsha


def cmd_score_holdout(args):
    out = Path(args.output_dir)
    protocol, psha = load_protocol(out)
    hp = holdout_paths(out)
    cur = this_file_sha256()
    amendment = getattr(args, 'protocol_amendment', None)
    rescore = getattr(args, 'rescore_reason', None)
    if cur != protocol['pipeline_sha256'] and not amendment:
        raise RuntimeError('This pipeline file differs from the frozen one. Restore it, or state a documented '
                           '--protocol-amendment "reason" (reported in the results).')
    xlog, fsha = check_extraction_record(out, psha)
    previous = [r for r in ledger_read(out) if r.get('event') == 'score-holdout completed']
    if (previous or hp['results'].exists()) and not rescore:
        raise RuntimeError('The holdout has already been scored. A further scoring needs --rescore-reason "...", which is '
                           'recorded in the ledger and the results; earlier results are kept.')
    cfg = protocol['config']
    FC, DT = cfg['forecast'], cfg['detection']
    pm = protocol['primary_model']
    alpha = protocol.get('decision_alpha', FC.get('decision_alpha', 0.05))
    found = Foundation(args.foundation_dir, allow_count_mismatch=getattr(args, 'allow_count_mismatch', False))
    if found.sha256 != protocol['inputs_sha256']:
        raise RuntimeError('Foundation files differ from those in the frozen protocol.')
    hold = FeatureTable.read(hp['features'])
    hids = set(hold.passages())
    if sorted(hids) != sorted(xlog.get('accepted_passage_ids', [])):
        raise RuntimeError('Passages in the holdout file differ from those accepted at extraction. Stop.')
    extra = hids - set(protocol['holdout_passage_ids'])
    if extra:
        raise RuntimeError(f'{len(extra)} passages in the holdout file are not frozen holdout passages.')
    if hold.n and o2d(hold.ord.min()) <= protocol['embargo_end']:
        raise RuntimeError('Holdout file contains dates inside the development period or embargo.')
    rdir = hp['results']
    superseded = None
    if rdir.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        superseded = out / f'holdout_results_superseded_{stamp}'
        shutil.move(str(rdir), str(superseded))
    rdir.mkdir(parents=True)
    ledger_append(out, 'score-holdout started', protocol_sha256=psha, holdout_sha256=fsha, protocol_amendment=amendment,
                  rescore_reason=rescore, previous_scorings=len(previous),
                  superseded_results=str(superseded) if superseded else None)
    rng = np.random.default_rng(cfg['seed'] + 1)
    mrows_f, rrows_f, mrows_s, rrows_s, ep_rows, alarm_rows, qc_rows, brk_rows = [], [], [], [], [], [], [], []
    burden = {}
    prim = protocol['primary_indicator']
    for key, spec in protocol['indicators'].items():
        model = spec['model']
        fits = spec['fits']
        thresholds = spec['alarm_thresholds']
        series = {}
        for d in DIRECTIONS:
            Adev = apply_indicator_model(model, found.table, d)
            Ahold = apply_indicator_model(model, hold, d)
            Ddev = day_aggregate(Adev)
            Dhold = day_aggregate(Ahold)
            qc_rows.append([key, d, len(Ahold['pids']), int(Ahold['qc'].sum()),
                            float(Ahold['qc'].mean()) if len(Ahold['qc']) else float('nan'),
                            float(nanmed(Ahold['offset'][Ahold['qc']])) if Ahold['qc'].any() else float('nan'),
                            float(nanmed(Adev['offset'][Adev['qc']])) if Adev['qc'].any() else float('nan'),
                            len(Dhold['ord'])])
            series[d] = combine_series(Ddev, Dhold)
        pairs = attach_breaks(series, model['feature'], FC)
        last_dev = max(int(D['ord'][~D['is_hold']].max()) for D in series.values() if (~D['is_hold']).any())
        for bd, bc in pairs:
            if bc > last_dev:
                brk_rows.append([key, o2d(bd), o2d(bc)])

        def fixed_origin(D, t, h):
            return int(np.sum(~D['is_hold'])) - 1

        hold_targets = lambda D: np.where(D['is_hold'])[0]
        recs_f, ranks_f = collect_forecasts(series, fits, hold_targets, dict(FC, horizons=['fixed origin']), origin_fn=fixed_origin)
        a, b = forecast_metric_rows(key, recs_f, ranks_f, FC, rng, 'holdout fixed origin')
        mrows_f += a
        rrows_f += b
        ep_rows += episode_rows(key, recs_f, ranks_f, 'holdout fixed origin')
        recs_s, ranks_s = collect_forecasts(series, fits, hold_targets, dict(FC, horizons=['next']))
        a, b = forecast_metric_rows(key, recs_s, ranks_s, FC, rng, 'holdout sequential')
        mrows_s += a
        rrows_s += b
        ep_rows += episode_rows(key, recs_s, ranks_s, 'holdout sequential')
        # Alarms exactly as the development null: origins at the last development day and at every holdout day,
        # statistics over the next m holdout observation days within the frozen horizon.
        for D in series.values():
            n_dev = int(np.sum(~D['is_hold']))
            D['alarm_origin'] = D['is_hold'] | (np.arange(len(D['ord'])) == n_dev - 1)
        recs_a = []
        segment_null_stats(series, pm, fits[pm]['params'], DT, -10 ** 9, extra_q(fits, pm), target_mask_key='is_hold',
                           records=recs_a, origin_mask_key='alarm_origin')
        for rec in recs_a:
            d, L, m_, b0, b1, chn, cor, stat, oday, f0, f1 = rec
            th = thresholds.get(f'{L}|{m_}')
            if th is None:
                continue
            B_ = burden.setdefault((key, L, m_), {'tested': 0, 'alarms': 0, 'tested_corridor': 0, 'alarms_corridor': 0})
            B_['tested'] += 1
            B_['tested_corridor'] += int(cor)
            if abs(stat) > th['threshold']:
                B_['alarms'] += 1
                B_['alarms_corridor'] += int(cor)
                alarm_rows.append([key, d, L, m_, b0, b1, chn, cor, stat, th['threshold'], oday, f0, f1])
    write_csv(rdir / 'holdout_forecast_metrics.csv', FORECAST_HEADER, mrows_f + mrows_s)
    write_csv(rdir / 'holdout_forecast_ranking.csv', RANK_HEADER, rrows_f + rrows_s)
    write_csv(rdir / 'holdout_episode_metrics.csv', EPISODE_HEADER, ep_rows)
    write_csv(rdir / 'holdout_alarms.csv', ['indicator', 'direction', 'segment_bins', 'days_after', 'first_bin', 'last_bin',
                                            'chainage_start_m', 'touches_corridor', 'statistic', 'threshold', 'origin_day',
                                            'first_day', 'last_day'], alarm_rows)
    bur_rows = [[k[0], k[1], k[2], v['tested'], v['alarms'], v['alarms'] / v['tested'] if v['tested'] else float('nan'),
                 v['tested_corridor'] / v['tested'] if v['tested'] else float('nan'),
                 v['alarms_corridor'] / v['alarms'] if v['alarms'] else float('nan')] for k, v in burden.items()]
    write_csv(rdir / 'holdout_alarm_burden.csv', ['indicator', 'segment_bins', 'days_after', 'segments_tested', 'alarms',
                                                  'alarm_share', 'corridor_share_of_tested', 'corridor_share_of_alarms'], bur_rows)
    write_csv(rdir / 'holdout_qc.csv', ['indicator', 'direction', 'holdout_passages', 'holdout_passages_qc',
                                        'qc_pass_fraction', 'holdout_median_level_offset', 'development_median_level_offset',
                                        'holdout_days'], qc_rows)
    write_csv(rdir / 'holdout_regime_breaks.csv', ['indicator', 'first_day_of_new_regime', 'confirmed_on'], brk_rows)
    res = OrderedDict()
    seq_m = [r for r in mrows_s if r[1] == prim and r[3] == pm]
    seq_r = [r for r in rrows_s if r[1] == prim and r[3] == pm]
    fix_m = [r for r in mrows_f if r[1] == prim and r[3] == pm]

    def fin(v):
        return v is not None and np.isfinite(v)

    def decided(p):
        return bool(fin(p) and p < alpha)

    c1, c2 = comparison_order(cfg)

    def against(r, base):
        if base == 'persistence':
            return OrderedDict([('supported', decided(r[24])), ('episodes_better', r[22]), ('episodes', r[23]),
                                ('sign_test_p', r[24]), ('skill', r[10]), ('bootstrap_lo95', r[11]),
                                ('bootstrap_hi95', r[12]), ('n_bin_days', r[4])])
        return OrderedDict([('supported', decided(r[21])), ('episodes_better', r[19]), ('episodes', r[20]),
                            ('sign_test_p', r[21]), ('skill', r[13]), ('bootstrap_lo95', r[14]),
                            ('bootstrap_hi95', r[15]), ('n_bin_days', r[4])])

    if seq_m:
        res[f'H1_sequential_vs_{c1}'] = against(seq_m[0], c1)
        res[f'H2_sequential_vs_{c2}'] = against(seq_m[0], c2)
    if seq_r:
        r = seq_r[0]
        res['H3_sequential_ranking'] = OrderedDict([
            ('supported', decided(r[14])), ('episodes_positive', r[12]), ('episodes', r[13]), ('sign_test_p', r[14]),
            ('mean_spearman', r[6]), ('bootstrap_lo95', r[7]), ('bootstrap_hi95', r[8])])
    if seq_m:
        r = seq_m[0]
        res['H4_sequential_calibration'] = OrderedDict([('supported', bool(fin(r[17]) and 0.70 <= r[17] <= 0.90)),
                                                        ('coverage80', r[17]), ('coverage95', r[18])])
    if fix_m:
        r = fix_m[0]
        res['H5_fixed_origin_vs_fingerprint'] = OrderedDict([
            ('supported', decided(r[21])), ('episodes_better', r[19]), ('episodes', r[20]), ('sign_test_p', r[21]),
            ('skill', r[13]), ('bootstrap_lo95', r[14]), ('bootstrap_hi95', r[15])])
    res['D1_alarms'] = OrderedDict((f'{L} bins, {m_} days', {'segments_tested': v['tested'], 'alarms': v['alarms'],
                                                              'alarm_share': (v['alarms'] / v['tested']) if v['tested'] else None,
                                                              'corridor_share_of_tested': (v['tested_corridor'] / v['tested']) if v['tested'] else None,
                                                              'corridor_share_of_alarms': (v['alarms_corridor'] / v['alarms']) if v['alarms'] else None})
                                   for (k, L, m_), v in burden.items() if k == prim)
    result = OrderedDict([
        ('created_utc', datetime.now(timezone.utc).isoformat(timespec='seconds')), ('protocol_sha256', psha),
        ('pipeline_sha256', cur), ('protocol_amendment', amendment), ('rescore_reason', rescore),
        ('previous_scorings', len(previous)), ('holdout_features_sha256', fsha),
        ('extraction_log_sha256', sha256_file(hp['log'])), ('registration', xlog.get('registration')),
        ('extraction_flags', xlog.get('flags')), ('primary_indicator', prim), ('primary_model', pm),
        ('decision_alpha', alpha), ('holdout_passages', len(hids)), ('holdout_windows', hold.n),
        ('hypotheses', res), ('regime_breaks_in_holdout', brk_rows), ('plan', protocol['plan'])])
    write_json(rdir / 'HOLDOUT_RESULTS.json', result)
    figs = {}
    plt = get_plt()
    if plt is not None:
        try:
            figs['forecast'] = fig_forecast(plt, mrows_s, [prim] + protocol['secondary_indicators'],
                                            rdir / 'figures' / 'holdout_forecast_errors.png', scope='holdout sequential')
        except Exception as exc:
            log(f'Figure failed: {exc}')
    hyp_text = protocol['plan']['hypotheses']

    def verdict(hk, hv):
        if hk.startswith('D'):
            return 'descriptive'
        return 'SUPPORTED' if hv.get('supported') else 'not supported'

    blocks = ['<h1>DR-Train Region 5 study: holdout evaluation</h1>',
              f'<p class="note">Protocol SHA-256 <code>{psha[:16]}</code>; pipeline SHA-256 <code>{cur[:16]}</code>; holdout file '
              f'SHA-256 <code>{fsha[:16]}</code>. External registration: {html_escape(xlog.get("registration") or "none recorded")}. '
              f'{"Protocol amendment: " + html_escape(amendment) if amendment else "No protocol amendment."} '
              f'{"Re-scoring, reason: " + html_escape(rescore) if rescore else "First scoring."} '
              f'Primary indicator {html_escape(prim)}, primary model {html_escape(pm)}.</p>',
              f'<p>{html_escape(protocol["plan"].get("decision_rule", ""))}</p>',
              '<div class="key"><h3>Pre-declared hypotheses</h3><ul>']
    for hk, hv in res.items():
        blocks.append(f'<li><b>{html_escape(hk)}: {verdict(hk, hv)}</b>. {html_escape(json.dumps(to_jsonable(hv)))}<br>'
                      f'<span class="note">{html_escape(hyp_text.get(hk, ""))}</span></li>')
    blocks.append('</ul></div>')
    blocks.append(html_figure(figs.get('forecast'), 'Holdout errors of sequential one-step-ahead forecasts.'))
    blocks.append(html_table(EPISODE_HEADER[1:], [r[1:] for r in ep_rows if r[1] == prim], 80,
                             caption='Primary indicator by calendar episode'))
    blocks.append(html_table(['Scope', 'Indicator', 'Horizon', 'Model', 'Bin-days', 'Episodes', 'MAE', 'Skill vs persistence', 'Low',
                              'High', 'Better than persistence', 'Sign test p', 'Skill vs fingerprint', 'Low', 'High',
                              'Better than fingerprint', 'Sign test p', 'Coverage 80%'],
                             [[r[0], r[1], r[2], r[3], iv(r[4]), iv(r[5]), r[8], r[10], r[11], r[12], f'{iv(r[22])} of {iv(r[23])}',
                               r[24], r[13], r[14], r[15], f'{iv(r[19])} of {iv(r[20])}', r[21], r[17]] for r in mrows_s + mrows_f],
                             80, caption='Forecast metrics (episodes: calendar episodes in which the model is better)'))
    blocks.append(html_table(RANK_HEADER, rrows_s + rrows_f, 60, caption='Ranking'))
    blocks.append(html_table(['Indicator', 'Segment bins', 'Days after', 'Segments tested', 'Alarms', 'Alarm share',
                              'Corridor share of tested', 'Corridor share of alarms'], bur_rows, 40,
                             caption='Alarm burden against the frozen thresholds (5% expected without change)'))
    blocks.append(html_table(['Indicator', 'Direction', 'Passages', 'Passages QC', 'QC pass', 'Holdout level offset',
                              'Development level offset', 'Days'], qc_rows, caption='Quality control and level offsets'))
    blocks.append(html_table(['Indicator', 'First day of new regime', 'Confirmed on'], brk_rows,
                             caption='Confirmed causal regime breaks inside the holdout'))
    blocks.append(html_table(['Indicator', 'Direction', 'Bins', 'Days', 'First bin', 'Last bin', 'Chainage', 'Corridor', 'Statistic',
                              'Threshold', 'Origin', 'First day', 'Last day'], alarm_rows, 60, caption='Alarms'))
    write_text(rdir / 'HOLDOUT_REPORT.html', html_page('DR-Train study holdout evaluation', blocks))
    write_text(rdir / 'scoring_log.txt', '\n'.join(LOG_LINES) + '\n')
    ledger_append(out, 'score-holdout completed', protocol_sha256=psha, holdout_sha256=fsha,
                  results_sha256=sha256_file(rdir / 'HOLDOUT_RESULTS.json'),
                  decisions={k: v.get('supported') for k, v in res.items() if not k.startswith('D')})
    log(f'Holdout scoring complete: {rdir / "HOLDOUT_REPORT.html"}')
    for hk, hv in res.items():
        log(f'RESULT {hk}: {json.dumps(to_jsonable(hv))}')
    return result


# ----------------------------------------------------------------------------
# Command: self-test (development data only)
# ----------------------------------------------------------------------------
def cmd_self_test(args):
    """Pseudo-holdout drawn from the latest DEVELOPMENT dates; the real holdout is never touched."""
    src = Path(args.foundation_dir)
    base = Path(args.output_dir) / 'self_test'
    if base.exists():
        shutil.rmtree(base)
    fdir = base / 'pseudo_foundation'
    odir = base / 'pseudo_output'
    fdir.mkdir(parents=True)
    cut = args.self_test_cut
    emb = (date.fromisoformat(cut) + timedelta(days=7)).isoformat()
    manifest = json.loads((src / FOUNDATION_FILES['manifest']).read_text(encoding='utf-8'))
    dev_rows, dev_scr = 0, 0
    dev_p, hold_p = set(), set()
    hp = holdout_paths(odir)
    staged_hold = base / 'staged_holdout.csv.gz'
    with gzip.open(src / FOUNDATION_FILES['features'], 'rt', newline='', encoding='utf-8') as fin, \
            gzip.open(fdir / FOUNDATION_FILES['features'], 'wt', newline='', encoding='utf-8') as fdev, \
            gzip.open(staged_hold, 'wt', newline='', encoding='utf-8') as fhold:
        r = csv.reader(fin)
        header = next(r)
        di = header.index('local_date')
        pi = header.index('passage_id')
        si = header.index('analysis_screening_flag')
        wd, wh = csv.writer(fdev), csv.writer(fhold)
        wd.writerow(header)
        wh.writerow(header)
        for row in r:
            if row[di] <= cut:
                wd.writerow(row)
                dev_rows += 1
                dev_scr += row[si] == 'speed_screen_pass'
                dev_p.add(row[pi])
            elif row[di] > emb:
                wh.writerow(row)
                hold_p.add(row[pi])
    ids = sorted(hold_p)
    manifest['chronological_split'].update({
        'development_last_date': cut, 'seven_day_embargo_end': emb, 'complete_heldout_recordings': len(ids),
        'holdout_passage_ids': ids, 'holdout_id_sha256': sha256_text('\n'.join(ids))})
    manifest['counts'].update({'development_complete_feature_windows': dev_rows,
                               'development_speed_screened_feature_windows': dev_scr,
                               'development_accepted_passages': len(dev_p)})
    write_json(fdir / FOUNDATION_FILES['manifest'], manifest)
    shutil.copy(src / FOUNDATION_FILES['coverage'], fdir / FOUNDATION_FILES['coverage'])
    log(f'Self-test: pseudo development to {cut} ({len(dev_p)} passages), pseudo holdout after {emb} ({len(ids)} passages)')
    cfg = apply_quick(CONFIG)
    ns = argparse.Namespace(foundation_dir=str(fdir), output_dir=str(odir), replace=True, quick=False,
                            allow_count_mismatch=False, allow_quick=True, protocol_amendment=None, rescore_reason=None,
                            registration='self-test, no registration')
    cmd_develop(ns, cfg)
    protocol = cmd_freeze(ns, cfg)
    _, psha = load_protocol(odir)
    # Stand-in for extract-holdout: the staged pseudo-holdout and an extraction record of the same form.
    ledger_append(odir, 'extract-holdout started', protocol_sha256=psha, flags={'self_test': True})
    hp['features'].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(staged_hold, hp['features'])
    info = OrderedDict([('created_utc', datetime.now(timezone.utc).isoformat(timespec='seconds')), ('protocol_sha256', psha),
                        ('pipeline_sha256', this_file_sha256()), ('frozen_pipeline_sha256', protocol['pipeline_sha256']),
                        ('protocol_amendment', None), ('registration', ns.registration), ('flags', {'self_test': True}),
                        ('verification', 'not applicable (self-test)'), ('accepted_passages', len(ids)),
                        ('accepted_passage_ids', ids), ('accepted_ids_sha256', sha256_text('\n'.join(ids))),
                        ('output_sha256', sha256_file(hp['features']))])
    write_extraction_record(odir, info)
    refused = False
    try:
        cmd_freeze(ns, cfg)
    except RuntimeError as exc:
        refused = True
        log(f'Self-test: re-freeze correctly refused ({exc}).')
    res = cmd_score_holdout(ns)
    second_refused = False
    try:
        cmd_score_holdout(ns)
    except RuntimeError as exc:
        second_refused = True
        log(f'Self-test: second scoring correctly refused ({exc}).')
    # A replaced protocol must be detected even with a stated re-scoring reason.
    tamper_refused = False
    pf, ph = protocol_paths(odir)
    saved_p, saved_h = pf.read_text(encoding='utf-8'), ph.read_text(encoding='utf-8')
    try:
        fake = saved_p.replace('"decision_alpha": 0.05', '"decision_alpha": 0.5')
        write_text(pf, fake)
        write_text(ph, sha256_text(fake) + '  FROZEN_PROTOCOL.json\n')
        ns2 = argparse.Namespace(**dict(vars(ns), rescore_reason='self-test of the tamper check'))
        try:
            cmd_score_holdout(ns2)
        except RuntimeError as exc:
            tamper_refused = True
            log(f'Self-test: scoring with a replaced protocol correctly refused ({exc}).')
    finally:
        write_text(pf, saved_p)
        write_text(ph, saved_h)
    checks = {
        'development_report': (odir / 'develop' / 'DEVELOPMENT_REPORT.html').is_file(),
        'protocol': (odir / 'FROZEN_PROTOCOL.json').is_file(),
        'refreeze_refused_after_extraction': refused,
        'holdout_report': (odir / 'holdout_results' / 'HOLDOUT_REPORT.html').is_file(),
        'hypotheses_evaluated': sum(1 for k in res['hypotheses'] if k.startswith('H')) >= 5,
        'second_scoring_refused': second_refused,
        'replaced_protocol_refused': tamper_refused,
        'ledger_complete': all(ledger_has(odir, e) for e in ('develop', 'freeze', 'extract-holdout completed',
                                                              'score-holdout completed')),
    }
    log('Self-test checks: ' + json.dumps(checks))
    if not all(checks.values()):
        raise SystemExit(4)
    log('SELF-TEST PASS')
    return checks


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main(argv=None):
    quiet_numpy_warnings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--foundation-dir', default=DEFAULTS['foundation_dir'])
        p.add_argument('--output-dir', default=DEFAULTS['output_dir'])
        p.add_argument('--allow-count-mismatch', action='store_true', help=argparse.SUPPRESS)

    p = sub.add_parser('develop', help='development-only analyses and report')
    common(p)
    p.add_argument('--replace', action='store_true', help='overwrite previous development results')
    p.add_argument('--quick', action='store_true', help='fewer repeats and placebo draws (for a trial run only)')
    p = sub.add_parser('freeze', help='freeze models, thresholds and hypotheses')
    common(p)
    p.add_argument('--replace', action='store_true', help='re-freeze (only before holdout extraction)')
    for name in ('verify-extraction', 'extract-holdout'):
        p = sub.add_parser(name)
        common(p)
        p.add_argument('--data-root', default=DEFAULTS['data_root'])
        p.add_argument('--builder', default=DEFAULTS['builder'])
        p.add_argument('--allow-builder-mismatch', action='store_true', help=argparse.SUPPRESS)
        if name == 'verify-extraction':
            p.add_argument('--n-passages', type=int, default=4)
        else:
            p.add_argument('--confirm-single-use', action='store_true')
            p.add_argument('--registration', default=None,
                           help='where the protocol SHA-256 was registered (for example an OSF or DOI link)')
            p.add_argument('--protocol-amendment', default=None, help=argparse.SUPPRESS)
            p.add_argument('--skip-verification-check', action='store_true', help=argparse.SUPPRESS)
            p.add_argument('--allow-holdout-manifest-mismatch', action='store_true', help=argparse.SUPPRESS)
    p = sub.add_parser('score-holdout', help='single prospective evaluation')
    common(p)
    p.add_argument('--protocol-amendment', default=None)
    p.add_argument('--rescore-reason', default=None, help='only for a documented second scoring; earlier results are kept')
    p = sub.add_parser('self-test', help='pseudo-holdout test on development data only')
    common(p)
    p.add_argument('--self-test-cut', default='2015-08-31')
    args = ap.parse_args(argv)
    try:
        if args.command == 'develop':
            cmd_develop(args)
        elif args.command == 'freeze':
            cmd_freeze(args)
        elif args.command == 'verify-extraction':
            cmd_verify_extraction(args)
        elif args.command == 'extract-holdout':
            cmd_extract_holdout(args)
        elif args.command == 'score-holdout':
            cmd_score_holdout(args)
        elif args.command == 'self-test':
            cmd_self_test(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        log(f'FAILED: {exc}')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
