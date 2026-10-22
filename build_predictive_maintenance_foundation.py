#!/usr/bin/env python3
"""Self-contained DR-Train predictive-maintenance FOUNDATION builder, version 1.0.

One Python file. It DOES NOT import or execute earlier project scripts and it
DOES NOT read earlier Stage 01-31 output files. Input is the ORIGINAL DR-Train
MATLAB-exported metadata CSV, raw GPS and accelerometer MAT files, Region 5
reference tracks, and original maintenance log CSV.

Three output files only (in an isolated directory):
    development_vibration_features.csv.gz : 100 m, five-sensor, 110-feature table
    maintenance_event_coverage.csv         : 27 individual schedule records and
                                             direction-specific observation counts
    foundation_manifest.json               : provenance, exclusions, holdout IDs,
                                             feature definitions and QA diagnostics

The 89 previously frozen 2016 passages are NEVER opened. The 2015-12-31 embargo
is excluded. This program is not a fitted anomaly detector, classification
model or verified maintenance predictor. It provides reproducible input to one.

Cluster example:
    python3 build_predictive_maintenance_foundation.py --check-inputs
    # Submit the command below through SLURM, NOT the login node:
    python3 -u build_predictive_maintenance_foundation.py

Dependencies: Python 3.9+, NumPy, SciPy, h5py. No pandas or scikit-learn.
The *original* LRV4306_metadata.csv is a pre-existing export of MATLAB pass
objects and is treated as a raw metadata input, not a previous analysis output.
The full MATLAB object dictionary is NOT decoded by this script.

Reference implementation: reproduces the salient original Stage 02, Stage 04,
Stage 04A and Stage 05 algorithms in one file, with equivalent screening
thresholds and feature names. It does not guarantee an identical record set to
older staged outputs because the full development pool is processed jointly.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.spatial import cKDTree

VERSION = '1.0.0'
CUTOFF = date(2015, 12, 24)
EMBARGO_DAYS = 7
# Frozen holdout is defined by its individual IDs, not merely the date range.
# SHA-256 of '\n'.join(sorted(passage IDs)) from Stage 28's 89-ID manifest.
EXPECTED_HOLDOUT_COUNT = 89
EXPECTED_HOLDOUT_ID_SHA256 = '631088d20db147d1c8d2846a26d79f8a46afdc687bd75b088c9c4bbf085fc000'
CROSSOVER_COORDS = {'Denise': (40.40060278, -79.99922500),
                    'Glenbury': (40.38193056, -79.99886389)}
CROSSOVER_NOTE = 'Approximate user-identified coordinates; not independently verified maintained work limits'
BIN_M = 100
FS = 1651.6
MIN_SPEED_MPS = 5.0
BANDS = [(0.5, 5), (5, 20), (20, 80), (80, 250), (250, 600)]
FEATURES = [
    'mean', 'rms_raw', 'rms_demeaned', 'peak_to_peak', 'median',
    'mean_abs_demeaned', 'p95_abs_demeaned', 'peak_abs_demeaned',
    'crest_factor', 'iqr', 'mad_median', 'skewness', 'excess_kurtosis',
    'zero_crossing_fraction', 'bandpower_0p5_5hz', 'bandpower_5_20hz',
    'bandpower_20_80hz', 'bandpower_80_250hz',
    'bandpower_250_600hz', 'bandpower_total_0p5_600hz',
    'spectral_centroid_hz', 'spectral_entropy',
]
META = [
    'passage_id', 'vehicle', 'region', 'direction', 'metadata_utc', 'local_date',
    'track_bin_id', 'chainage_start_m', 'chainage_end_m', 'corridor_candidate_bin',
    'estimated_speed_mps', 'estimated_duration_s',
    'gps_median_track_residual_m', 'gps_p95_track_residual_m',
    'gps_points_inside_bin', 'gps_max_gap_near_bin_s',
    'gps_metadata_start_difference_s', 'n_sensors_with_features',
    'raw_acc_time_alignment', 'analysis_screening_flag',
]
FIELDS = META + [f's{k}_{n}' for k in range(1, 6) for n in FEATURES]
EVENT_FIELDS = [
    'record_id', 'start_date', 'end_date', 'direction', 'activity',
    'source_work', 'source_location', 'source_repeats',
    'localisation', 'before_days', 'before_passages', 'during_days',
    'during_passages', 'after_days', 'after_passages',
    'other_schedules_within_30d', 'other_schedule_ids',
    'min_4days_each_side', 'label_reliability',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def open_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def file_name(s):
    return str(s or '').strip().replace('\\', '/').rsplit('/', 1)[-1]


def parse_utc(s):
    s = str(s or '').strip()
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        d = None
        for fmt in ('%d-%b-%Y %H:%M:%S', '%d-%b-%Y %H:%M:%S.%f',
                    '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %I:%M:%S %p',
                    '%Y-%m-%d', '%d-%b-%Y'):
            try:
                d = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
    if d is None:
        return None
    return (d.replace(tzinfo=timezone.utc) if d.tzinfo is None
            else d.astimezone(timezone.utc))


def catalogue(metadata, data_root):
    src = open_csv(metadata)
    required = {'region', 'direction', 'sensor', 'date', 'acc_file', 'gps_file'}
    if not src or not required.issubset(src[0]):
        raise ValueError('Metadata is empty or missing fields: ' + ', '.join(sorted(required)))
    groups = defaultdict(list)
    for row in src:
        if str(row['region']).strip() not in {'5', '5.0'}:
            continue
        direction = row['direction'].strip().lower()
        if direction not in {'inbound', 'outbound'}:
            continue
        gps = file_name(row['gps_file'])
        if not gps:
            continue
        groups[('LRV4306', '5', direction, gps)].append(row)
    all_gps = set(os.listdir(data_root / 'LRV4306' / 'gps_data'))
    all_acc = set(os.listdir(data_root / 'LRV4306' / 'accelerometer_data'))
    result = []
    for key, rows in sorted(groups.items()):
        veh, region, direction, gps = key
        per_sensor = defaultdict(list)
        for r in rows:
            try:
                sid = int(float(r['sensor']))
            except (ValueError, TypeError):
                sid = -1
            per_sensor[sid].append(r)
        dates = {r['date'].strip() for r in rows}
        ts = parse_utc(next(iter(dates))) if len(dates) == 1 else None
        if ts is None:
            continue
        fnames = {}
        for i in range(1, 6):
            rr = per_sensor.get(i, [])
            fnames[i] = file_name(rr[0]['acc_file']) if len(rr) == 1 else ''
        complete = (len(rows) == 5 and len(per_sensor) == 5 and
                    all(fnames[i] in all_acc and fnames[i] for i in range(1, 6)) and
                    gps in all_gps)
        raw_id = '|'.join(key)
        passage_id = 'P4306_' + hashlib.sha1(raw_id.encode()).hexdigest()[:12].upper()
        result.append(dict(passage_id=passage_id, vehicle=veh, region=region,
                           direction=direction, gps_file=gps,
                           metadata_utc=ts.isoformat(timespec='seconds'),
                           local_date=ts.astimezone(ZoneInfo('America/New_York')).date().isoformat(),
                           complete_file_set=complete,
                           **{f'acc_sensor_{i}_file': fnames[i] for i in range(1, 6)}))
    return result


def read_gps(path):
    try:
        a = loadmat(str(path), variable_names=['save_var_gps'])['save_var_gps']
    except (ValueError, NotImplementedError):
        with h5py.File(path, 'r') as h:
            a = h['save_var_gps'][()]
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError('bad GPS matrix dimensions')
    if a.shape[1] != 5 and a.shape[0] == 5:
        a = a.T
    if a.shape[1] < 5 or len(a) < 5:
        raise ValueError('bad GPS matrix shape')
    return np.asarray(a[:, :5], dtype=np.float64)


def array_digest(values):
    a = np.ascontiguousarray(np.asarray(values, dtype='<f8'))
    h = hashlib.sha256()
    h.update(str(tuple(a.shape)).encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def load_acc(path):
    try:
        with h5py.File(path, 'r') as f:
            a = f['save_var'][()]
    except OSError:
        a = loadmat(str(path), variable_names=['save_var'])['save_var']
    a = np.asarray(a, dtype=np.float64)
    if a.ndim not in (1, 2) or (a.ndim == 2 and 1 not in a.shape):
        raise ValueError('bad acceleration shape')
    return a.reshape(-1)


def acc_length(path):
    try:
        with h5py.File(path, 'r') as h:
            ds = h['save_var']
            if ds.ndim not in (1, 2) or (ds.ndim == 2 and min(ds.shape) != 1):
                raise ValueError('bad acceleration shape')
            return int(ds.size)
    except OSError:
        return len(load_acc(path))


def ll_to_xy(lon, lat, origin):
    lon0, lat0 = origin
    return np.column_stack(((np.asarray(lon) - lon0) * 111320 * np.cos(np.deg2rad(lat0)),
                            (np.asarray(lat) - lat0) * 111320))


def rigid(x, y, weights):
    sw = np.sum(weights)
    if not np.isfinite(sw) or sw <= 1.e-6:
        return np.eye(2), np.zeros(2)
    cx = np.sum(x * weights[:, None], axis=0) / sw
    cy = np.sum(y * weights[:, None], axis=0) / sw
    cov = ((x-cx) * weights[:, None]).T @ (y-cy)
    u, _, vt = np.linalg.svd(cov)
    d = np.eye(2)
    d[-1, -1] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    return r, cy-cx@r


def icp(x, ref, tree):
    original_sep, original_idx = tree.query(x)
    moving = x.copy()
    total_rotation = np.eye(2)
    for _ in range(35):
        sep, near = tree.query(moving)
        scale = 4.7536 * max(1.9 * float(np.median(sep)), 1e-9)
        weights = np.exp(-(sep / scale) ** 2)
        r, t = rigid(moving, ref[near], weights)
        updated = moving @ r + t
        total_rotation = total_rotation @ r
        delta = float(np.max(np.linalg.norm(updated - moving, axis=1)))
        moving = updated
        if delta < 0.005:
            break
    sep, near = tree.query(moving)
    movement = float(np.max(np.linalg.norm(moving - x, axis=1)))
    rotation = abs(float(np.degrees(np.arctan2(total_rotation[1, 0], total_rotation[0, 0]))))
    if (movement > 100.0 or rotation > 5.0 or
            np.median(sep) > np.median(original_sep) + 0.01):
        return original_sep, original_idx, False, movement, rotation
    return sep, near, True, movement, rotation


def feature_vector(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.size < 1000 or not np.isfinite(x).all():
        raise ValueError('invalid acceleration segment')
    mean = float(np.mean(x))
    xc = x - mean
    sd = math.sqrt(float(np.mean(xc * xc)))
    if sd < 1e-10:
        raise ValueError('constant signal')
    absxc = np.abs(xc)
    med = float(np.median(x))
    q25, q75 = np.percentile(x, (25, 75))
    nseg = min(len(x), 4096)
    freq, psd = signal.welch(xc, fs=FS, window='hann', nperseg=nseg,
                             noverlap=nseg//2, detrend='constant', scaling='density')
    df = float(freq[1] - freq[0])
    power = [float(np.sum(psd[(freq >= low) & (freq < high)]) * df)
             for low, high in BANDS]
    mask = (freq >= 0.5) & (freq < 600)
    spectrum = np.maximum(0.0, psd[mask])
    total = float(np.sum(spectrum)*df)
    if total > 0 and len(spectrum) > 1:
        p = spectrum / np.sum(spectrum)
        centroid = float(np.sum(freq[mask]*p))
        entropy = float(-np.sum(p[p > 0]*np.log(p[p > 0])) / np.log(len(p)))
    else:
        centroid = entropy = float('nan')
    vals = [mean, float(np.sqrt(np.mean(x*x))), sd, float(np.ptp(x)), med,
            float(np.mean(absxc)), float(np.percentile(absxc, 95)), float(np.max(absxc)),
            float(np.max(absxc) / sd), float(q75-q25), float(np.median(np.abs(x-med))),
            float(np.mean((xc/sd)**3)), float(np.mean((xc/sd)**4)-3),
            float(np.count_nonzero(np.diff(np.signbit(xc)))/(len(x)-1)),
            *power, total, centroid, entropy]
    if len(vals) != len(FEATURES) or not np.isfinite(vals).all():
        raise ValueError('non-finite vibration features')
    return vals


def sensor_window(start, stop, n1, n):
    if n == n1:
        return start, stop
    scale = (n-1)/(n1-1)
    a = int(math.ceil(start*scale))
    b = int(math.floor((stop-1)*scale)) + 1
    return max(0,min(a,n)),max(0,min(b,n))


def make_track(data_root, direction):
    p = data_root/'ground_truth_rail_layout_GPS'/f'region5_{direction}.csv'
    a = np.loadtxt(p, delimiter=',', ndmin=2)
    if a.shape[1] < 3 or len(a) < 2 or not np.isfinite(a[:, :3]).all():
        raise ValueError(f'invalid track reference {p}')
    a = a[:, :3]
    if np.any(np.diff(a[:,0]) <= 0):
        raise ValueError(f'non-increasing reference chainage {p}')
    origin = (float(np.mean(a[:, 2])),float(np.mean(a[:, 1])))
    coords = ll_to_xy(a[:,2],a[:,1],origin)
    tree = cKDTree(coords)
    marks = {}
    for name, (lat, lon) in CROSSOVER_COORDS.items():
        xy = ll_to_xy(np.array([lon]),np.array([lat]),origin)
        residual, ind = tree.query(xy)
        marks[name] = {'chainage_m': float(a[int(ind[0]), 0]),
                       'track_separation_m': float(residual[0])}
    lo, hi = sorted([marks['Denise']['chainage_m'], marks['Glenbury']['chainage_m']])
    return dict(ref=a, xyz=coords, tree=tree, origin=origin, landmarks=marks,
                corridor=(lo,hi), path=str(p))


def build_windows(rec, gps, track, lens):
    ok = np.isfinite(gps[:,[0,1,4]]).all(axis=1)
    ok &= (gps[:,0]>-180)&(gps[:,0]<180)&(gps[:,1]>-90)&(gps[:,1]<90)
    g = gps[ok]
    if len(g) < 5:
        raise ValueError('too few GPS samples')
    t = (g[:,4]-g[0,4])*86400
    if not np.isfinite(t).all() or np.any(np.diff(t)<=0) or t[-1]<=0:
        raise ValueError('non-increasing GPS timestamps')
    if any(not (1630 < n/len(g) < 2100) for n in lens):
        raise ValueError('accelerometer-to-GPS sample ratio failed')
    if max(lens)-min(lens) > max(lens)*0.002:
        raise ValueError('inconsistent accelerometer channel lengths')
    xy = ll_to_xy(g[:,0],g[:,1],track['origin'])
    sep, ind, accepted, movement, rotation = icp(xy,track['xyz'],track['tree'])
    if not accepted:
        raise ValueError('ICP transformation rejected')
    chain = track['ref'][ind,0]
    sign = 1 if np.median(chain[-min(20,len(chain)):]) >= np.median(chain[:min(20,len(chain))]) else -1
    signed = chain*sign
    monotone = np.maximum.accumulate(signed)
    uniq, idx = np.unique(monotone,return_index=True)
    tmax = float(t[-1])
    windows = []
    full_len = float(track['ref'][-1,0])
    for st in np.arange(0,full_len,BIN_M):
        ed = min(st+BIN_M, full_len)
        a = st*sign if sign == 1 else -ed
        b = ed*sign if sign == 1 else -st
        if a < uniq[0] or b > uniq[-1]:
            continue
        ta, tb = float(np.interp(a,uniq,t[idx])),float(np.interp(b,uniq,t[idx]))
        if not (0 <= ta < tb <= tmax):
            continue
        speed = (ed-st)/(tb-ta)
        if speed <= 0 or speed > 35:
            continue
        local = np.where((t>=ta)&(t<=tb))[0]
        if len(local) < 3:
            continue
        ii = max(0,int(np.searchsorted(t,ta,side='right'))-1)
        jj = min(len(t),int(np.searchsorted(t,tb,side='left'))+1)
        sl = np.arange(ii,jj)
        if len(sl) < 2:
            continue
        gaps = np.diff(t[sl])
        median_sep = float(np.median(sep[sl]))
        p95_sep = float(np.percentile(sep[sl],95))
        if (float(np.max(gaps))>3.0 or median_sep>15.0 or p95_sep>30.0 or
                np.any(np.diff(signed[sl])<-8.0)):
            continue
        ia = int(math.ceil(ta*(lens[0]-1)/tmax))
        ib = int(math.floor(tb*(lens[0]-1)/tmax))+1
        ia = max(0,min(lens[0]-1,ia))
        ib = max(ia,min(lens[0],ib))
        if ib-ia<1000:
            continue
        corridor_lo, corridor_hi=track['corridor']
        corridor = (st>=corridor_lo and ed<=corridor_hi)
        windows.append(dict(
            track_bin_id=f"R5_{rec['direction']}_{int(round(st)):05d}_{int(round(ed)):05d}",
            chainage_start_m=round(float(st),2),chainage_end_m=round(float(ed),2),
            corridor_candidate_bin='yes' if corridor else 'no',
            estimated_speed_mps=round(float(speed),3),
            estimated_duration_s=round(float(tb-ta),4),
            gps_median_track_residual_m=round(median_sep,2),
            gps_p95_track_residual_m=round(p95_sep,2),
            gps_points_inside_bin=int(len(local)),
            gps_max_gap_near_bin_s=round(float(np.max(gaps)),3),
            ia=ia,ib=ib,
        ))
    if len(windows)<5:
        raise ValueError('fewer than five usable 100 m windows')
    diff_sec = None
    try:
        metadata=datetime.fromisoformat(rec['metadata_utc']).replace(tzinfo=None)
        gps_epoch=(datetime.fromordinal(int(g[0,4]))+timedelta(days=float(g[0,4])%1)-
                   timedelta(days=366))
        diff_sec=round((gps_epoch-metadata).total_seconds(),1)
    except (ValueError, OverflowError):
        pass
    return windows, diff_sec


def parse_events(raw_path):
    src=open_csv(raw_path)
    required = ['Start Date','End Date','Start Time','End Time','Location',
                'Track or lane being worked on','Type of Work']
    if not src or not set(required).issubset(src[0]):
        raise ValueError('maintenance log columns missing')
    seen = {}
    events=[]
    for r in src:
        k=tuple(r[c] for c in required)
        if k in seen:
            seen[k]['source_repeats']+=1
            continue
        i=len(seen)+1
        e={'record_id':f'M{i:04d}','source_repeats':1,'source_work':r['Type of Work'],
           'source_location':r['Location']}
        seen[k]=e
        v=' '.join(r['Location'].strip().lower().replace('-',' ').split())
        relevant=('denise' in v and 'glenbury' in v)
        if not relevant:
            continue
        try:
            start=datetime.strptime(r['Start Date'].strip(),'%m/%d/%Y').date()
            end=datetime.strptime(r['End Date'].strip(),'%m/%d/%Y').date()
        except ValueError:
            continue
        direction=r['Track or lane being worked on'].strip().lower()
        if direction not in ('inbound','outbound'):
            if 'inbound' in direction and 'outbound' in direction:
                direction='both'
            else:
                direction='unknown'
        work=r['Type of Work'].lower()
        activity='tamping' if re.search(r'\btamp(?:ing|ed|s)?\b',work) else 'surfacing' if re.search(r'\bsurfac(?:e|ing)\b',work) else 'other'
        e.update(start_date=start.isoformat(),end_date=end.isoformat(),
                 direction=direction,activity=activity)
        events.append(e)
    return events


def coverage_for_events(events, day_passages):
    result=[]
    for e in events:
        ds=date.fromisoformat(e['start_date']); de=date.fromisoformat(e['end_date'])
        dirs=['inbound','outbound'] if e['direction'] == 'both' else [e['direction']]
        for direction in dirs:
            before=defaultdict(set);during=defaultdict(set);after=defaultdict(set)
            for (day, d),passes in day_passages.items():
                if d != direction:
                    continue
                dt=date.fromisoformat(day)
                if ds-timedelta(days=30)<=dt<ds:
                    before[day].update(passes)
                elif ds<=dt<=de:
                    during[day].update(passes)
                elif de<dt<=de+timedelta(days=30):
                    after[day].update(passes)
            interfering=[]
            for other in events:
                if other['record_id'] == e['record_id']:
                    continue
                if other['direction'] not in ('both',direction,'unknown'):
                    continue
                od=date.fromisoformat(other['start_date'])
                oe=date.fromisoformat(other['end_date'])
                if oe >= ds-timedelta(days=30) and od<=de+timedelta(days=30):
                    interfering.append(other['record_id'])
            result.append({
                'record_id':e['record_id'],'start_date':e['start_date'],
                'end_date':e['end_date'],'direction':direction,'activity':e['activity'],
                'source_work':e['source_work'],'source_location':e['source_location'],
                'source_repeats':e['source_repeats'],
                'localisation':'provisional crossover corridor, NOT work footprint',
                'before_days':len(before),'before_passages':len(set().union(*before.values())) if before else 0,
                'during_days':len(during),'during_passages':len(set().union(*during.values())) if during else 0,
                'after_days':len(after),'after_passages':len(set().union(*after.values())) if after else 0,
                'other_schedules_within_30d':len(interfering),
                'other_schedule_ids':';'.join(sorted(interfering)),
                'min_4days_each_side':int(len(before)>=4 and len(after)>=4),
                'label_reliability':'scheduled_not_confirmed; spatial_extent_unverified',
            })
    return result


def episodes(dates):
    days=sorted({date.fromisoformat(x) for x in dates})
    groups=[]
    for day in days:
        if not groups or (day-groups[-1][-1]).days>7:
            groups.append([day])
        else:
            groups[-1].append(day)
    return [dict(first=x[0].isoformat(),last=x[-1].isoformat(),days=len(x)) for x in groups]


def execute(args):
    data=args.data_root.resolve()
    output=args.output_dir.resolve()
    metadata=data/'LRV4306'/'LRV4306_metadata.csv'
    raw_log=data/'track_maintenance_logs'/'track_logs.csv'
    required=[metadata,raw_log,
              data/'ground_truth_rail_layout_GPS'/'region5_inbound.csv',
              data/'ground_truth_rail_layout_GPS'/'region5_outbound.csv']
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f'Required ORIGINAL input not found: {p}')
    region=catalogue(metadata,data)
    complete=[r for r in region if r['complete_file_set']]
    frozen=[r for r in complete if date.fromisoformat(r['local_date'])>CUTOFF+timedelta(days=EMBARGO_DAYS)]
    actual_hash=hashlib.sha256('\n'.join(sorted(r['passage_id'] for r in frozen)).encode()).hexdigest()
    if not args.allow_holdout_manifest_mismatch and (len(frozen)!=EXPECTED_HOLDOUT_COUNT or actual_hash!=EXPECTED_HOLDOUT_ID_SHA256):
        raise RuntimeError(
            f'FROZEN HOLDOUT MISMATCH: expected {EXPECTED_HOLDOUT_COUNT} IDs and '
            f'{EXPECTED_HOLDOUT_ID_SHA256}, got {len(frozen)} IDs and {actual_hash}. '
            'No output or waveform read. Investigate the raw metadata. Use the '
            'override only if a NEW evaluation protocol is formally approved.')
    dev=sorted([r for r in complete if date.fromisoformat(r['local_date'])<=CUTOFF],
               key=lambda r:(r['metadata_utc'],r['direction'],r['passage_id']))
    embargo=[r for r in complete if CUTOFF<date.fromisoformat(r['local_date'])<=CUTOFF+timedelta(days=EMBARGO_DAYS)]
    print('Original Region 5 catalogue:',len(region),'complete:',len(complete),flush=True)
    print('Development raw candidates:',len(dev),'embargo:',len(embargo),
          'reserved holdout:',len(frozen),'SHA256:',actual_hash,flush=True)
    print('Holdout files are not inspected, even for GPS QC.',flush=True)
    if args.check_inputs:
        print('INPUT CHECK ONLY: no raw GPS or acceleration opened, no outputs written.')
        return
    if not dev:
        raise RuntimeError('No development recordings')
    if output.exists() and any(output.glob('development_vibration_features.csv.gz')) and not args.replace:
        raise RuntimeError(f'Existing output at {output}; specify --replace to overwrite ONLY this standalone output.')
    if args.limit_passages:
        dev=dev[:args.limit_passages]
    tracks={d:make_track(data,d) for d in ['inbound','outbound']}
    events=parse_events(raw_log)
    if len(events)!=27:
        raise RuntimeError(f'Expected 27 Denise–Glenbury distinct schedule rows; got {len(events)}. No signals read.')
    output.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    tmp=output/'development_vibration_features.csv.gz.partial'
    counters=Counter(); reasons=Counter(); day_passages=defaultdict(set)
    bin_passages=defaultdict(set)
    speed_values=[]; qc_residuals=[]; gps_collision={}; valid=[]
    error_examples=[]
    try:
        with gzip.open(tmp,'wt',newline='',encoding='utf-8',compresslevel=5) as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader()
            for idx,r in enumerate(dev,1):
                pid=r['passage_id']; direction=r['direction']
                try:
                    # The chronological split above MUST happen before any raw-file open.
                    g=read_gps(data/r['vehicle']/'gps_data'/r['gps_file'])
                    key=(direction,array_digest(g))
                    if key in gps_collision:
                        same=[]
                        for passage in (r,gps_collision[key]):
                            h=hashlib.sha256()
                            for ch in range(1,6):
                                p=data/passage['vehicle']/'accelerometer_data'/passage[f'acc_sensor_{ch}_file']
                                h.update(array_digest(load_acc(p)).encode('ascii'))
                            same.append(h.hexdigest())
                        if same[0]==same[1]:
                            reasons['confirmed_duplicate']+=1
                            continue
                        reasons['same_gps_different_sensors_review']+=1
                        continue  # conservative: cannot resolve GPS reuse independently
                    gps_collision[key]=r
                    apaths=[data/r['vehicle']/'accelerometer_data'/r[f'acc_sensor_{j}_file'] for j in range(1,6)]
                    lens=[acc_length(p) for p in apaths]
                    windows,gps_dt=build_windows(r,g,tracks[direction],lens)
                    rows=[]
                    # Read each sensor once; compute features from its own indices.
                    for v in windows:
                        rec={k:v.get(k,'') for k in META}
                        rec.update(passage_id=pid,vehicle='LRV4306',region=5,
                                   direction=direction,metadata_utc=r['metadata_utc'],
                                   local_date=r['local_date'],gps_metadata_start_difference_s=gps_dt,
                                   n_sensors_with_features=0,
                                   raw_acc_time_alignment='assumed_common_first_last_samples_unverified',
                                   analysis_screening_flag='speed_screen_pass' if v['estimated_speed_mps']>=MIN_SPEED_MPS else 'review_low_speed')
                        rows.append(rec)
                    for sensor_id,ap in enumerate(apaths,1):
                        x=load_acc(ap)
                        for v,rec in zip(windows,rows):
                            a,b=sensor_window(v['ia'],v['ib'],lens[0],len(x))
                            try:
                                vals=feature_vector(x[a:b])
                                for name,val in zip(FEATURES,vals):
                                    rec[f's{sensor_id}_{name}']=round(float(val),9)
                                rec['n_sensors_with_features']+=1
                            except (ValueError,OverflowError,FloatingPointError):
                                counters['sensor_window_feature_failures']+=1
                        del x
                    all_count=0
                    for rec in rows:
                        if rec['n_sensors_with_features']!=5:
                            counters['incomplete_windows']+=1
                            continue
                        all_count+=1
                        if rec['analysis_screening_flag']=='review_low_speed':
                            counters['low_speed_windows']+=1
                        else:
                            speed_values.append(float(rec['estimated_speed_mps']))
                            qc_residuals.append(float(rec['gps_median_track_residual_m']))
                            if rec['corridor_candidate_bin']=='yes':
                                day_passages[(rec['local_date'], direction)].add(pid)
                            bin_passages[rec['track_bin_id']].add(pid)
                            counters['speed_screened_windows']+=1
                        w.writerow(rec)
                        counters['complete_windows']+=1
                    if all_count:
                        valid.append(r)
                        counters['processed_passages']+=1
                    else:
                        reasons['no_complete_windows']+=1
                except (OSError,ValueError,KeyError,IndexError,TypeError,RuntimeError) as exc:
                    key_name=type(exc).__name__ + ':' + str(exc)[:90]
                    reasons[key_name]+=1
                    if len(error_examples)<20:
                        error_examples.append({'passage_id':pid,'error':key_name})
                if idx%20==0 or idx==len(dev):
                    print(f'{idx}/{len(dev)} processed; accepted={counters["processed_passages"]}, '
                          f'complete_windows={counters["complete_windows"]}',flush=True)
        event_rows=coverage_for_events(events,day_passages)
        event_tmp=output/'maintenance_event_coverage.csv.partial'
        with event_tmp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=EVENT_FIELDS)
            writer.writeheader(); writer.writerows(event_rows)
        all_dates={r['local_date'] for r in valid}
        reserved_ids=sorted([r['passage_id'] for r in frozen])
        record={'version':VERSION,'purpose':'Initial platform-ready DEVELOPMENT data; no maintenance prediction trained',
                'scope':'LRV4306, Region 5, both travel directions',
                'original_inputs_sha256':{str(p):digest(p) for p in required},
                'source_directories':{'data_root':str(data)},
                'chronological_split':{
                    'development_last_date':CUTOFF.isoformat(),
                    'seven_day_embargo_end':(CUTOFF+timedelta(days=EMBARGO_DAYS)).isoformat(),
                    'complete_heldout_recordings':len(frozen),
                    'holdout_id_sha256':actual_hash,
                    'holdout_passage_ids':reserved_ids,
                    'holdout_first_local_date':min(r['local_date'] for r in frozen) if frozen else None,
                    'holdout_last_local_date':max(r['local_date'] for r in frozen) if frozen else None,
                    'holdout_raw_files_read':False,
                    'embargo_recordings':len(embargo),
                    'holdout_episodes':episodes(r['local_date'] for r in frozen),
                },
                'historical_stage31_checkpoint_not_a_target': {
                    'previous_development_passages': 428,
                    'previous_complete_windows': 23189,
                    'previous_speed_screened_windows': 21781,
                    'interpretation': 'Standalone full-pool reconstruction can differ because older stages used separate candidate screening and duplicate audits; compare counts before scientific use.'
                },
                'counts':{
                    'region5_catalogue_entries':len(region),
                    'region5_complete_file_sets':len(complete),
                    'development_complete_candidates':len(dev),
                    'development_accepted_passages':len(valid),
                    'development_distinct_dates':len(all_dates),
                    'development_distinct_direction_days':len({(r['local_date'],r['direction']) for r in valid}),
                    'development_complete_feature_windows':int(counters['complete_windows']),
                    'development_speed_screened_feature_windows':int(counters['speed_screened_windows']),
                    'direction_specific_100m_bins_with_screened_features':len(bin_passages),
                    'unique_denise_glenbury_schedule_rows':len(events),
                    'event_direction_rows':len(event_rows),
                    'events_with_4_days_each_side':sum(int(e['min_4days_each_side']) for e in event_rows),
                },
                'source_processing':'Standalone implementation of Stage 02 catalogue, robust Stage 04 ICP 100 m mapping, exact array duplicate fingerprints and Stage 05 22 features x 5 channels',
                'feature_definitions':FEATURES,
                'feature_columns':[f's{i}_{name}' for i in range(1,6) for name in FEATURES],
                'geometry':{'reference_bin_m':BIN_M,'candidate_crossovers':CROSSOVER_COORDS,
                            'candidate_coordinate_note':CROSSOVER_NOTE,
                            'crossovers_on_reference':{d:t['landmarks'] for d,t in tracks.items()},
                            'corridor_chainage_bounds':{d:t['corridor'] for d,t in tracks.items()}},
                'QC':{'gps_max_gap_s':3.0,'gps_median_track_residual_max_m':15.0,
                      'gps_p95_track_residual_max_m':30.0,'icp_translation_max_m':100,
                      'icp_rotation_max_degrees':5,
                      'acceleration_to_gps_sample_ratio_allowed':'1630 < samples/GPS_point < 2100',
                      'nominal_acc_sampling_hz':FS,'original_gps_sampling_hz':1,
                      'minimum_samples_per_window':1000,'minimum_analysis_speed_mps':MIN_SPEED_MPS,
                      'speed_screened_median_mps':round(float(np.median(speed_values)),3) if speed_values else None,
                      'speed_screened_median_gps_residual_m':round(float(np.median(qc_residuals)),3) if qc_residuals else None,
                      'exclusions':dict(reasons),'other_counters':dict(counters),
                      'processing_error_examples':error_examples},
                'research_status':{
                    'maintenance_work_date':'Scheduled, NOT verified completed',
                    'track_work_footprint':'Unknown; Denise–Glenbury corridor only approximate',
                    'physical_fault_labels':'Unavailable',
                    'speed_confounding':'Known; lower-speed features flagged not corrected',
                    'gps_absolute_clock':'GPS fifth-column assumed comparable to metadata UTC, source clock independently unverified',
                    'gps_to_acc_absolute_offset':'Unknown: same first-last recording alignment assumed',
                    'holdout':'Preserved, no held-out waveform analysed',
                    'correlated_samples':'Windows and passages from same date are dependent; use date and episode grouping',
                    'model_validation':'No detector or predictive model fitted or validated in this program',
                    'scope_note':'Does not re-run exploratory Stages 06-27 or LRV4313; consult handoff for those results',
                    'full_run_not_yet_verified_on_cluster':True,
                },
                'output_contents':{
                    'development_vibration_features.csv.gz':'All complete 5-sensor QC-passing development windows, with low-speed flags (filter to speed_screen_pass for modelling)',
                    'maintenance_event_coverage.csv':'Every Denise–Glenbury scheduled record, direction-specific counts inside the candidate corridor within +/-30 days',
                    'foundation_manifest.json':'Split, frozen holdout identifiers, QC, feature specification, history and explicit caveats',
                },'elapsed_seconds':round(time.monotonic()-t0,1)}
        manifest_tmp=output/'foundation_manifest.json.partial'
        manifest_tmp.write_text(json.dumps(record,indent=2,default=str)+'\n',encoding='utf-8')
        # Atomic finalisation, no completed output when a processing exception occurs.
        os.replace(tmp,output/'development_vibration_features.csv.gz')
        os.replace(event_tmp,output/'maintenance_event_coverage.csv')
        os.replace(manifest_tmp,output/'foundation_manifest.json')
        print('COMPLETE: independently rebuilt 3 foundation files in',output,flush=True)
        print('Development passages:',len(valid),'feature windows:',counters['complete_windows'],
              'speed-screened:',counters['speed_screened_windows'],flush=True)
        print('Schedules:',len(events),'holdout unopened:',len(frozen),flush=True)
    except BaseException:
        # Avoid representing partial feature files as a successful run.
        if tmp.exists():
            tmp.unlink()
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root',type=Path,default=Path('/data2/nr1/data_files'))
    ap.add_argument('--output-dir',type=Path,default=Path('/data2/nr1/V1_RailTrack/FOUNDATION_for_new_chat'))
    ap.add_argument('--check-inputs',action='store_true',help='Metadata-only verification; no raw sensor or GPS data opened')
    ap.add_argument('--limit-passages',type=int,default=0,help='DEVELOPMENT diagnostic subset; 0 processes all')
    ap.add_argument('--replace',action='store_true',help='Allow overwriting the 3 isolated output files')
    ap.add_argument('--allow-holdout-manifest-mismatch',action='store_true',
                    help='Expert override only after an approved new freeze; not for standard use')
    args=ap.parse_args()
    if args.limit_passages < 0:
        ap.error('negative limit')
    try:
        execute(args)
    except (OSError,KeyError,ValueError,RuntimeError) as exc:
        print(f'FAILED: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(2)


if __name__=='__main__':
    main()
