"""Publication-quality redraw of Figs. 1, 5 and 6a from the saved development outputs.

Reads only the CSV outputs of the frozen pipeline (folder develop) and the foundation's
maintenance_event_coverage.csv; the pipeline itself is not imported or modified.
Writes, for each figure, a vector PDF, an SVG whose text uses Times New Roman, a 1000 dpi TIFF
and a 600 dpi PNG. Text is set in Liberation Serif, which has the same metrics as Times New Roman.
"""
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEV = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/claude/railtrack/out_full/develop')
COVERAGE = Path(sys.argv[2] if len(sys.argv) > 2 else '/home/claude/railtrack/foundation/maintenance_event_coverage.csv')
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else '/home/claude/paper/figs/out')
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WIDTH = 160 * MM          # manuscript text width (A4, 2.5 cm margins)
BLACK = '#000000'
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
GREY_FAIL, SPAN, GRID = '#8a8a8a', '#e1e0d9', '#e9e9e9'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'text.color': BLACK, 'axes.labelcolor': BLACK, 'axes.titlecolor': BLACK,
    'xtick.color': BLACK, 'ytick.color': BLACK, 'axes.edgecolor': BLACK, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'axes.unicode_minus': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white',
})


def minus(x, fmt):
    return format(x, fmt).replace('-', '−')


def save(fig, stem):
    for ext, kw in (('pdf', {}), ('svg', {}), ('tiff', {'dpi': 1000, 'pil_kwargs': {'compression': 'tiff_lzw'}}),
                    ('png', {'dpi': 600})):
        fig.savefig(OUT / f'{stem}.{ext}', bbox_inches='tight', pad_inches=0.03, **kw)
    svg = OUT / f'{stem}.svg'
    txt = svg.read_text(encoding='utf8')
    txt = re.sub(r"font-family:[^;\"]*", "font-family: 'Times New Roman', 'Liberation Serif', serif", txt)
    svg.write_text(txt, encoding='utf8')
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1: channel levels
def fig_levels():
    pc = pd.read_csv(DEV / '01_audit_passage_channel.csv', parse_dates=['local_date'])
    rg = pd.read_csv(DEV / '01_audit_regimes.csv', parse_dates=['first_date', 'last_date'])
    cov = pd.read_csv(COVERAGE, parse_dates=['start_date', 'end_date'])
    names = {1: 'bogie longitudinal (tri-axial X)', 2: 'bogie transverse (tri-axial Y)',
             3: 'bogie vertical (tri-axial Z)', 4: 'front cabin vertical (uni-axial)',
             5: 'back cabin vertical (uni-axial)'}
    fig, axes = plt.subplots(5, 1, figsize=(WIDTH, 205 * MM), sharex=True)
    for k, (ax, c) in enumerate(zip(axes, range(1, 6))):
        for _, r in cov.iterrows():
            ax.axvspan(r.start_date, r.end_date + pd.Timedelta(days=1), color=SPAN, lw=0, zorder=0)
        d = pc[pc.channel == c]
        ok = d.qc_pass == 'yes'
        ax.scatter(d.local_date[~ok], d.level_log_rms[~ok], s=7, facecolors='none', edgecolors=GREY_FAIL,
                   linewidths=0.5, zorder=2)
        ax.scatter(d.local_date[ok], d.level_log_rms[ok], s=5, color=BLUE, lw=0, zorder=3)
        for _, r in rg[rg.channel == c].iterrows():
            ax.plot([r.first_date, r.last_date], [r.median_level_log] * 2, color=ORANGE, lw=1.6, zorder=4,
                    solid_capstyle='round')
        ax.set_title(f'({"abcde"[k]}) s{c}, {names[c]}', loc='left', pad=3)
        ax.set_ylabel('Level (log RMS)')
        ax.grid(axis='x', visible=False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].set_xlim(pd.Timestamp('2013-11-01'), pd.Timestamp('2016-01-10'))
    handles = [Line2D([], [], marker='o', ls='', color=BLUE, ms=3.5, label='Passage, QC pass'),
               Line2D([], [], marker='o', ls='', mfc='none', mec=GREY_FAIL, mew=0.6, ms=3.5, label='Passage, QC fail'),
               Line2D([], [], color=ORANGE, lw=1.6, label='Working-regime median'),
               Patch(color=SPAN, label='Scheduled corridor work')]
    fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.4,
               columnspacing=1.4)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.6)
    save(fig, 'Fig1_channel_levels')


# ---------------------------------------------------------------- Fig. 5: scheduled-work events
def fig_events():
    key = 's45_bandpower_0p5_5hz'
    b = pd.read_csv(DEV / '05_events_bins.csv')
    s = pd.read_csv(DEV / '05_events_summary.csv')
    pr = pd.read_csv(DEV / '07_priority_list.csv')
    corridor = {d: (g[g.corridor == 'yes'].chainage_start_m.min(), g[g.corridor == 'yes'].chainage_start_m.max() + 100)
                for d, g in pr.groupby('direction')}
    events = [('M0032', 'outbound'), ('M0082', 'inbound'), ('M0144', 'inbound')]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 150 * MM), sharex=True)
    for k, (ax, (rid, d)) in enumerate(zip(axes, events)):
        g = b[(b.indicator == key) & (b.record_id == rid) & (b.direction == d)].sort_values('chainage_start_m')
        x = g.chainage_start_m.to_numpy(float) + 50
        y = g.change.to_numpy(float)
        sd = g.null_sd_calibrated.to_numpy(float)
        sig = (g.fdr_significant == 'yes').to_numpy()
        ax.fill_between(x, -1.645 * sd, 1.645 * sd, color=SPAN, lw=0, zorder=1)
        a0, a1 = corridor[d]
        ax.axvspan(a0, a1, color=BLUE, alpha=0.08, lw=0, zorder=0)
        ax.axhline(0, color=BLACK, lw=0.6, zorder=2)
        ax.plot(x, y, color=BLUE, lw=1.0, marker='o', ms=2.4, zorder=3)
        if sig.any():
            ax.scatter(x[sig], y[sig], s=22, color=RED, zorder=4, edgecolors='white', linewidths=0.6)
        r = s[(s.indicator == key) & (s.record_id == rid) & (s.direction == d)].iloc[0]
        act = r.activity
        stat = (f'corridor minus control {minus(r.corridor_minus_control_change, "+.3f")}, '
                f'z = {minus(r.z, ".2f")}, p = {r.p_two_sided:.3f}')
        if r.regime_change_in_window == 'yes':
            stat = 'invalid test, window contains an instrument regime change'
        ax.set_title(f'({"abc"[k]}) {rid}, {d} {act}: {stat}', loc='left', pad=3)
        ax.set_ylabel('Change (log units)')
    axes[-1].set_xlabel('Chainage (m)')
    axes[-1].set_xlim(0, 5700)
    handles = [Patch(color=SPAN, label='90% range expected without an effect'),
               Patch(color=BLUE, alpha=0.15, label='Maintenance corridor (provisional)'),
               Line2D([], [], color=BLUE, lw=1.0, marker='o', ms=2.4, label='Change, after minus before'),
               Line2D([], [], ls='', marker='o', ms=4.5, color=RED, label='FDR-significant bin')]
    fig.legend(handles=handles, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0), handletextpad=0.5,
               columnspacing=2.0)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8)
    save(fig, 'Fig5_events')


# ---------------------------------------------------------------- Fig. 6a: detection probability
def fig_detection():
    p = pd.read_csv(DEV / '06_detection_power.csv')
    p = p[(p.scope == 'development') & (p.indicator == 's45_bandpower_0p5_5hz') & (p.days_after == 3)]
    fig, ax = plt.subplots(figsize=(WIDTH, 68 * MM))
    for L, col in ((1, BLUE), (3, ORANGE), (5, AQUA)):
        g = p[p.segment_bins == L].sort_values('step_log')
        x = g.step_percent_amplitude_equivalent.to_numpy()
        keep = (x >= -60) & (x <= 150)
        ax.plot(x[keep], g.detection_probability.to_numpy()[keep], color=col, lw=1.6, label=f'{L * 100} m segment')
    ax.axhline(0.8, color=BLACK, lw=0.6)
    ax.axhline(0.05, color=BLACK, lw=0.6, ls=(0, (3, 3)))
    ax.text(149, 0.815, '80% detection', ha='right', va='bottom', fontsize=7.5)
    ax.text(149, 0.065, '5% alarm burden', ha='right', va='bottom', fontsize=7.5)
    ax.set_xlim(-60, 150)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([-50, -25, 0, 25, 50, 75, 100, 125, 150])
    ax.set_xlabel('Injected persistent change, amplitude equivalent (%)')
    ax.set_ylabel('Detection probability')
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.42), handlelength=1.8)
    fig.tight_layout()
    save(fig, 'Fig6a_detection')


fig_levels()
fig_events()
fig_detection()
print('written to', OUT)
