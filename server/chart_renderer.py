"""
Server-side chart renderer for PDF reports.

Uses matplotlib with the Agg backend (headless) to generate PNG images
of the DFA α1 threshold chart, matching the visual style of the
browser-based Chart.js charts.
"""

import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# COLOUR PALETTE (print-optimised — white background, dark text)
# ---------------------------------------------------------------------------
ZONE1_FILL = (0.13, 0.77, 0.37, 0.12)   # green tint
ZONE2_FILL = (0.92, 0.70, 0.03, 0.12)   # yellow tint
ZONE3_FILL = (0.94, 0.27, 0.27, 0.10)   # red tint

ZONE1_TEXT = (0.13, 0.77, 0.37, 0.45)
ZONE2_TEXT = (0.80, 0.60, 0.02, 0.45)
ZONE3_TEXT = (0.94, 0.27, 0.27, 0.40)

RAW_DOT = (0.39, 0.40, 0.95, 0.12)
STAGE_BLUE = '#3b82f6'
STAGE_BORDER = '#1d4ed8'
ERR_BAR_COLOUR = (0.23, 0.51, 0.97, 0.5)

REG_COLOUR = '#8b5cf6'
REG_EXTRAP = '#8b5cf6'

HRVT1S_COLOUR = '#3b82f6'
HRVT1C_COLOUR = '#22c55e'
HRVT2_COLOUR = '#ef4444'

REFLINE_COLOUR = (0.6, 0.6, 0.6, 0.25)
INFO_BOX_BG = (0.95, 0.95, 0.95, 0.85)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _speed_to_pace_str(speed_ms):
    """Convert m/s to M:SS/km string."""
    if not speed_ms or speed_ms <= 0:
        return '--:--'
    pace_sec = 1000.0 / speed_ms
    m = int(pace_sec // 60)
    s = int(pace_sec % 60)
    return f'{m}:{s:02d}'


def _pace_formatter(speed_ms, _pos):
    """Matplotlib tick formatter: speed (m/s) → pace string."""
    return _speed_to_pace_str(speed_ms)


# ---------------------------------------------------------------------------
# MAIN CHART: DFA α1 vs Power (or Pace)
# ---------------------------------------------------------------------------
def render_threshold_chart(stage_data, regression, thresholds, windows,
                           effort_validation, is_run=False):
    """
    Render the DFA α1 vs Power (cycling) or vs Pace (running) chart.

    Args:
        stage_data:  list of stage dicts with mean_power, a1_mean, a1_sd, stage_number
        regression:  dict with slope, intercept, r2, n
        thresholds:  dict with hrvt1s_power, hrvt1c_power, hrvt2_power, a1_star, a1_max_early, etc.
        windows:     list of window dicts with power, alpha1, reliable
        effort_validation: dict with avg_power (may be None)
        is_run:      if True, x-axis shows pace labels

    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # --- Filter valid data ---
    valid_stages = [s for s in (stage_data or [])
                    if s.get('a1_mean') is not None and s.get('mean_power') is not None]
    if not valid_stages:
        plt.close(fig)
        return _empty_chart_png()

    stage_x = [s['mean_power'] for s in valid_stages]
    stage_y = [s['a1_mean'] for s in valid_stages]
    stage_sd = [s.get('a1_sd', 0) or 0 for s in valid_stages]
    stage_nums = [str(s.get('stage_number', '')) for s in valid_stages]

    min_x = min(stage_x) - (10 if not is_run else min(stage_x) * 0.08)
    max_x = max(stage_x) + (10 if not is_run else max(stage_x) * 0.08)

    effort_power = (effort_validation or {}).get('avg_power')
    hrvt2 = thresholds.get('hrvt2_power')

    if effort_power:
        chart_x_max = max(effort_power + 10, (hrvt2 or 0) + 30) if not is_run \
            else max(effort_power * 1.05, (hrvt2 or 0) * 1.1)
    else:
        chart_x_max = max(max_x + 30, (hrvt2 or max_x) + 40) if not is_run \
            else max(max_x * 1.1, (hrvt2 or max_x) * 1.08)

    chart_x_min = min_x

    # --- Raw windows scatter ---
    raw_pts = [(w['power'], w['alpha1']) for w in (windows or [])
               if w.get('power') is not None and w.get('alpha1') is not None
               and w.get('reliable', True)]
    if raw_pts:
        rx, ry = zip(*raw_pts)
        ax.scatter(rx, ry, s=8, c=[RAW_DOT], edgecolors='none',
                   zorder=1, label='Raw Windows')

    # --- Zone fills ---
    t1c = thresholds.get('hrvt1c_power')
    t1s = thresholds.get('hrvt1s_power')
    t2 = hrvt2
    z_bound = t1c or t1s

    if z_bound:
        ax.axvspan(chart_x_min, z_bound, color=ZONE1_FILL, zorder=0)
        ax.text((chart_x_min + z_bound) / 2, 0.18, 'Zone 1',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=ZONE1_TEXT, zorder=0)
    if z_bound and t2:
        ax.axvspan(z_bound, t2, color=ZONE2_FILL, zorder=0)
        ax.text((z_bound + t2) / 2, 0.18, 'Zone 2',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=ZONE2_TEXT, zorder=0)
    elif z_bound:
        ax.axvspan(z_bound, chart_x_max, color=ZONE2_FILL, zorder=0)
    if t2:
        ax.axvspan(t2, chart_x_max, color=ZONE3_FILL, zorder=0)
        ax.text((t2 + chart_x_max) / 2, 0.18, 'Zone 3',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=ZONE3_TEXT, zorder=0)

    # --- Regression line ---
    slope = (regression or {}).get('slope')
    intercept = (regression or {}).get('intercept')
    if slope is not None and intercept is not None:
        # Solid line over observed range
        reg_x = np.linspace(chart_x_min, max_x, 200)
        reg_y = slope * reg_x + intercept
        mask = (reg_y >= 0) & (reg_y <= 1.5)
        ax.plot(reg_x[mask], reg_y[mask], color=REG_COLOUR, linewidth=2,
                zorder=3, label='Regression')

        # Dashed extrapolation beyond observed data
        extrap_limit = chart_x_max if effort_power else max(max_x + 30, (t2 or max_x) + 30)
        ext_x = np.linspace(max_x, extrap_limit, 100)
        ext_y = slope * ext_x + intercept
        mask_e = (ext_y >= 0) & (ext_y <= 1.5)
        if mask_e.any():
            ax.plot(ext_x[mask_e], ext_y[mask_e], color=REG_EXTRAP, linewidth=1.5,
                    linestyle='--', zorder=3)

    # --- Stage scatter with error bars ---
    ax.errorbar(stage_x, stage_y, yerr=stage_sd,
                fmt='o', markersize=7, markerfacecolor=STAGE_BLUE,
                markeredgecolor=STAGE_BORDER, markeredgewidth=1.5,
                ecolor=ERR_BAR_COLOUR, elinewidth=1.2, capsize=3,
                zorder=5, label='Ramp Stages')

    # Stage number labels above points
    for x, y, sd, num in zip(stage_x, stage_y, stage_sd, stage_nums):
        ax.annotate(num, (x, y), textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8, fontweight='bold', color='#374151',
                    zorder=6)

    # --- Threshold vertical lines ---
    if t1s:
        ax.axvline(t1s, color=HRVT1S_COLOUR, linewidth=1.2, linestyle='--',
                   alpha=0.6, zorder=4)
    if t1c and thresholds.get('a1_star'):
        ax.axvline(t1c, color=HRVT1C_COLOUR, linewidth=1, linestyle='--',
                   alpha=0.5, zorder=4)
    if t2:
        ax.axvline(t2, color=HRVT2_COLOUR, linewidth=1.2, linestyle='--',
                   alpha=0.6, zorder=4)

    # Threshold points on the regression line
    if t1s:
        ax.scatter([t1s], [0.75], s=60, c=HRVT1S_COLOUR, edgecolors=STAGE_BORDER,
                   linewidths=1.5, zorder=7)
    if t1c and thresholds.get('a1_star'):
        ax.scatter([t1c], [thresholds['a1_star']], s=50, c=HRVT1C_COLOUR,
                   edgecolors='#15803d', linewidths=1.5, zorder=7)
    if t2:
        ax.scatter([t2], [0.50], s=60, c=HRVT2_COLOUR, edgecolors='#b91c1c',
                   linewidths=1.5, zorder=7)

    # Threshold labels at bottom
    label_y = 0.06
    if t1s:
        unit = '' if is_run else 'W'
        lbl = f'HRVT1s  {_speed_to_pace_str(t1s)}' if is_run else f'HRVT1s  {t1s:.0f}{unit}'
        ax.annotate(lbl, (t1s, label_y), ha='center', fontsize=7,
                    fontweight='bold', color='white', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.25', fc=HRVT1S_COLOUR, alpha=0.85))
    if t1c and thresholds.get('a1_star'):
        lbl = f'HRVT1c  {_speed_to_pace_str(t1c)}' if is_run else f'HRVT1c  {t1c:.0f}W'
        # Offset if close to t1s
        offset_y = label_y + 0.08 if t1s and abs(t1c - t1s) < (20 if not is_run else 0.3) else label_y
        ax.annotate(lbl, (t1c, offset_y), ha='center', fontsize=7,
                    fontweight='bold', color='white', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.25', fc=HRVT1C_COLOUR, alpha=0.85))
    if t2:
        lbl = f'HRVT2  {_speed_to_pace_str(t2)}' if is_run else f'HRVT2  {t2:.0f}W'
        ax.annotate(lbl, (t2, label_y), ha='center', fontsize=7,
                    fontweight='bold', color='white', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.25', fc=HRVT2_COLOUR, alpha=0.85))

    # --- Reference lines at α1 = 0.75 and 0.50 ---
    ax.axhline(0.75, color=REFLINE_COLOUR, linewidth=0.8, linestyle='--', zorder=2)
    ax.axhline(0.50, color=REFLINE_COLOUR, linewidth=0.8, linestyle='--', zorder=2)

    # --- R² info box ---
    r2 = (regression or {}).get('r2')
    n_stages = (regression or {}).get('n')
    if r2 is not None:
        info = f'R² = {r2:.3f}'
        if n_stages:
            info += f'\nn = {n_stages} stages'
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc=INFO_BOX_BG, ec='#d1d5db'),
                zorder=10)

    # --- Axes ---
    a1_max = thresholds.get('a1_max_early') or 1.2
    ax.set_xlim(chart_x_min, chart_x_max)
    ax.set_ylim(0, a1_max + 0.2)

    if is_run:
        ax.set_xlabel('Pace (min:sec / km)', fontsize=10, color='#374151')
        ax.xaxis.set_major_formatter(FuncFormatter(_pace_formatter))
        # Faster pace = higher speed = right side; invert for natural reading
        # Actually for running charts speed increases left-to-right which maps to
        # faster pace, so no inversion needed — pace labels handle it
    else:
        ax.set_xlabel('Power (W)', fontsize=10, color='#374151')

    ax.set_ylabel('DFA \u03b11', fontsize=10, color='#374151')
    ax.tick_params(colors='#6b7280', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d1d5db')
    ax.spines['bottom'].set_color('#d1d5db')
    ax.set_title('DFA \u03b11 vs ' + ('Pace' if is_run else 'Power'),
                 fontsize=11, fontweight='bold', color='#1f2937', pad=10)

    # Legend
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8,
              edgecolor='#d1d5db', fancybox=True)

    fig.tight_layout(pad=0.8)

    # --- Export as PNG bytes ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _empty_chart_png():
    """Return a minimal PNG with a 'No data' message."""
    fig, ax = plt.subplots(figsize=(7.0, 2.0), dpi=100)
    fig.patch.set_facecolor('white')
    ax.text(0.5, 0.5, 'Insufficient data to generate chart',
            ha='center', va='center', fontsize=12, color='#9ca3af',
            transform=ax.transAxes)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
