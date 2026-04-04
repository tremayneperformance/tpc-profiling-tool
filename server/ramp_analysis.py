"""
DFA Alpha1 Multi-Segment Ramp Test Analysis Engine

Analyses structured ramp test files following a prescribed protocol:
  20-min warm-up → 10×3-min step ramp → 8-min recovery → 3-min all-out → cooldown

Produces:
  - Segment auto-detection (warm-up, ramp stages, recovery, max effort, cooldown)
  - Per-stage DFA alpha1 computation
  - Three thresholds: HRVT1s (standard), HRVT1c (individualised), HRVT2 (= new FTP)
  - Ramp validation and max effort validation
  - Metabolic archetype classification (High Aerobic / Balanced / High Anaerobic)
  - Data quality report

All detection is power-pattern based — FIT lap markers are never used.
"""

import json
import gzip
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

from dfa_core import (
    parse_fit_file, clean_rr_intervals,
    dfa_alpha1, _interp_value, build_windows
)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Segment detection
ROLLING_WINDOW_SEC = 30
WARMUP_ANALYSIS_START = 5 * 60      # Skip first 5 min for strap settling
WARMUP_ANALYSIS_END = 15 * 60       # Median power from min 5-15
RAMP_SCAN_START = 10 * 60           # Start scanning for ramp from min 10
RAMP_POWER_MULTIPLIER = 1.25        # warmup × 1.25 triggers ramp start
RAMP_SUSTAIN_SEC = 90               # Must hold above threshold for 90s
STEP_CHANGE_PCT = 0.05              # >5% jump signals new stage
STEP_STABILISE_SEC = 20             # Jump must happen within 20s
PLATEAU_MIN_SEC = 150               # Minimum stage duration (2.5 min)
PLATEAU_MAX_SEC = 210               # Maximum stage duration (3.5 min)
RAMP_END_DROP_PCT = 0.40            # >40% drop from peak = ramp end
RAMP_END_WINDOW_SEC = 30            # Drop must happen within 30s
RECOVERY_GAP_MIN_SEC = 5 * 60       # Min 5 min between ramp end and max effort
MAX_EFFORT_POWER_PCT = 0.80         # Spike >80% of ramp peak
MAX_EFFORT_SUSTAIN_SEC = 60         # Must sustain for 60s
COOLDOWN_POWER_MULTIPLIER = 1.3     # Drops below warmup × 1.3
COOLDOWN_SUSTAIN_SEC = 60           # Must stay low for 60s

# Stage analysis
STAGE_DISCARD_SEC = 60              # Discard first 60s of each stage (Rogers: 60-90s)
STAGE_ANALYSIS_SEC = 120            # Use final 120s for DFA a1
WARMUP_A1_WINDOW_SEC = 4 * 60      # Final 4 min of warm-up for a1_max

# Validation thresholds
# Lowered from 8/6 to 6/4 to accommodate early ramp termination
# when athlete reaches HRVT2 (DFA < 0.50 for a full stage).
# A 7-stage ramp ending at HRVT2 is scientifically complete —
# the remaining stages would have been below threshold anyway.
RAMP_STAGES_VALID = 6
RAMP_STAGES_FLAGGED = 4
RAMP_R2_VALID = 0.70
RAMP_R2_FLAGGED = 0.60

# Max effort validation
EFFORT_DUR_MIN_VALID = 165          # seconds
EFFORT_DUR_MIN_FLAGGED = 150
EFFORT_DUR_MAX_VALID = 210
EFFORT_PACING_INVALID = 1.50       # P1/P3 — only extreme fade invalidates
EFFORT_PACING_FLAGGED_HIGH = 1.30  # moderate fade is acceptable
EFFORT_PACING_FLAGGED_LOW = 0.75
EFFORT_POWER_CV_FLAGGED = 0.35

# Composite scoring weights (hidden — never shown to athlete or coach)
AFU_WEIGHT = 5
ATPR_WEIGHT = 3
AR_WEIGHT = 2
COMPOSITE_HIGH_AEROBIC = 7.5
COMPOSITE_BALANCED_LOW = 4.5
FLAGGED_EFFORT_ADJUSTMENT = -0.5

# Fallback classification (ATPR only — no max effort)
ATPR_FALLBACK_HIGH_AEROBIC = 0.85
ATPR_FALLBACK_BALANCED_LOW = 0.75

# Development level (ATPR-based)
DEV_LEVEL_ELITE = 0.85
DEV_LEVEL_ADVANCED = 0.80
DEV_LEVEL_INTERMEDIATE = 0.75

# Race pace bands — percentage of FTP by distance and development level
# Format: {level: {distance: pct_of_ftp}}
RACE_PACE_BANDS = {
    'Beginner': {
        'Sprint': 0.88, 'Olympic': 0.82, '70.3': 0.72, 'Ironman': 0.65,
    },
    'Intermediate': {
        'Sprint': 0.92, 'Olympic': 0.86, '70.3': 0.76, 'Ironman': 0.70,
    },
    'Advanced': {
        'Sprint': 0.95, 'Olympic': 0.90, '70.3': 0.80, 'Ironman': 0.74,
    },
    'Elite': {
        'Sprint': 0.97, 'Olympic': 0.93, '70.3': 0.84, 'Ironman': 0.78,
    },
}

# Ceiling-limited flag
CEILING_LIMITED_AR_THRESHOLD = 0.10

# Run ramp protocol steps — intensity as fraction of threshold speed
RUN_RAMP_INTENSITIES = [0.70, 0.74, 0.78, 0.82, 0.86, 0.91, 0.95, 0.99, 1.03, 1.07]
RUN_STEP_DURATION_SEC = 180

# Run protocol timing
RUN_WARMUP_DURATION = 15 * 60          # 900s
RUN_WARMUP_PCT = 0.60
RUN_RECOVERY_DURATION = 8 * 60         # 480s
RUN_RECOVERY_PCT = 0.30
RUN_TTE_START_OFFSET = 53 * 60         # minute 53 = 3180s
RUN_TTE_MAX_DURATION = 6 * 60          # 360s
RUN_TTE_PCT = 1.20
RUN_COOLDOWN_DURATION = 10 * 60        # 600s
RUN_COOLDOWN_PCT = 0.60

# TTE HR-based end detection
TTE_HR_DROP_BPM = 10
TTE_HR_DROP_SUSTAIN_SEC = 15
TTE_MIN_DURATION_CEILING = 120         # TTE < 120s + ATPR > 0.85 = ceiling hint

# Run race pace bands — percentage of HRVT2 speed
RUN_RACE_PACE_BANDS = {
    'Beginner': {'5K': 0.88, '10K': 0.82, 'Half Marathon': 0.76, 'Marathon': 0.68},
    'Intermediate': {'5K': 0.92, '10K': 0.86, 'Half Marathon': 0.80, 'Marathon': 0.72},
    'Advanced': {'5K': 0.95, '10K': 0.90, 'Half Marathon': 0.84, 'Marathon': 0.76},
    'Elite': {'5K': 0.97, '10K': 0.93, 'Half Marathon': 0.87, 'Marathon': 0.80},
}

# History file
HISTORY_DIR = Path.home() / '.dfatool'
HISTORY_FILE = HISTORY_DIR / 'history.json'
RESULTS_DIR = HISTORY_DIR / 'results'


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS: speed / pace conversion
# ═══════════════════════════════════════════════════════════════════════════

def speed_to_pace_sec(speed_ms):
    """Convert speed in m/s to pace in seconds per km."""
    if speed_ms is None or speed_ms <= 0:
        return None
    return round(1000.0 / speed_ms, 1)

def pace_sec_to_speed(pace_sec):
    """Convert pace in seconds per km to speed in m/s."""
    if pace_sec is None or pace_sec <= 0:
        return None
    return round(1000.0 / pace_sec, 4)

def format_pace(pace_sec):
    """Format pace in seconds/km to 'M:SS' string."""
    if pace_sec is None:
        return '--:--'
    mins = int(pace_sec // 60)
    secs = int(round(pace_sec % 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f'{mins}:{secs:02d}'


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: rolling average
# ═══════════════════════════════════════════════════════════════════════════

def _rolling_avg(times, values, window_sec=ROLLING_WINDOW_SEC):
    """Compute rolling average power over a window. Returns (times, avgs)."""
    t = np.array(times, dtype=float)
    v = np.array(values, dtype=float)
    n = len(t)
    out_t = []
    out_v = []
    for i in range(n):
        # All points within [t[i] - window_sec, t[i]]
        mask = (t >= t[i] - window_sec) & (t <= t[i])
        if mask.sum() > 0:
            out_t.append(float(t[i]))
            out_v.append(float(np.mean(v[mask])))
    return out_t, out_v


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: SEGMENT AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_warmup_power(powers, start_sec=WARMUP_ANALYSIS_START,
                          end_sec=WARMUP_ANALYSIS_END):
    """Compute median power from minutes 5-15 as the warmup baseline."""
    vals = [p for t, p in powers if start_sec <= t <= end_sec]
    if not vals:
        vals = [p for _, p in powers[:600]]  # Fallback: first 600 records
    return float(np.median(vals)) if vals else 100.0


def _detect_ramp_start(roll_t, roll_v, warmup_power, scan_start=RAMP_SCAN_START):
    """
    Find where rolling power exceeds warmup_power × 1.25 and sustains
    for >90 seconds. Returns elapsed seconds or None.
    """
    threshold = warmup_power * RAMP_POWER_MULTIPLIER
    above_start = None

    for i, (t, v) in enumerate(zip(roll_t, roll_v)):
        if t < scan_start:
            continue
        if v >= threshold:
            if above_start is None:
                above_start = t
            elif t - above_start >= RAMP_SUSTAIN_SEC:
                return above_start
        else:
            above_start = None

    return None


def _detect_ramp_end(roll_t, roll_v, ramp_start):
    """
    Detect ramp end: sharp power drop (>40% of peak within 30s)
    after sustained high power.
    Returns elapsed seconds or None.
    """
    ramp_vals = [(t, v) for t, v in zip(roll_t, roll_v) if t >= ramp_start]
    if not ramp_vals:
        return None

    # Find the ramp peak from the ascending phase only.
    # A later max-effort spike (after recovery) must not inflate the peak,
    # otherwise the high-zone threshold becomes unreachable during the
    # actual ramp and the ramp end is never detected.
    running_max = 0.0
    ramp_peak_idx = 0
    for i, (t, v) in enumerate(ramp_vals):
        if v >= running_max:
            running_max = v
            ramp_peak_idx = i
        elif running_max > 0 and v < running_max * 0.60:
            # Major drop — check if sustained (stays below 75% of peak
            # for at least 60 seconds), indicating end of ascending phase
            check_end = t + 60
            sustained = True
            for j in range(i, len(ramp_vals)):
                t2, v2 = ramp_vals[j]
                if t2 > check_end:
                    break
                if v2 >= running_max * 0.75:
                    sustained = False
                    break
            if sustained:
                break  # running_max is the true ramp peak

    peak_power = running_max

    # Scan forward from the ramp peak for the first sustained power drop.
    # The ramp end is the last point still near peak before the drop.
    for i in range(ramp_peak_idx, len(ramp_vals)):
        t, v = ramp_vals[i]
        if v < peak_power * 0.70:
            # Verify sustained: stays below 80% of peak for 30+ seconds
            check_end = t + 30
            sustained = True
            for j in range(i, len(ramp_vals)):
                t2, v2 = ramp_vals[j]
                if t2 > check_end:
                    break
                if v2 >= peak_power * 0.80:
                    sustained = False
                    break
            if sustained:
                # Walk back to the last point that was still near peak
                for j in range(i - 1, -1, -1):
                    if ramp_vals[j][1] >= peak_power * 0.85:
                        return ramp_vals[j][0]
                return ramp_vals[max(0, i - 1)][0]

    # Fallback: ramp end at the peak if no drop detected
    return ramp_vals[ramp_peak_idx][0]


def _detect_ramp_stages(roll_t, roll_v, ramp_start, ramp_end):
    """
    Detect up to 10 staircase stages within the ramp region.
    Each stage: power stabilises at a plateau for ~3 minutes, then
    jumps >5% to the next plateau.

    Returns list of dicts: {stage_number, start_sec, end_sec, mean_power, duration_sec}
    """
    # Filter to ramp region
    ramp_data = [(t, v) for t, v in zip(roll_t, roll_v)
                 if ramp_start <= t <= ramp_end]
    if len(ramp_data) < 10:
        return []

    stages = []
    current_stage_start = ramp_start
    current_vals = []
    stage_num = 1

    i = 0
    while i < len(ramp_data) and stage_num <= 10:
        t, v = ramp_data[i]

        if not current_vals:
            current_vals.append(v)
            i += 1
            continue

        current_median = np.median(current_vals)
        elapsed_in_stage = t - current_stage_start

        # Check for step-change: power jumps >5% above current plateau
        if v > current_median * (1 + STEP_CHANGE_PCT) and elapsed_in_stage >= PLATEAU_MIN_SEC:
            # End current stage
            stages.append({
                'stage_number': stage_num,
                'start_sec': round(current_stage_start, 1),
                'end_sec': round(t, 1),
                'mean_power': round(float(current_median), 1),
                'duration_sec': round(elapsed_in_stage, 1),
            })
            stage_num += 1
            current_stage_start = t
            current_vals = [v]
        else:
            current_vals.append(v)

        i += 1

    # Capture the final stage if it has reasonable duration
    if current_vals and stage_num <= 10:
        final_t = ramp_data[-1][0]
        elapsed = final_t - current_stage_start
        if elapsed >= 90:  # At least 1.5 min for final stage
            stages.append({
                'stage_number': stage_num,
                'start_sec': round(current_stage_start, 1),
                'end_sec': round(final_t, 1),
                'mean_power': round(float(np.median(current_vals)), 1),
                'duration_sec': round(elapsed, 1),
            })

    return stages


def _detect_max_effort(roll_t, roll_v, ramp_end, ramp_peak_power,
                       raw_times=None, raw_powers=None):
    """
    Detect 3-min max effort after recovery.

    Uses rolling average to locate the high-power region, then finds
    the best 180-second window from raw power data for accurate avg.
    Returns (start_sec, end_sec) or None.
    """
    min_start = ramp_end + RECOVERY_GAP_MIN_SEC
    threshold = ramp_peak_power * MAX_EFFORT_POWER_PCT

    # --- Phase 1: find the sustained high-power region via rolling avg ---
    region_start = None
    region_end = None
    effort_start = None
    for t, v in zip(roll_t, roll_v):
        if t < min_start:
            continue
        if v >= threshold:
            if effort_start is None:
                effort_start = t
            elif t - effort_start >= MAX_EFFORT_SUSTAIN_SEC:
                region_start = effort_start
                region_end = t
                # Extend to find where rolling avg drops below threshold*0.6
                for t2, v2 in zip(roll_t, roll_v):
                    if t2 <= t:
                        continue
                    if v2 >= threshold * 0.6:
                        region_end = t2
                    else:
                        break
                break
        else:
            effort_start = None

    if region_start is None:
        return None

    # --- Phase 2: best 180s window from raw data in expanded region ---
    if raw_times is not None and raw_powers is not None:
        raw_t = np.asarray(raw_times, dtype=float)
        raw_p = np.asarray(raw_powers, dtype=float)

        # Expand search window: 60s before rolling-detected start,
        # 30s after rolling-detected end (rolling avg lags real start)
        search_lo = region_start - 60
        search_hi = region_end + 30
        mask = (raw_t >= search_lo) & (raw_t <= search_hi)
        seg_t = raw_t[mask]
        seg_p = raw_p[mask]

        if len(seg_t) >= 120:
            best_avg = 0.0
            best_start = region_start
            # Slide a 180s window across the raw data
            for i in range(len(seg_t)):
                win_end = seg_t[i] + 180
                win_mask = (seg_t >= seg_t[i]) & (seg_t < win_end)
                win_p = seg_p[win_mask]
                if len(win_p) >= 120:
                    avg = float(np.mean(win_p))
                    if avg > best_avg:
                        best_avg = avg
                        best_start = seg_t[i]
            return (float(best_start), float(best_start + 180))

    # Fallback: return the rolling-average boundaries
    return (region_start, region_end)


def _detect_cooldown(roll_t, roll_v, warmup_power, max_effort_end,
                     ramp_end):
    """
    Detect cooldown start: power drops below warmup_power × 1.3
    and stays low for >60s.
    """
    start_after = max_effort_end if max_effort_end else ramp_end + 60
    threshold = warmup_power * COOLDOWN_POWER_MULTIPLIER
    cooldown_start = None

    for t, v in zip(roll_t, roll_v):
        if t < start_after:
            continue
        if v < threshold:
            if cooldown_start is None:
                cooldown_start = t
            elif t - cooldown_start >= COOLDOWN_SUSTAIN_SEC:
                return cooldown_start
        else:
            cooldown_start = None

    return cooldown_start


def detect_segments(powers, heart_rates):
    """
    Auto-detect all ramp test segments from power time series.

    Returns dict with:
        warmup:       (start_sec, end_sec)
        ramp_start:   float
        stages:       list of stage dicts
        ramp_end:     float
        ramp_peak_power: float
        recovery:     (start_sec, end_sec) or None
        max_effort:   (start_sec, end_sec) or None
        cooldown_start: float or None
        detection_method: 'auto'
        warnings:     list of str
    """
    warnings = []

    if powers is None or len(powers) < 600:
        return {
            'status': 'error',
            'message': 'Insufficient power data for ramp test analysis. '
                       'At least 10 minutes of data required.',
            'warnings': ['File too short for ramp test protocol.'],
        }

    pw_times = [t for t, _ in powers]
    pw_vals = [p for _, p in powers]

    # Compute rolling 30s average
    roll_t, roll_v = _rolling_avg(pw_times, pw_vals, ROLLING_WINDOW_SEC)

    # Step 1: Warmup power baseline
    warmup_power = _compute_warmup_power(powers)

    # Step 2: Ramp start
    ramp_start = _detect_ramp_start(roll_t, roll_v, warmup_power)
    if ramp_start is None:
        warnings.append('Could not detect ramp start. Power may not have '
                        'risen sufficiently above warm-up level.')
        # Fallback: try with lower multiplier
        ramp_start = _detect_ramp_start(roll_t, roll_v, warmup_power * 0.9)
    if ramp_start is None:
        return {
            'status': 'error',
            'message': 'Could not detect ramp start in power data.',
            'warnings': warnings,
        }

    # Step 3: Ramp end
    ramp_end = _detect_ramp_end(roll_t, roll_v, ramp_start)
    if ramp_end is None:
        # Fallback: assume ramp ends at ~35 min after start
        ramp_end = ramp_start + 35 * 60
        if ramp_end > roll_t[-1]:
            ramp_end = roll_t[-1] - 60
        warnings.append('Ramp end auto-detection failed. '
                        'Using estimated boundary.')

    # Validate ramp duration
    ramp_duration = ramp_end - ramp_start
    if ramp_duration < 20 * 60:
        warnings.append(f'Ramp duration ({ramp_duration/60:.1f} min) is '
                        f'shorter than expected (25-35 min).')
    elif ramp_duration > 40 * 60:
        warnings.append(f'Ramp duration ({ramp_duration/60:.1f} min) is '
                        f'longer than expected (25-35 min).')

    # Ramp peak power
    ramp_vals = [v for t, v in zip(roll_t, roll_v) if ramp_start <= t <= ramp_end]
    ramp_peak_power = max(ramp_vals) if ramp_vals else warmup_power * 2

    # Step 4: Detect stages within ramp
    stages = _detect_ramp_stages(roll_t, roll_v, ramp_start, ramp_end)

    if len(stages) < 4:
        warnings.append(f'Only {len(stages)} ramp stages detected '
                        f'(expected 10). Consider manual adjustment.')

    # Validate stages
    for i in range(1, len(stages)):
        if stages[i]['mean_power'] <= stages[i-1]['mean_power']:
            warnings.append(f'Stage {stages[i]["stage_number"]} has lower '
                            f'power than stage {stages[i-1]["stage_number"]} '
                            f'(non-monotonic).')
        dur = stages[i]['duration_sec']
        if dur < PLATEAU_MIN_SEC or dur > PLATEAU_MAX_SEC:
            warnings.append(f'Stage {stages[i]["stage_number"]} duration '
                            f'({dur:.0f}s) outside expected range '
                            f'({PLATEAU_MIN_SEC}-{PLATEAU_MAX_SEC}s).')

    # Step 5: Max effort
    max_effort = _detect_max_effort(roll_t, roll_v, ramp_end, ramp_peak_power,
                                    raw_times=pw_times, raw_powers=pw_vals)
    if max_effort is None:
        warnings.append('No 3-min max effort detected after recovery.')

    # Step 6: Cooldown
    max_effort_end = max_effort[1] if max_effort else None
    cooldown_start = _detect_cooldown(roll_t, roll_v, warmup_power,
                                       max_effort_end, ramp_end)

    # Recovery segment
    recovery = None
    if max_effort:
        recovery = (ramp_end, max_effort[0])

    return {
        'status': 'ok',
        'warmup': (0.0, ramp_start),
        'ramp_start': ramp_start,
        'stages': stages,
        'ramp_end': ramp_end,
        'ramp_peak_power': round(ramp_peak_power, 1),
        'recovery': recovery,
        'max_effort': max_effort,
        'cooldown_start': cooldown_start,
        'warmup_power': round(warmup_power, 1),
        'detection_method': 'auto',
        'warnings': warnings,
    }


def build_run_segments(threshold_speed_ms, file_start_time, total_duration):
    """
    Build protocol-defined segments for run ramp test.
    No data scanning — all from protocol timing and threshold speed.
    """
    warnings = []
    warmup_speed = threshold_speed_ms * RUN_WARMUP_PCT
    warmup_start = file_start_time
    warmup_end = warmup_start + RUN_WARMUP_DURATION  # 900s

    # Build 10 ramp stages
    stages = []
    stage_start = warmup_end
    for i, intensity in enumerate(RUN_RAMP_INTENSITIES):
        stage_end = stage_start + RUN_STEP_DURATION_SEC
        stages.append({
            'stage_number': i + 1,
            'start_sec': round(stage_start, 1),
            'end_sec': round(stage_end, 1),
            'mean_power': round(threshold_speed_ms * intensity, 4),  # speed in m/s
            'duration_sec': RUN_STEP_DURATION_SEC,
        })
        stage_start = stage_end

    ramp_end = stage_start  # After 10 stages = warmup_end + 1800
    ramp_peak_speed = stages[-1]['mean_power']  # Highest stage speed

    # Recovery period
    recovery_start = ramp_end
    recovery_end = recovery_start + RUN_RECOVERY_DURATION  # +480s

    # TTE segment
    tte_start = recovery_end  # = file_start + 53 min
    tte_end_max = tte_start + RUN_TTE_MAX_DURATION  # +360s

    # Validate against file duration
    if total_duration < warmup_end:
        warnings.append(f'File duration ({total_duration:.0f}s) shorter than warmup ({RUN_WARMUP_DURATION}s).')
    if total_duration < ramp_end:
        # Trim stages that don't fit
        stages = [s for s in stages if s['start_sec'] < total_duration]
        warnings.append(f'File duration ({total_duration:.0f}s) shorter than full ramp. Only {len(stages)} stages fit.')

    tte_segment = None
    if total_duration > tte_start:
        tte_segment = (round(tte_start, 1), round(min(tte_end_max, total_duration), 1))

    cooldown_start = None
    if tte_segment:
        cooldown_start = tte_segment[1]
    elif total_duration > recovery_end:
        cooldown_start = recovery_end

    return {
        'status': 'ok',
        'warmup': (round(warmup_start, 1), round(warmup_end, 1)),
        'ramp_start': round(warmup_end, 1),
        'stages': stages,
        'ramp_end': round(ramp_end, 1),
        'ramp_peak_power': ramp_peak_speed,
        'recovery': (round(recovery_start, 1), round(recovery_end, 1)),
        'max_effort': None,  # Run uses TTE, not max effort
        'tte_segment': tte_segment,
        'cooldown_start': cooldown_start,
        'warmup_power': warmup_speed,
        'detection_method': 'protocol',
        'warnings': warnings,
    }


def detect_tte_end(heart_rates, tte_start_sec, max_duration_sec=None):
    """
    Detect TTE end from HR data.
    HR drops 10+ bpm below rolling peak, sustained for 15s.
    """
    if max_duration_sec is None:
        max_duration_sec = RUN_TTE_MAX_DURATION

    tte_end_max = tte_start_sec + max_duration_sec

    # Filter HR to TTE window
    tte_hrs = [(t, h) for t, h in heart_rates if tte_start_sec <= t <= tte_end_max]
    if len(tte_hrs) < 10:
        return {
            'tte_end_sec': tte_end_max,
            'tte_duration_sec': max_duration_sec,
            'tte_capped': True,
            'detection_method': 'insufficient_hr_data',
        }

    rolling_peak = 0
    drop_start = None

    for t, hr in tte_hrs:
        if hr > rolling_peak:
            rolling_peak = hr
            drop_start = None  # Reset drop tracking on new peak
            continue

        # Check for sustained drop
        if rolling_peak - hr >= TTE_HR_DROP_BPM:
            if drop_start is None:
                drop_start = t
            elif t - drop_start >= TTE_HR_DROP_SUSTAIN_SEC:
                # Walk back to last moment before drop
                return {
                    'tte_end_sec': round(drop_start, 1),
                    'tte_duration_sec': round(drop_start - tte_start_sec, 1),
                    'tte_capped': False,
                    'detection_method': 'hr_drop',
                }
        else:
            drop_start = None  # Reset if HR recovers

    # No drop detected — cap at max duration
    return {
        'tte_end_sec': round(tte_end_max, 1),
        'tte_duration_sec': max_duration_sec,
        'tte_capped': True,
        'detection_method': 'max_duration',
    }


def compute_dprime(tte_belt_speed, hrvt2_speed, tte_duration_sec):
    """
    Compute D-prime and derived max 3-min speed.
    D' = (tte_belt_speed - hrvt2_speed) * tte_duration_sec [metres]
    """
    if tte_belt_speed is None or hrvt2_speed is None or tte_duration_sec is None:
        return {'d_prime_metres': None, 'max_3min_speed': None, 'max_3min_pace_sec': None, 'valid': False}

    if tte_belt_speed <= hrvt2_speed:
        return {'d_prime_metres': None, 'max_3min_speed': None, 'max_3min_pace_sec': None, 'valid': False}

    d_prime = (tte_belt_speed - hrvt2_speed) * tte_duration_sec
    max_3min_speed = hrvt2_speed + (d_prime / 180.0)

    return {
        'd_prime_metres': round(d_prime, 1),
        'max_3min_speed': round(max_3min_speed, 4),
        'max_3min_pace_sec': speed_to_pace_sec(max_3min_speed),
        'valid': True,
    }


def validate_tte(tte_duration_sec, tte_capped, tte_belt_speed, hrvt2_speed,
                 dprime_result, atpr=None):
    """Validate TTE effort (replaces validate_max_effort for run)."""
    flags = []

    if tte_duration_sec is None:
        return {
            'status': 'ABSENT',
            'duration_sec': None,
            'avg_power': None,  # Compatibility field — holds max 3-min speed
            'max_power': None,
            'avg_hr': None,
            'max_hr': None,
            'pacing_ratio': None,
            'power_cv': None,
            'flags': ['TTE not performed.'],
            'd_prime_metres': None,
            'max_3min_speed': None,
            'max_3min_pace_sec': None,
            'tte_belt_speed': tte_belt_speed,
            'tte_belt_pace_sec': speed_to_pace_sec(tte_belt_speed) if tte_belt_speed else None,
            'conservative': False,
            'ceiling_limited_hint': False,
        }

    conservative = False
    ceiling_hint = False

    # Check if D' is valid
    if not dprime_result or not dprime_result.get('valid'):
        flags.append('TTE belt speed is at or below HRVT2. D-prime cannot be calculated.')
        return {
            'status': 'INVALID',
            'duration_sec': round(tte_duration_sec, 1),
            'avg_power': None,
            'max_power': None,
            'avg_hr': None,
            'max_hr': None,
            'pacing_ratio': None,
            'power_cv': None,
            'flags': flags,
            'd_prime_metres': None,
            'max_3min_speed': None,
            'max_3min_pace_sec': None,
            'tte_belt_speed': tte_belt_speed,
            'tte_belt_pace_sec': speed_to_pace_sec(tte_belt_speed) if tte_belt_speed else None,
            'conservative': False,
            'ceiling_limited_hint': False,
        }

    status = 'VALID'

    if tte_capped:
        status = 'FLAGGED'
        conservative = True
        flags.append(f'TTE capped at {RUN_TTE_MAX_DURATION}s. D-prime is a lower-bound estimate.')

    if tte_duration_sec < TTE_MIN_DURATION_CEILING and atpr is not None and atpr > DEV_LEVEL_ELITE:
        ceiling_hint = True
        flags.append(f'TTE duration ({tte_duration_sec:.0f}s) very short with high ATPR ({atpr:.2f}). Likely ceiling-limited.')

    return {
        'status': status,
        'duration_sec': round(tte_duration_sec, 1),
        'avg_power': dprime_result['max_3min_speed'],  # Compatibility with archetype scoring
        'max_power': dprime_result['max_3min_speed'],
        'avg_hr': None,
        'max_hr': None,
        'pacing_ratio': None,
        'power_cv': None,
        'flags': flags,
        'd_prime_metres': dprime_result['d_prime_metres'],
        'max_3min_speed': dprime_result['max_3min_speed'],
        'max_3min_pace_sec': dprime_result['max_3min_pace_sec'],
        'tte_belt_speed': tte_belt_speed,
        'tte_belt_pace_sec': speed_to_pace_sec(tte_belt_speed) if tte_belt_speed else None,
        'conservative': conservative,
        'ceiling_limited_hint': ceiling_hint,
    }


def detect_segments_run(speeds, heart_rates):
    """
    Auto-detect ramp test segments from speed time series (running).
    Uses the same algorithm as the bike detector but with speed instead
    of power.

    Returns dict with the same structure as detect_segments().
    The key 'mean_power' in stages holds mean speed (m/s) for compatibility.
    """
    warnings = []

    if speeds is None or len(speeds) < 600:
        return {
            'status': 'error',
            'message': 'Insufficient speed data for run ramp test analysis. '
                       'At least 10 minutes of GPS data required.',
            'warnings': ['File too short for ramp test protocol.'],
        }

    sp_times = [t for t, _ in speeds]
    sp_vals = [s for _, s in speeds]

    # Compute rolling 30s average speed
    roll_t, roll_v = _rolling_avg(sp_times, sp_vals, ROLLING_WINDOW_SEC)

    # Warmup speed baseline (min 5-15)
    warmup_vals = [s for t, s in speeds if WARMUP_ANALYSIS_START <= t <= WARMUP_ANALYSIS_END]
    if not warmup_vals:
        warmup_vals = [s for _, s in speeds[:600]]
    warmup_speed = float(np.median(warmup_vals)) if warmup_vals else 2.0

    # Ramp start: speed exceeds warmup × 1.25 sustained for 90s
    ramp_start = _detect_ramp_start(roll_t, roll_v, warmup_speed)
    if ramp_start is None:
        ramp_start = _detect_ramp_start(roll_t, roll_v, warmup_speed * 0.9)
    if ramp_start is None:
        return {
            'status': 'error',
            'message': 'Could not detect ramp start in speed data.',
            'warnings': warnings,
        }

    # Ramp end
    ramp_end = _detect_ramp_end(roll_t, roll_v, ramp_start)
    if ramp_end is None:
        ramp_end = ramp_start + 35 * 60
        if ramp_end > roll_t[-1]:
            ramp_end = roll_t[-1] - 60
        warnings.append('Ramp end auto-detection failed. Using estimated boundary.')

    ramp_duration = ramp_end - ramp_start
    if ramp_duration < 20 * 60:
        warnings.append(f'Ramp duration ({ramp_duration/60:.1f} min) shorter than expected.')
    elif ramp_duration > 40 * 60:
        warnings.append(f'Ramp duration ({ramp_duration/60:.1f} min) longer than expected.')

    ramp_vals = [v for t, v in zip(roll_t, roll_v) if ramp_start <= t <= ramp_end]
    ramp_peak_speed = max(ramp_vals) if ramp_vals else warmup_speed * 2

    # Detect stages (speed step-change detection identical to power)
    stages = _detect_ramp_stages(roll_t, roll_v, ramp_start, ramp_end)

    if len(stages) < 4:
        warnings.append(f'Only {len(stages)} ramp stages detected (expected 10).')

    # Max effort detection
    max_effort = _detect_max_effort(roll_t, roll_v, ramp_end, ramp_peak_speed)
    if max_effort is None:
        warnings.append('No 3-min max effort detected after recovery.')

    max_effort_end = max_effort[1] if max_effort else None
    cooldown_start = _detect_cooldown(roll_t, roll_v, warmup_speed,
                                       max_effort_end, ramp_end)
    recovery = (ramp_end, max_effort[0]) if max_effort else None

    return {
        'status': 'ok',
        'warmup': (0.0, ramp_start),
        'ramp_start': ramp_start,
        'stages': stages,
        'ramp_end': ramp_end,
        'ramp_peak_power': round(ramp_peak_speed, 3),
        'recovery': recovery,
        'max_effort': max_effort,
        'cooldown_start': cooldown_start,
        'warmup_power': round(warmup_speed, 3),
        'detection_method': 'auto',
        'warnings': warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: DFA ALPHA1 COMPUTATION (full file + segment slicing)
# ═══════════════════════════════════════════════════════════════════════════

def compute_full_file_dfa(file_bytes):
    """
    Parse FIT file and compute DFA a1 windows across the full file.
    Returns dict with parsed data, cleaned RR, artifact info, and windows.
    """
    parsed = parse_fit_file(file_bytes)
    rr_ms = parsed['rr_ms']
    rr_times = parsed['rr_times']

    if len(rr_ms) < 30:
        return {
            'status': 'error',
            'message': 'No heart rate data found in file.',
            'parsed': parsed,
        }

    rr_clean, times_clean, artifact_pct = clean_rr_intervals(rr_ms, rr_times)

    rr_orig = np.array(rr_ms)
    rr_fixed = np.array(rr_clean)
    artifact_mask = (np.abs(rr_orig - rr_fixed) > 0.01).tolist()

    windows = build_windows(
        rr_clean, times_clean,
        parsed['heart_rates'], parsed['powers'],
        artifact_mask=artifact_mask
    )

    return {
        'status': 'ok',
        'parsed': parsed,
        'rr_clean': rr_clean,
        'rr_times': times_clean,
        'artifact_pct': round(artifact_pct, 2),
        'artifact_mask': artifact_mask,
        'windows': windows,
    }


def slice_windows_by_segment(windows, start, end):
    """Filter windows whose midpoint falls within [start, end]."""
    return [w for w in windows if start <= w['time'] <= end]


def compute_segment_artifact_rate(rr_times, artifact_mask, start, end):
    """Compute artifact percentage for beats within a time segment."""
    t = np.array(rr_times)
    m = np.array(artifact_mask, dtype=bool)
    in_seg = (t >= start) & (t <= end)
    seg_mask = m[in_seg]
    if len(seg_mask) == 0:
        return 0.0
    return round(float(seg_mask.sum()) / len(seg_mask) * 100.0, 2)


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: RAMP STAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_stage_a1(windows, stage):
    """
    For a single stage, discard first STAGE_DISCARD_SEC (120s),
    then compute mean/SD of DFA a1 over the final STAGE_ANALYSIS_SEC (60s).
    """
    analysis_start = stage['start_sec'] + STAGE_DISCARD_SEC
    analysis_end = stage['end_sec']

    # If stage is too short for full discard, use the last 60s available
    if analysis_start >= analysis_end:
        analysis_start = max(stage['start_sec'], analysis_end - STAGE_ANALYSIS_SEC)

    stage_windows = [
        w for w in windows
        if analysis_start <= w['time'] <= analysis_end
        and w['reliable']
    ]

    if not stage_windows:
        # Try with all windows (including unreliable) as fallback
        stage_windows = [
            w for w in windows
            if analysis_start <= w['time'] <= analysis_end
        ]

    if not stage_windows:
        return {
            'a1_mean': None, 'a1_sd': None, 'a1_count': 0,
            'mean_hr': None, 'mean_power': stage['mean_power'],
            'reliable': False,
        }

    a1_vals = [w['alpha1'] for w in stage_windows]
    hr_vals = [w['hr'] for w in stage_windows if w['hr'] is not None]

    return {
        'a1_mean': round(float(np.mean(a1_vals)), 4),
        'a1_sd': round(float(np.std(a1_vals)), 4) if len(a1_vals) > 1 else 0.0,
        'a1_count': len(a1_vals),
        'mean_hr': round(float(np.mean(hr_vals)), 1) if hr_vals else None,
        'mean_power': stage['mean_power'],
        'reliable': True,
    }


def _compute_a1_max_early(windows, warmup_end):
    """
    Compute baseline DFA a1 from the final 4 minutes of the warm-up period.

    Uses the median of reliable warm-up windows rather than the maximum.
    This matches Rogers et al. methodology where a1_max represents the
    stable low-intensity baseline, not the peak outlier.  Taking the max
    inflates a1_max in noisy data, which pushes a1* higher and shifts
    HRVT1c to unrealistically high power/HR values.
    """
    lookback_start = max(0, warmup_end - WARMUP_A1_WINDOW_SEC)
    warmup_windows = [
        w for w in windows
        if lookback_start <= w['time'] <= warmup_end
        and w['reliable']
    ]

    if not warmup_windows:
        warmup_windows = [
            w for w in windows
            if lookback_start <= w['time'] <= warmup_end
        ]

    if not warmup_windows:
        return None

    a1_vals = [w['alpha1'] for w in warmup_windows]
    return round(float(np.median(a1_vals)), 4)


def _fit_regression_and_solve(stage_data, x_key, threshold):
    """
    Linear regression of a1_mean vs x_key (power or HR).
    Identify stages with near-linear decline, fit regression, solve for threshold.

    Uses iterative outlier rejection: fit initial regression, compute
    standardised residuals, exclude points > 2 SD, refit. This prevents
    late-stage rebounds (e.g. chaotic HRV at exhaustion) from skewing
    threshold estimates.

    Returns {value, slope, intercept, r2, n, extrapolated, ci_95, excluded_stages}
    """
    # Filter valid stages
    valid = [
        s for s in stage_data
        if s['a1_mean'] is not None and s[x_key] is not None
    ]

    if len(valid) < 4:
        return {'value': None, 'slope': None, 'intercept': None,
                'r2': None, 'n': 0, 'extrapolated': False, 'ci_95': None,
                'excluded_stages': []}

    # Identify the declining subset
    # Start: first stage where a1 is declining from peak
    a1_vals = [s['a1_mean'] for s in valid]
    peak_idx = int(np.argmax(a1_vals))

    # Build initial regression subset: from peak onwards, excluding stages
    # where a1 increases by >0.15 from previous (gross artifact filter)
    reg_stages = [valid[peak_idx]]
    for i in range(peak_idx + 1, len(valid)):
        if valid[i]['a1_mean'] > reg_stages[-1]['a1_mean'] + 0.15:
            continue  # Skip obvious anomalous increase
        reg_stages.append(valid[i])

    if len(reg_stages) < 4:
        return {'value': None, 'slope': None, 'intercept': None,
                'r2': None, 'n': 0, 'extrapolated': False, 'ci_95': None,
                'excluded_stages': []}

    # Iterative outlier rejection (up to 2 passes)
    excluded_stages = []
    for _pass in range(2):
        x = np.array([s[x_key] for s in reg_stages])
        y = np.array([s['a1_mean'] for s in reg_stages])

        if len(x) < 4:
            break

        slope, intercept, r_value, _, _ = stats.linregress(x, y)

        # Only reject outliers if we have enough points and slope is negative
        if slope >= 0 or len(x) < 5:
            break

        # Compute standardised residuals
        predicted = slope * x + intercept
        residuals = y - predicted
        std_res = np.std(residuals)

        if std_res < 1e-9:
            break  # Perfect fit, no outliers

        z_scores = np.abs(residuals / std_res)

        # Exclude points with |z| > 2.0 (beyond 2 standard deviations)
        keep = z_scores <= 2.0
        if keep.all():
            break  # No outliers found

        # Track which stages were excluded
        for i, kept in enumerate(keep):
            if not kept:
                excluded_stages.append(reg_stages[i].get('stage_num'))

        reg_stages = [s for s, k in zip(reg_stages, keep) if k]

    if len(reg_stages) < 4:
        return {'value': None, 'slope': None, 'intercept': None,
                'r2': None, 'n': 0, 'extrapolated': False, 'ci_95': None,
                'excluded_stages': excluded_stages}

    # Final fit on cleaned data
    x = np.array([s[x_key] for s in reg_stages])
    y = np.array([s['a1_mean'] for s in reg_stages])

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r_value ** 2

    if slope >= 0:
        # No negative relationship — threshold cannot be estimated
        return {'value': None, 'slope': round(float(slope), 6),
                'intercept': round(float(intercept), 4),
                'r2': round(float(r2), 4), 'n': len(reg_stages),
                'extrapolated': False, 'ci_95': None,
                'excluded_stages': excluded_stages}

    # Solve: threshold = slope * x_cross + intercept
    x_cross = (threshold - intercept) / slope
    extrapolated = not (x.min() <= x_cross <= x.max())

    # 95% prediction interval if extrapolated
    ci_95 = None
    if extrapolated and len(reg_stages) >= 4:
        n = len(x)
        x_mean = np.mean(x)
        ss_x = np.sum((x - x_mean) ** 2)
        residuals = y - (slope * x + intercept)
        mse = np.sum(residuals ** 2) / (n - 2)
        se_pred = np.sqrt(mse * (1 + 1/n + (x_cross - x_mean)**2 / ss_x))
        t_crit = stats.t.ppf(0.975, n - 2)
        ci_95 = (
            round(float(x_cross - t_crit * se_pred / abs(slope)), 1),
            round(float(x_cross + t_crit * se_pred / abs(slope)), 1),
        )

    return {
        'value': round(float(x_cross), 1),
        'slope': round(float(slope), 6),
        'intercept': round(float(intercept), 4),
        'r2': round(float(r2), 4),
        'n': len(reg_stages),
        'extrapolated': extrapolated,
        'ci_95': ci_95,
        'excluded_stages': excluded_stages,
    }


def analyze_ramp_stages(windows, stages, warmup_end):
    """
    Complete ramp stage analysis including per-stage DFA a1,
    regression, and threshold estimation.
    """
    # Per-stage DFA a1
    stage_data = []
    for stage in stages:
        sa1 = _compute_stage_a1(windows, stage)
        stage_data.append({
            'stage_number': stage['stage_number'],
            'start_sec': stage['start_sec'],
            'end_sec': stage['end_sec'],
            'duration_sec': stage['duration_sec'],
            'mean_power': stage['mean_power'],
            **sa1,
        })

    # Early-ramp a1 maximum (final 4 min of warm-up)
    a1_max_early = _compute_a1_max_early(windows, warmup_end)

    # Warn if a1_max_early is low
    a1_warnings = []
    if a1_max_early is not None and a1_max_early < 1.0:
        a1_warnings.append(
            f'Warm-up a1 baseline ({a1_max_early:.3f}) is below 1.0. '
            f'Athlete may be fatigued or data quality compromised.'
        )

    # HRVT1s — Standard (a1 = 0.75)
    hrvt1s_power_result = _fit_regression_and_solve(stage_data, 'mean_power', 0.75)
    hrvt1s_hr_result = _fit_regression_and_solve(stage_data, 'mean_hr', 0.75)

    # HRVT2 — a1 = 0.50 (= new FTP)
    hrvt2_power_result = _fit_regression_and_solve(stage_data, 'mean_power', 0.50)
    hrvt2_hr_result = _fit_regression_and_solve(stage_data, 'mean_hr', 0.50)

    # HRVT1c — Individualised (Rogers 2024)
    a1_star = None
    hrvt1c_power_result = {'value': None, 'extrapolated': False, 'ci_95': None}
    hrvt1c_hr_result = {'value': None, 'extrapolated': False, 'ci_95': None}
    if a1_max_early is not None:
        a1_star = round((a1_max_early + 0.50) / 2, 4)
        hrvt1c_power_result = _fit_regression_and_solve(
            stage_data, 'mean_power', a1_star)
        hrvt1c_hr_result = _fit_regression_and_solve(
            stage_data, 'mean_hr', a1_star)

    # Derived metrics
    tsr = None
    atpr = None
    if hrvt2_power_result['value'] and hrvt1c_power_result['value']:
        hrvt2_p = hrvt2_power_result['value']
        hrvt1c_p = hrvt1c_power_result['value']
        if hrvt2_p > 0:
            tsr = round((hrvt2_p - hrvt1c_p) / hrvt2_p, 4)
            atpr = round(hrvt1c_p / hrvt2_p, 4)

    return {
        'stage_data': stage_data,
        'a1_max_early': a1_max_early,
        'a1_star': a1_star,
        'a1_warnings': a1_warnings,
        'regression_power': {
            'slope': hrvt2_power_result['slope'],
            'intercept': hrvt2_power_result['intercept'],
            'r2': hrvt2_power_result['r2'],
            'n': hrvt2_power_result['n'],
        },
        'regression_hr': {
            'slope': hrvt2_hr_result['slope'],
            'intercept': hrvt2_hr_result['intercept'],
            'r2': hrvt2_hr_result['r2'],
            'n': hrvt2_hr_result['n'],
        },
        'hrvt1s_power': hrvt1s_power_result['value'],
        'hrvt1s_hr': hrvt1s_hr_result['value'],
        'hrvt1c_power': hrvt1c_power_result['value'],
        'hrvt1c_hr': hrvt1c_hr_result['value'],
        'hrvt2_power': hrvt2_power_result['value'],
        'hrvt2_hr': hrvt2_hr_result['value'],
        'hrvt2_extrapolated': hrvt2_power_result['extrapolated'],
        'hrvt2_ci_95': hrvt2_power_result['ci_95'],
        'tsr': tsr,
        'atpr': atpr,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: RAMP VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_ramp(stage_data, regression_power, regression_hr):
    """
    Validate ramp test quality.
    Returns dict with overall_status, individual checks, and flags.
    """
    flags = []
    valid_stages = [s for s in stage_data if s['a1_mean'] is not None]
    n_stages = len(valid_stages)

    # Stages completed
    if n_stages >= RAMP_STAGES_VALID:
        stages_status = 'VALID'
    elif n_stages >= RAMP_STAGES_FLAGGED:
        stages_status = 'FLAGGED'
        flags.append(f'Only {n_stages} stages completed '
                     f'(minimum {RAMP_STAGES_VALID} for full confidence).')
    else:
        stages_status = 'INVALID'
        flags.append(f'Only {n_stages} stages completed '
                     f'(minimum {RAMP_STAGES_FLAGGED} required).')

    # DFA a1 decline
    a1_vals = [s['a1_mean'] for s in valid_stages]
    if len(a1_vals) >= 2:
        slope_check = np.polyfit(range(len(a1_vals)), a1_vals, 1)[0]
        decline_valid = bool(slope_check < 0)
    else:
        decline_valid = False
    if not decline_valid:
        flags.append('No negative DFA a1 trend across stages. '
                     'Ramp may not have been progressive enough.')

    # 0.75 crossing
    min_a1 = min(a1_vals) if a1_vals else 1.0
    crossing_valid = bool(min_a1 <= 0.75)
    if not crossing_valid:
        flags.append(f'Lowest stage a1 mean ({min_a1:.3f}) is above 0.75. '
                     f'Ramp did not reach aerobic threshold.')

    # Regression R²
    r2_power = regression_power.get('r2')
    r2_hr = regression_hr.get('r2')
    best_r2 = max(r2_power or 0, r2_hr or 0)

    if best_r2 >= RAMP_R2_VALID:
        r2_status = 'VALID'
    elif best_r2 >= RAMP_R2_FLAGGED:
        r2_status = 'FLAGGED'
        flags.append(f'Regression R² ({best_r2:.3f}) is below optimal '
                     f'({RAMP_R2_VALID}). Data may have elevated noise.')
    else:
        r2_status = 'INVALID'
        flags.append(f'Regression R² ({best_r2:.3f}) is too low '
                     f'(minimum {RAMP_R2_FLAGGED} required).')

    # Overall status
    statuses = [stages_status, r2_status]
    if not decline_valid:
        statuses.append('INVALID')
    if not crossing_valid:
        statuses.append('INVALID')

    if 'INVALID' in statuses:
        overall = 'INVALID'
    elif 'FLAGGED' in statuses:
        overall = 'FLAGGED'
    else:
        overall = 'VALID'

    return {
        'stages_completed': n_stages,
        'stages_status': stages_status,
        'decline_valid': decline_valid,
        'crossing_valid': crossing_valid,
        'r2_power': r2_power,
        'r2_hr': r2_hr,
        'r2_status': r2_status,
        'overall_status': overall,
        'flags': flags,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: 3-MIN MAX EFFORT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_max_effort(powers, heart_rates, max_effort_segment,
                        hrvt2_power, stages):
    """
    Validate the 3-min max effort segment.
    Returns dict with status, metrics, and flags.
    """
    if max_effort_segment is None:
        return {
            'status': 'ABSENT',
            'duration_sec': None,
            'avg_power': None,
            'max_power': None,
            'avg_hr': None,
            'max_hr': None,
            'pacing_ratio': None,
            'power_cv': None,
            'flags': ['No max effort segment detected.'],
        }

    start, end = max_effort_segment
    duration = end - start

    # Extract effort power data
    effort_powers = [p for t, p in powers if start <= t <= end]
    effort_hrs = [h for t, h in heart_rates if start <= t <= end]

    if not effort_powers:
        return {
            'status': 'INVALID',
            'duration_sec': round(duration, 1),
            'avg_power': None, 'max_power': None,
            'avg_hr': None, 'max_hr': None,
            'pacing_ratio': None, 'power_cv': None,
            'flags': ['No power data in max effort segment.'],
        }

    avg_power = float(np.mean(effort_powers))
    max_power = float(np.max(effort_powers))
    avg_hr = float(np.mean(effort_hrs)) if effort_hrs else None
    max_hr = float(np.max(effort_hrs)) if effort_hrs else None

    flags = []

    # Duration check
    if duration < EFFORT_DUR_MIN_FLAGGED:
        flags.append(f'Effort duration ({duration:.0f}s) too short '
                     f'(minimum {EFFORT_DUR_MIN_FLAGGED}s).')
        dur_status = 'INVALID'
    elif duration < EFFORT_DUR_MIN_VALID:
        flags.append(f'Effort duration ({duration:.0f}s) slightly short '
                     f'(optimal {EFFORT_DUR_MIN_VALID}-{EFFORT_DUR_MAX_VALID}s).')
        dur_status = 'FLAGGED'
    elif duration > EFFORT_DUR_MAX_VALID:
        flags.append(f'Effort duration ({duration:.0f}s) longer than expected '
                     f'(max {EFFORT_DUR_MAX_VALID}s).')
        dur_status = 'FLAGGED'
    else:
        dur_status = 'VALID'

    # Pacing: divide into 3 equal segments
    third = len(effort_powers) // 3
    if third > 0:
        p1 = float(np.mean(effort_powers[:third]))
        p3 = float(np.mean(effort_powers[2*third:]))
        pacing_ratio = round(p1 / p3, 3) if p3 > 0 else None
    else:
        pacing_ratio = None

    if pacing_ratio is not None:
        if pacing_ratio > EFFORT_PACING_INVALID:
            flags.append(f'Severe pacing fade (P1/P3 = {pacing_ratio:.2f}). '
                         f'Power dropped significantly.')
            pace_status = 'INVALID'
        elif pacing_ratio > EFFORT_PACING_FLAGGED_HIGH:
            flags.append(f'Moderate pacing fade (P1/P3 = {pacing_ratio:.2f}).')
            pace_status = 'FLAGGED'
        elif pacing_ratio < EFFORT_PACING_FLAGGED_LOW:
            flags.append(f'Unusual negative split (P1/P3 = {pacing_ratio:.2f}).')
            pace_status = 'FLAGGED'
        else:
            pace_status = 'VALID'
    else:
        pace_status = 'VALID'

    # Power floor
    if hrvt2_power is not None:
        if avg_power < hrvt2_power:
            flags.append(f'3-min avg power ({avg_power:.0f}W) is below HRVT2 '
                         f'({hrvt2_power:.0f}W). Physiologically implausible.')
            floor_status = 'INVALID'
        elif avg_power < hrvt2_power * 1.05:
            flags.append(f'3-min avg power ({avg_power:.0f}W) is only '
                         f'{(avg_power/hrvt2_power - 1)*100:.0f}% above HRVT2.')
            floor_status = 'FLAGGED'
        else:
            floor_status = 'VALID'
    else:
        floor_status = 'VALID'

    # HR response — not used for validation (power-only)
    hr_status = 'VALID'

    # Power CV
    power_cv = None
    if len(effort_powers) > 10:
        # 5-second rolling average CV
        window = 5
        rolling = [float(np.mean(effort_powers[max(0,i-window):i+1]))
                   for i in range(len(effort_powers))]
        power_cv = round(float(np.std(rolling) / np.mean(rolling)), 4)
        if power_cv > EFFORT_POWER_CV_FLAGGED:
            flags.append(f'High power variability (CV = {power_cv:.3f}).')

    cv_status = 'FLAGGED' if (power_cv and power_cv > EFFORT_POWER_CV_FLAGGED) else 'VALID'

    # Composite status
    all_statuses = [dur_status, pace_status, floor_status, hr_status, cv_status]
    if 'INVALID' in all_statuses:
        overall = 'INVALID'
    elif 'FLAGGED' in all_statuses:
        overall = 'FLAGGED'
    else:
        overall = 'VALID'

    return {
        'status': overall,
        'duration_sec': round(duration, 1),
        'avg_power': round(avg_power, 1),
        'max_power': round(max_power, 1),
        'avg_hr': round(avg_hr, 1) if avg_hr else None,
        'max_hr': round(max_hr, 1) if max_hr else None,
        'pacing_ratio': pacing_ratio,
        'power_cv': power_cv,
        'flags': flags,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 6: METABOLIC ARCHETYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _score_afu(afu):
    """Score AFU on 0-10 scale. Higher AFU = more aerobic."""
    if afu > 0.75:
        return 10
    elif afu >= 0.70:
        return 8
    elif afu >= 0.65:
        return 6
    elif afu >= 0.60:
        return 4
    elif afu >= 0.55:
        return 2
    return 0


def _score_atpr(atpr):
    """Score ATPR on 0-10 scale. Higher ATPR = more aerobic."""
    if atpr > 0.87:
        return 10
    elif atpr >= 0.83:
        return 8
    elif atpr >= 0.78:
        return 6
    elif atpr >= 0.73:
        return 4
    elif atpr >= 0.68:
        return 2
    return 0


def _score_ar(ar):
    """Score AR on 0-10 scale. Lower AR = more aerobic (inverted)."""
    if ar < 0.08:
        return 10
    elif ar <= 0.12:
        return 8
    elif ar <= 0.16:
        return 6
    elif ar <= 0.20:
        return 4
    elif ar <= 0.25:
        return 2
    return 0


def classify_metabolic_archetype(hrvt1c_power, hrvt2_power,
                                 max_effort_power, ramp_status,
                                 effort_status):
    """
    Classify metabolic archetype using weighted composite scoring.

    Three scoring metrics (hidden scores, never shown):
      AFU  (weight 5): hrvt1c_power / max_effort_power
      ATPR (weight 3): hrvt1c_power / hrvt2_power
      AR   (weight 2): 1 - (hrvt2_power / max_effort_power), inverted

    Composite = (AFU_score×5 + ATPR_score×3 + AR_score×2) / 10
    Classification: >7.5 High Aerobic, 4.5–7.5 Balanced, <4.5 High Anaerobic

    Fallback (no valid effort): ATPR-only classification.
    """
    result = {
        'archetype': None,
        'afu': None, 'anfu': None, 'ar': None,
        'tsr': None, 'atpr': None,
        'method': 'insufficient',
        'confidence': 'low',
        'flags': [],
        'recommendation': None,
    }

    if ramp_status == 'INVALID':
        result['flags'].append('Ramp INVALID — no thresholds, no classification. '
                               'Recommend full retest.')
        result['recommendation'] = 'Full retest recommended.'
        return result

    # Always compute TSR and ATPR if we have both thresholds
    if hrvt2_power and hrvt1c_power and hrvt2_power > 0:
        result['tsr'] = round((hrvt2_power - hrvt1c_power) / hrvt2_power, 4)
        result['atpr'] = round(hrvt1c_power / hrvt2_power, 4)

    # Full method (requires valid/flagged effort + both thresholds)
    if max_effort_power and effort_status in ('VALID', 'FLAGGED') and \
       hrvt1c_power and hrvt2_power and max_effort_power > 0:

        afu = hrvt1c_power / max_effort_power
        anfu = hrvt2_power / max_effort_power
        ar = 1.0 - anfu

        result['afu'] = round(afu, 4)
        result['anfu'] = round(anfu, 4)
        result['ar'] = round(ar, 4)
        result['method'] = 'full'

        # Score each metric (hidden)
        afu_score = _score_afu(afu)
        atpr_score = _score_atpr(result['atpr'])
        ar_score = _score_ar(ar)

        composite = (afu_score * AFU_WEIGHT
                     + atpr_score * ATPR_WEIGHT
                     + ar_score * AR_WEIGHT) / 10.0

        # FLAGGED effort adjustment — ceiling likely underestimated
        if effort_status == 'FLAGGED':
            composite += FLAGGED_EFFORT_ADJUSTMENT

        # Classify
        if composite > COMPOSITE_HIGH_AEROBIC:
            result['archetype'] = 'High Aerobic'
        elif composite >= COMPOSITE_BALANCED_LOW:
            result['archetype'] = 'Balanced'
        else:
            result['archetype'] = 'High Anaerobic'

        # Confidence
        if effort_status == 'VALID' and ramp_status == 'VALID':
            result['confidence'] = 'high'
        else:
            result['confidence'] = 'medium'
            result['flags'].append('Reduced confidence — '
                                   'some validation checks flagged.')
            if effort_status == 'FLAGGED':
                result['recommendation'] = ('Recommend retesting with a '
                                            'well-paced 3-min max effort.')
            if ramp_status == 'FLAGGED':
                result['recommendation'] = 'Recommend ramp retest.'

    # Fallback: ATPR only (no valid max effort)
    elif result['atpr'] is not None:
        result['method'] = 'tsr_atpr_only'
        result['confidence'] = 'low'

        atpr = result['atpr']
        if atpr > ATPR_FALLBACK_HIGH_AEROBIC:
            result['archetype'] = 'High Aerobic'
        elif atpr >= ATPR_FALLBACK_BALANCED_LOW:
            result['archetype'] = 'Balanced'
        else:
            result['archetype'] = 'High Anaerobic'

        label_suffix = ' (Threshold Ratios Only)'
        result['archetype'] += label_suffix

        if effort_status in ('INVALID', 'ABSENT'):
            result['flags'].append(
                'Classification based on threshold ratio only. '
                'No valid max effort data.')
            result['recommendation'] = (
                'Recommend retesting with a well-paced 3-min max effort '
                'for full profiling.')
        if ramp_status == 'FLAGGED':
            result['flags'].append('Ramp flagged — reduced confidence.')
            result['recommendation'] = 'Recommend full retest.'

    return result


def classify_development_level(atpr, hrvt2_power):
    """
    Classify athlete development level based on ATPR and provide
    race pace bands as percentage of FTP with watt calculations.

    Returns dict with:
        level       - 'Beginner' / 'Intermediate' / 'Advanced' / 'Elite'
        pace_bands  - list of {distance, pct, watts} dicts
        note        - plain-language description
    """
    if atpr is None:
        return None

    if atpr > DEV_LEVEL_ELITE:
        level = 'Elite'
    elif atpr >= DEV_LEVEL_ADVANCED:
        level = 'Advanced'
    elif atpr >= DEV_LEVEL_INTERMEDIATE:
        level = 'Intermediate'
    else:
        level = 'Beginner'

    bands = RACE_PACE_BANDS[level]
    pace_bands = []
    for distance in ('Sprint', 'Olympic', '70.3', 'Ironman'):
        pct = bands[distance]
        watts = round(hrvt2_power * pct) if hrvt2_power else None
        pace_bands.append({
            'distance': distance,
            'pct': round(pct * 100),
            'watts': watts,
        })

    notes = {
        'Beginner': (
            'Your aerobic threshold sits well below your FTP. There is '
            'significant room to develop your aerobic base before focusing '
            'on threshold work.'
        ),
        'Intermediate': (
            'Your aerobic base is developing. Continued zone 2 volume will '
            'close the gap between your aerobic threshold and FTP.'
        ),
        'Advanced': (
            'Your aerobic threshold is close to your FTP. You have a '
            'well-developed aerobic engine with room for further refinement.'
        ),
        'Elite': (
            'Your aerobic threshold sits very close to your FTP. Your '
            'aerobic development is at a high level — gains from here are '
            'incremental.'
        ),
    }

    return {
        'level': level,
        'pace_bands': pace_bands,
        'note': notes[level],
    }


def detect_ceiling_limited(ar, ramp_status, effort_status):
    """
    Detect whether FTP is limited by the VO2max ceiling.

    When AR < 0.10, the athlete has converted nearly all available capacity
    into sustainable output — their FTP cannot rise without first raising
    their ceiling (VO2max / max power).

    Returns True only when ramp is VALID/FLAGGED and effort is VALID/FLAGGED.
    """
    if ar is None:
        return False
    if ramp_status == 'INVALID' or effort_status not in ('VALID', 'FLAGGED'):
        return False
    return ar < CEILING_LIMITED_AR_THRESHOLD


def classify_development_level_run(atpr, hrvt2_speed):
    """Run-specific development level with race pace bands in pace format."""
    if atpr is None:
        return None

    if atpr > DEV_LEVEL_ELITE:
        level = 'Elite'
    elif atpr >= DEV_LEVEL_ADVANCED:
        level = 'Advanced'
    elif atpr >= DEV_LEVEL_INTERMEDIATE:
        level = 'Intermediate'
    else:
        level = 'Beginner'

    bands = RUN_RACE_PACE_BANDS.get(level, {})
    pace_bands = []
    for distance in ['5K', '10K', 'Half Marathon', 'Marathon']:
        pct = bands.get(distance, 0)
        speed = round(hrvt2_speed * pct, 4) if hrvt2_speed else None
        pace_bands.append({
            'distance': distance,
            'pct': round(pct * 100),
            'speed_ms': speed,
            'pace_sec': speed_to_pace_sec(speed) if speed else None,
            'pace_formatted': format_pace(speed_to_pace_sec(speed)) if speed else '--:--',
            'watts': None,  # Not applicable for run
        })

    notes = {
        'Beginner': 'Your aerobic threshold sits well below your lactate threshold. There is significant room to develop your aerobic base before focusing on threshold work.',
        'Intermediate': 'Your aerobic base is developing. Continued easy running volume will close the gap between your aerobic threshold and lactate threshold.',
        'Advanced': 'Your aerobic threshold is close to your lactate threshold. You have a well-developed aerobic engine with room for further refinement.',
        'Elite': 'Your aerobic threshold sits very close to your lactate threshold. Your aerobic development is at a high level. Gains from here are incremental.',
    }

    return {
        'level': level,
        'pace_bands': pace_bands,
        'note': notes.get(level, ''),
    }


def detect_ceiling_limited_run(ar, ramp_status, tte_status, tte_duration_sec=None,
                                tte_capped=False, atpr=None):
    """Run-specific ceiling-limited detection with TTE duration trigger."""
    if ramp_status == 'INVALID':
        return False

    # Standard AR check (same as bike, but only when TTE valid and not capped)
    if ar is not None and ar < CEILING_LIMITED_AR_THRESHOLD:
        if tte_status == 'VALID' and not tte_capped:
            return True

    # Run-specific: very short TTE with high ATPR
    if (tte_duration_sec is not None and tte_duration_sec < TTE_MIN_DURATION_CEILING
            and atpr is not None and atpr > DEV_LEVEL_ELITE):
        return True

    return False


def generate_athlete_feedback(hrvt1c_power, hrvt2_power, max_effort_power,
                              effort_status, archetype_result, is_run=False):
    """
    Generate concise, objective feedback split into strengths and weaknesses.

    Returns dict: { 'strengths': [...], 'weaknesses': [...] }
    """
    strengths = []
    weaknesses = []
    afu = archetype_result.get('afu')
    anfu = archetype_result.get('anfu')
    ar = archetype_result.get('ar')
    tsr = archetype_result.get('tsr')
    atpr = archetype_result.get('atpr')

    has_effort = (max_effort_power is not None
                  and effort_status in ('VALID', 'FLAGGED')
                  and hrvt1c_power is not None
                  and hrvt2_power is not None)

    # Helper: format a value as pace (running) or watts (cycling)
    def _fv(val):
        if is_run:
            return format_pace(speed_to_pace_sec(val))
        return f'{val:.0f} W'

    unit_label = '/km' if is_run else ''
    ftp_label = 'CV' if is_run else 'FTP'

    # ── Max Aerobic Capacity ──
    if has_effort:
        if is_run:
            strengths.append(f'3-min max aerobic pace: {_fv(max_effort_power)}/km')
        else:
            strengths.append(f'3-min max aerobic power: {_fv(max_effort_power)}')

    # ── AFU — Aerobic Base ──
    if has_effort and afu is not None:
        afu_score = _score_afu(afu)
        pct = f'{afu * 100:.0f}'
        if afu_score >= 8:
            strengths.append(
                f'Strong aerobic base: {_fv(hrvt1c_power)} ({pct}% of {_fv(max_effort_power)} ceiling)')
        elif afu_score >= 6:
            weaknesses.append(
                f'Moderate aerobic base: {_fv(hrvt1c_power)} ({pct}% of {_fv(max_effort_power)} ceiling). Room to improve.')
        else:
            weaknesses.append(
                f'Underdeveloped aerobic base: {_fv(hrvt1c_power)} ({pct}% of {_fv(max_effort_power)} ceiling). Priority area.')

    # ── ATPR — Threshold Gap ──
    if hrvt1c_power is not None and hrvt2_power is not None and atpr is not None:
        atpr_score = _score_atpr(atpr)
        if is_run:
            h1_pace = format_pace(speed_to_pace_sec(hrvt1c_power))
            h2_pace = format_pace(speed_to_pace_sec(hrvt2_power))
            gap_sec = (speed_to_pace_sec(hrvt1c_power) or 0) - (speed_to_pace_sec(hrvt2_power) or 0)
            gap_str = f'{gap_sec:.0f}s/km'
            if atpr_score >= 8:
                strengths.append(
                    f'Narrow threshold gap: {gap_str} between HRVT1 ({h1_pace}/km) and {ftp_label} ({h2_pace}/km)')
            elif atpr_score >= 6:
                weaknesses.append(
                    f'Moderate threshold gap: {gap_str} between HRVT1 ({h1_pace}/km) and {ftp_label} ({h2_pace}/km)')
            else:
                weaknesses.append(
                    f'Wide threshold gap: {gap_str} between HRVT1 ({h1_pace}/km) and {ftp_label} ({h2_pace}/km)')
        else:
            gap = f'{hrvt2_power - hrvt1c_power:.0f}'
            if atpr_score >= 8:
                strengths.append(
                    f'Narrow threshold gap: {gap} W between HRVT1 ({hrvt1c_power:.0f} W) and {ftp_label} ({hrvt2_power:.0f} W)')
            elif atpr_score >= 6:
                weaknesses.append(
                    f'Moderate threshold gap: {gap} W between HRVT1 ({hrvt1c_power:.0f} W) and {ftp_label} ({hrvt2_power:.0f} W)')
            else:
                weaknesses.append(
                    f'Wide threshold gap: {gap} W between HRVT1 ({hrvt1c_power:.0f} W) and {ftp_label} ({hrvt2_power:.0f} W)')

    # ── AR — Reserve Above Threshold ──
    if has_effort and ar is not None:
        ar_score = _score_ar(ar)
        ceiling_limited = ar < CEILING_LIMITED_AR_THRESHOLD
        if is_run:
            h2_pace = format_pace(speed_to_pace_sec(hrvt2_power))
            reserve_sec = (speed_to_pace_sec(hrvt2_power) or 0) - (speed_to_pace_sec(max_effort_power) or 0)
            reserve_str = f'{reserve_sec:.0f}s/km'
            if ceiling_limited:
                weaknesses.append(
                    f'Ceiling-limited: only {reserve_str} faster than {ftp_label} ({h2_pace}/km). '
                    f'VO\u2082max work needed to raise the ceiling.')
            elif ar_score >= 8:
                strengths.append(
                    f'High pace conversion: {reserve_str} reserve beyond {ftp_label} ({h2_pace}/km)')
            elif ar_score >= 6:
                strengths.append(
                    f'Moderate reserve: {reserve_str} beyond {ftp_label} ({h2_pace}/km)')
            else:
                weaknesses.append(
                    f'Large untapped reserve: {reserve_str} beyond {ftp_label} ({h2_pace}/km). '
                    f'Sustainable pace has not matched capacity.')
        else:
            reserve = f'{max_effort_power - hrvt2_power:.0f}'
            h2 = f'{hrvt2_power:.0f}'
            if ceiling_limited:
                weaknesses.append(
                    f'Ceiling-limited: only {reserve} W above {ftp_label} ({h2} W). '
                    f'VO\u2082max work needed to raise the ceiling.')
            elif ar_score >= 8:
                strengths.append(
                    f'High power conversion: {reserve} W reserve above {ftp_label} ({h2} W)')
            elif ar_score >= 6:
                strengths.append(
                    f'Moderate reserve: {reserve} W above {ftp_label} ({h2} W)')
            else:
                weaknesses.append(
                    f'Large untapped reserve: {reserve} W above {ftp_label} ({h2} W). '
                    f'Sustainable power has not matched capacity.')

    # ── AnFU — Sustainable Fraction ──
    if has_effort and anfu is not None:
        pct = f'{anfu * 100:.0f}'
        if anfu > 0.85:
            strengths.append(
                f'High sustainability: {_fv(hrvt2_power)} sustained ({pct}% of {_fv(max_effort_power)})')
        elif anfu >= 0.78:
            strengths.append(
                f'Good sustainability: {_fv(hrvt2_power)} sustained ({pct}% of {_fv(max_effort_power)})')
        else:
            weaknesses.append(
                f'Low sustainability: {_fv(hrvt2_power)} sustained ({pct}% of {_fv(max_effort_power)})')

    # ── TSR — Zone 2 Width ──
    if hrvt1c_power is not None and hrvt2_power is not None and tsr is not None:
        if is_run:
            gap_sec = (speed_to_pace_sec(hrvt1c_power) or 0) - (speed_to_pace_sec(hrvt2_power) or 0)
            gap_str = f'{gap_sec:.0f}s/km'
        else:
            gap_str = f'{hrvt2_power - hrvt1c_power:.0f} W'
        if tsr < 0.18:
            strengths.append(
                f'Narrow Zone 2: {gap_str} band. Low pacing penalty.')
        elif tsr <= 0.25:
            weaknesses.append(
                f'Moderate Zone 2 width: {gap_str} band. Pacing matters.')
        else:
            weaknesses.append(
                f'Wide Zone 2: {gap_str} band. Staying near HRVT1 critical for fuel economy.')

    return {'strengths': strengths, 'weaknesses': weaknesses}


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING RECOMMENDATIONS (Part 10)
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_recommendations(hrvt1c_power, hrvt2_power, max_effort_power,
                                       effort_status, archetype_result, is_run=False):
    """
    Generate 3 dot-point training recommendations based on hidden AFU, ATPR, AR scores.
    Uses watts for cycling, pace for running. Returns list of strings.
    """
    afu = archetype_result.get('afu')
    atpr = archetype_result.get('atpr')
    ar = archetype_result.get('ar')
    ceiling_limited = archetype_result.get('ceiling_limited', False)
    has_effort = effort_status in ('VALID', 'FLAGGED') and max_effort_power

    # Format threshold values appropriately
    if is_run:
        h1 = format_pace(speed_to_pace_sec(hrvt1c_power)) + '/km' if hrvt1c_power else '?'
        h2 = format_pace(speed_to_pace_sec(hrvt2_power)) + '/km' if hrvt2_power else '?'
    else:
        h1 = f'{hrvt1c_power:.0f} W' if hrvt1c_power else '?'
        h2 = f'{hrvt2_power:.0f} W' if hrvt2_power else '?'

    intensity_word = 'pace' if is_run else 'power'
    slower_word = 'slower than' if is_run else 'below'
    between_word = 'between' if not is_run else 'from'

    # Compute hidden scores
    afu_score = _score_afu(afu) if afu is not None else None
    atpr_score = _score_atpr(atpr) if atpr is not None else None
    ar_score = _score_ar(ar) if ar is not None else None

    # Check if all three are elite (8-10)
    all_elite = (afu_score is not None and afu_score >= 8
                 and atpr_score is not None and atpr_score >= 8
                 and ar_score is not None and ar_score >= 8)

    if all_elite:
        return [
            f'Your profile is well balanced and highly developed. Focus on race-specific '
            f'preparation, sustainment at target race intensities for race-relevant durations, '
            f'and maintaining your aerobic base with consistent volume {slower_word} {h1}.'
        ]

    points = []

    # AFU advice
    if afu_score is not None:
        if afu_score <= 2:
            points.append(
                f'Your aerobic base is underdeveloped. Prioritise easy, sub-threshold '
                f'volume {slower_word} {h1} to raise it.')
        elif afu_score == 4:
            points.append(
                f'Your aerobic base is low. Consistent volume {slower_word} {h1} is the most '
                f'effective way to build it.')
        elif afu_score == 6:
            points.append(
                f'Your aerobic base is developing well. Maintain volume {slower_word} {h1} and '
                f'begin adding controlled threshold-zone exposure.')
        else:  # 8-10
            points.append(
                f'Your aerobic base is strong. Maintain it with consistent easy volume {slower_word} '
                f'{h1} and look elsewhere for gains.')

    # ATPR advice
    if atpr_score is not None:
        if atpr_score <= 2:
            points.append(
                f'Your threshold gap is very wide. Pacing on long efforts needs to be '
                f'conservative as small errors {"faster than" if is_run else "above"} {h1} carry a disproportionate fuel cost.')
        elif atpr_score == 4:
            points.append(
                f'Your threshold gap is wide. The gap narrows primarily from below as '
                f'aerobic volume pushes {h1} {"faster" if is_run else "higher"}.')
        elif atpr_score == 6:
            points.append(
                f'Your threshold gap is moderate. Controlled efforts {between_word} {h1} and '
                f'{h2} will continue narrowing it.')
        else:  # 8-10
            points.append(
                f'Your thresholds are well compressed. Race pace can sit closer to HRVT2 '
                f'with lower metabolic cost.')

    # AR advice (only if valid effort)
    if has_effort and ar_score is not None:
        if ar_score == 0:
            points.append(
                f'Your reserve above HRVT2 is very large. This converts naturally as your '
                f'base develops, no need to chase top-end {intensity_word} right now.')
        elif ar_score == 2:
            points.append(
                f'Your reserve above HRVT2 is large. Focus on raising HRVT2 through base '
                f'and threshold work rather than building more top end.')
        elif ar_score == 4:
            points.append(
                f'Your reserve above HRVT2 is moderate. Sustainable {intensity_word} has room to grow '
                f'as your base and threshold develop.')
        elif ar_score == 6:
            points.append(
                f'Your reserve above HRVT2 is modest. Threshold duration work will continue '
                f'to convert this into sustainable {intensity_word}.')
        elif ar_score >= 8:
            if ceiling_limited:
                points.append(
                    f'Your HRVT2 is near your ceiling. VO2max development is needed to '
                    f'create room for further threshold growth.')
            else:
                points.append(
                    f'Most of your capacity has been converted into sustainable output, '
                    f'with little headroom remaining.')
    elif not has_effort:
        points.append(
            'A valid 3-min max effort would allow more complete training guidance. '
            'Recommend including this in your next test.')

    return points


# ═══════════════════════════════════════════════════════════════════════════
# PART 7/8: DATA QUALITY + FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def compute_data_quality(windows, stage_data, warmup_end, ramp_start,
                         ramp_end, max_effort_segment, rr_times,
                         artifact_mask, artifact_pct, regression_power):
    """Comprehensive data quality report."""
    quality_flags = []

    # Artifact rates per segment
    art_warmup = compute_segment_artifact_rate(
        rr_times, artifact_mask, 0, warmup_end or ramp_start)
    art_ramp = compute_segment_artifact_rate(
        rr_times, artifact_mask, ramp_start, ramp_end)
    art_effort = None
    if max_effort_segment:
        art_effort = compute_segment_artifact_rate(
            rr_times, artifact_mask, max_effort_segment[0], max_effort_segment[1])

    # Ramp artifact quality label
    if art_ramp > 5:
        ramp_artifact_label = 'UNRELIABLE'
        quality_flags.append(f'Ramp artifact rate ({art_ramp:.1f}%) exceeds 5%. '
                             f'Results may be unreliable.')
    elif art_ramp > 3:
        ramp_artifact_label = 'CAUTION'
        quality_flags.append(f'Ramp artifact rate ({art_ramp:.1f}%) is elevated.')
    else:
        ramp_artifact_label = 'GOOD'

    # Warm-up a1 stability
    warmup_windows = [w for w in windows
                      if max(0, (warmup_end or ramp_start) - WARMUP_A1_WINDOW_SEC)
                      <= w['time'] <= (warmup_end or ramp_start)]
    warmup_a1_sd = None
    warmup_stable = True
    if warmup_windows:
        warmup_a1_vals = [w['alpha1'] for w in warmup_windows]
        warmup_a1_sd = round(float(np.std(warmup_a1_vals)), 4)
        if warmup_a1_sd > 0.3:
            warmup_stable = False
            quality_flags.append(f'Warm-up a1 unstable (SD = {warmup_a1_sd:.3f}).')

    # Ramp monotonicity
    a1_means = [s['a1_mean'] for s in stage_data if s['a1_mean'] is not None]
    consistent_decline = 0
    total_transitions = 0
    for i in range(1, len(a1_means)):
        total_transitions += 1
        if a1_means[i] < a1_means[i-1]:
            consistent_decline += 1

    if total_transitions > 0 and consistent_decline < total_transitions * 0.5:
        quality_flags.append(f'Poor ramp monotonicity: only {consistent_decline}/'
                             f'{total_transitions} stages show declining a1.')

    # Regression quality
    r2 = regression_power.get('r2', 0) or 0
    if r2 >= 0.85:
        reg_label = 'GOOD'
    elif r2 >= 0.70:
        reg_label = 'ACCEPTABLE'
    else:
        reg_label = 'POOR'
        quality_flags.append(f'Regression quality is poor (R² = {r2:.3f}).')

    # Overall quality
    if 'UNRELIABLE' in ramp_artifact_label or reg_label == 'POOR':
        overall = 'poor'
    elif 'CAUTION' in ramp_artifact_label or reg_label == 'ACCEPTABLE' or not warmup_stable:
        overall = 'acceptable'
    else:
        overall = 'good'

    return {
        'overall_quality': overall,
        'artifact_rate_overall': artifact_pct,
        'artifact_rate_warmup': art_warmup,
        'artifact_rate_ramp': art_ramp,
        'artifact_rate_ramp_label': ramp_artifact_label,
        'artifact_rate_effort': art_effort,
        'warmup_a1_sd': warmup_a1_sd,
        'warmup_stable': warmup_stable,
        'ramp_monotonicity': f'{consistent_decline}/{total_transitions}',
        'ramp_monotonic': consistent_decline >= total_transitions * 0.7 if total_transitions > 0 else False,
        'regression_r2': r2,
        'regression_quality_label': reg_label,
        'quality_flags': quality_flags,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_ramp_test(file_bytes, segments_override=None,
                      protocol_type='bike', threshold_pace_sec=None,
                      tte_duration_sec=None):
    """
    Complete ramp test analysis pipeline.

    Args:
        file_bytes: Raw FIT file bytes (already decompressed if .fit.gz)
        segments_override: Optional manual segment boundaries dict
        protocol_type: 'bike' or 'run'
        threshold_pace_sec: For run protocol, threshold pace in sec/km
        tte_duration_sec: For run protocol, manual TTE duration (or -1 to skip)

    Returns comprehensive analysis result.
    """
    threshold_speed = None  # Default; set in is_run branch
    # Part 2: Full-file DFA computation
    dfa_result = compute_full_file_dfa(file_bytes)
    if dfa_result['status'] != 'ok':
        return dfa_result

    parsed = dfa_result['parsed']
    windows = dfa_result['windows']
    rr_times = dfa_result['rr_times']
    artifact_mask = dfa_result['artifact_mask']
    artifact_pct = dfa_result['artifact_pct']

    is_run = protocol_type == 'run'

    if is_run:
        # For run: we need heart rate data (speed from FIT is ignored)
        if not parsed.get('heart_rates'):
            return {
                'status': 'error',
                'message': 'No heart rate data found in FIT file.',
            }

        # Build protocol-defined segments from threshold pace
        if threshold_pace_sec:
            threshold_speed = pace_sec_to_speed(threshold_pace_sec)
        else:
            return {
                'status': 'error',
                'message': 'Threshold pace is required for run ramp analysis.',
            }

        # Determine file duration from HR data
        total_dur = parsed['heart_rates'][-1][0] if parsed['heart_rates'] else 0

        # Use GPS speed list as a dummy for compatibility (won't be used for values)
        run_speeds = parsed.get('speeds', parsed.get('powers', []))
    else:
        if not parsed['powers']:
            return {
                'status': 'error',
                'message': 'No power data found. Ramp test analysis requires '
                           'a power meter.',
            }

    if not windows:
        return {
            'status': 'error',
            'message': 'DFA computation produced no valid windows.',
        }

    # Use speed data for running, power for bike
    intensity_data = run_speeds if is_run else parsed['powers']

    # Part 1: Segment detection
    if segments_override:
        segments = _apply_override(segments_override, intensity_data)
    elif is_run and threshold_pace_sec:
        segments = build_run_segments(threshold_speed, 0.0, total_dur)
        # Override window speeds with protocol values
        windows = _assign_protocol_speed_to_windows(windows, segments, threshold_speed)
        # Set intensity_data to None — not used for run
        intensity_data = run_speeds  # Keep for compatibility but not used for values
    elif is_run:
        segments = detect_segments_run(run_speeds, parsed['heart_rates'])
    else:
        segments = detect_segments(parsed['powers'], parsed['heart_rates'])

    if segments.get('status') == 'error':
        return segments

    stages = segments['stages']
    warmup_end = segments['ramp_start']
    ramp_start = segments['ramp_start']
    ramp_end = segments['ramp_end']
    max_effort_segment = segments.get('max_effort')

    # Part 3: Ramp stage analysis
    ramp_result = analyze_ramp_stages(windows, stages, warmup_end)

    # Part 4: Ramp validation
    ramp_validation = validate_ramp(
        ramp_result['stage_data'],
        ramp_result['regression_power'],
        ramp_result['regression_hr'],
    )

    # Part 5: Max effort / TTE validation
    if is_run and threshold_pace_sec:
        tte_segment = segments.get('tte_segment')
        tte_belt_speed = threshold_speed * RUN_TTE_PCT

        if tte_duration_sec == -1:
            # Skip TTE explicitly
            effort_validation = validate_tte(None, False, tte_belt_speed,
                                              ramp_result['hrvt2_power'], None)
        elif tte_segment:
            # Determine TTE duration
            if tte_duration_sec and tte_duration_sec > 0:
                tte_result = {
                    'tte_end_sec': round(tte_segment[0] + tte_duration_sec, 1),
                    'tte_duration_sec': round(tte_duration_sec, 1),
                    'tte_capped': False,
                    'detection_method': 'manual',
                }
            else:
                tte_result = detect_tte_end(parsed['heart_rates'], tte_segment[0])

            # Compute D-prime
            dprime = compute_dprime(tte_belt_speed, ramp_result['hrvt2_power'],
                                     tte_result['tte_duration_sec'])

            effort_validation = validate_tte(
                tte_result['tte_duration_sec'],
                tte_result['tte_capped'],
                tte_belt_speed,
                ramp_result['hrvt2_power'],
                dprime,
                atpr=ramp_result.get('atpr'),
            )
        else:
            effort_validation = validate_tte(None, False, tte_belt_speed,
                                              ramp_result['hrvt2_power'], None)
    else:
        effort_validation = validate_max_effort(
            intensity_data, parsed['heart_rates'],
            max_effort_segment, ramp_result['hrvt2_power'],
            ramp_result['stage_data'],
        )

    # Part 6: Metabolic archetype + athlete feedback
    archetype = classify_metabolic_archetype(
        ramp_result['hrvt1c_power'],
        ramp_result['hrvt2_power'],
        effort_validation.get('avg_power'),
        ramp_validation['overall_status'],
        effort_validation['status'],
    )
    archetype['feedback'] = generate_athlete_feedback(
        ramp_result['hrvt1c_power'],
        ramp_result['hrvt2_power'],
        effort_validation.get('avg_power'),
        effort_validation['status'],
        archetype,
        is_run=is_run,
    )

    # Part 7: Development level classification
    if is_run:
        dev_level = classify_development_level_run(
            archetype.get('atpr'),
            ramp_result['hrvt2_power'],  # This is speed in m/s for run
        )
    else:
        dev_level = classify_development_level(
            archetype.get('atpr'),
            ramp_result['hrvt2_power'],
        )
    archetype['development_level'] = dev_level

    # Part 8: Ceiling-limited flag
    if is_run and threshold_pace_sec:
        archetype['ceiling_limited'] = detect_ceiling_limited_run(
            archetype.get('ar'),
            ramp_validation['overall_status'],
            effort_validation['status'],
            tte_duration_sec=effort_validation.get('duration_sec'),
            tte_capped=effort_validation.get('conservative', False),
            atpr=archetype.get('atpr'),
        )
    else:
        archetype['ceiling_limited'] = detect_ceiling_limited(
            archetype.get('ar'),
            ramp_validation['overall_status'],
            effort_validation['status'],
        )

    # Part 10: Training recommendations
    archetype['training_recommendations'] = generate_training_recommendations(
        ramp_result['hrvt1c_power'],
        ramp_result['hrvt2_power'],
        effort_validation.get('avg_power'),
        effort_validation['status'],
        archetype,
        is_run=is_run,
    )

    # Data quality
    data_quality = compute_data_quality(
        windows, ramp_result['stage_data'],
        warmup_end, ramp_start, ramp_end,
        max_effort_segment, rr_times, artifact_mask,
        artifact_pct, ramp_result['regression_power'],
    )

    # Build compact timeline for session chart
    if is_run:
        timeline = _build_timeline_run(parsed['heart_rates'], segments)
    else:
        timeline = _build_timeline(intensity_data, parsed['heart_rates'])

    # Combine all warnings
    all_warnings = (
        parsed['warnings']
        + segments.get('warnings', [])
        + ramp_result.get('a1_warnings', [])
        + ramp_validation['flags']
        + effort_validation['flags']
        + archetype['flags']
        + data_quality['quality_flags']
    )

    return {
        'status': 'ok',
        'warnings': all_warnings,
        'source': parsed['source'],
        'segments': {
            'warmup': segments.get('warmup'),
            'ramp_start': ramp_start,
            'stages': stages,
            'ramp_end': ramp_end,
            'ramp_peak_power': segments.get('ramp_peak_power'),
            'recovery': segments.get('recovery'),
            'max_effort': max_effort_segment,
            'cooldown_start': segments.get('cooldown_start'),
            'warmup_power': segments.get('warmup_power'),
            'detection_method': segments.get('detection_method', 'auto'),
        },
        'stage_data': ramp_result['stage_data'],
        'thresholds': {
            'a1_max_early': ramp_result['a1_max_early'],
            'a1_star': ramp_result['a1_star'],
            'hrvt1s_power': ramp_result['hrvt1s_power'],
            'hrvt1s_hr': ramp_result['hrvt1s_hr'],
            'hrvt1c_power': ramp_result['hrvt1c_power'],
            'hrvt1c_hr': ramp_result['hrvt1c_hr'],
            'hrvt2_power': ramp_result['hrvt2_power'],
            'hrvt2_hr': ramp_result['hrvt2_hr'],
            'hrvt2_extrapolated': ramp_result['hrvt2_extrapolated'],
            'hrvt2_ci_95': ramp_result['hrvt2_ci_95'],
            'tsr': ramp_result['tsr'],
            'atpr': ramp_result['atpr'],
        },
        'regression_power': ramp_result['regression_power'],
        'regression_hr': ramp_result['regression_hr'],
        'ramp_validation': ramp_validation,
        'effort_validation': effort_validation,
        'archetype': archetype,
        'data_quality': data_quality,
        'windows': windows,
        'timeline': timeline,
        'artifact_pct': artifact_pct,
        'total_beats': len(dfa_result['rr_clean']),
        'duration_minutes': round(rr_times[-1] / 60, 1) if rr_times else 0,
        'protocol_type': protocol_type,
        'pace_data': {
            'threshold_speed_ms': threshold_speed,
            'threshold_pace_sec': threshold_pace_sec,
            'threshold_pace_formatted': format_pace(threshold_pace_sec),
            'hrvt1s_speed': ramp_result['hrvt1s_power'],
            'hrvt1s_pace': format_pace(speed_to_pace_sec(ramp_result['hrvt1s_power'])),
            'hrvt1c_speed': ramp_result['hrvt1c_power'],
            'hrvt1c_pace': format_pace(speed_to_pace_sec(ramp_result['hrvt1c_power'])),
            'hrvt2_speed': ramp_result['hrvt2_power'],
            'hrvt2_pace': format_pace(speed_to_pace_sec(ramp_result['hrvt2_power'])),
        } if is_run and threshold_pace_sec else None,
        'tte_data': {
            'd_prime_metres': effort_validation.get('d_prime_metres'),
            'max_3min_speed': effort_validation.get('max_3min_speed'),
            'max_3min_pace': format_pace(effort_validation.get('max_3min_pace_sec')),
            'tte_belt_speed': effort_validation.get('tte_belt_speed'),
            'tte_belt_pace': format_pace(effort_validation.get('tte_belt_pace_sec')),
            'tte_duration_sec': effort_validation.get('duration_sec'),
            'conservative': effort_validation.get('conservative', False),
        } if is_run and threshold_pace_sec else None,
    }


def _assign_protocol_speed_to_windows(windows, segments, threshold_speed):
    """
    Override each window's 'power' field with protocol-defined speed
    based on which segment it falls in.
    """
    warmup_start, warmup_end = segments.get('warmup', (0, 0))
    stages = segments.get('stages', [])
    recovery = segments.get('recovery')
    warmup_speed = segments.get('warmup_power', threshold_speed * RUN_WARMUP_PCT)
    recovery_speed = threshold_speed * RUN_RECOVERY_PCT

    for w in windows:
        t = w['time']
        assigned = False

        # Check ramp stages first (most important for regression)
        for stage in stages:
            if stage['start_sec'] <= t <= stage['end_sec']:
                w['power'] = stage['mean_power']
                assigned = True
                break

        if not assigned:
            if warmup_start <= t < warmup_end:
                w['power'] = warmup_speed
            elif recovery and recovery[0] <= t <= recovery[1]:
                w['power'] = recovery_speed
            else:
                w['power'] = None  # Outside known segments

    return windows


def _apply_override(override, powers):
    """Apply manual segment overrides to create a segments dict."""
    stages_raw = override.get('stages', [])
    stages = []
    pw_times = [t for t, _ in powers]
    pw_vals = [p for _, p in powers]

    for i, s in enumerate(stages_raw):
        stage_powers = [p for t, p in powers
                        if s['start_sec'] <= t <= s['end_sec']]
        mean_pw = float(np.median(stage_powers)) if stage_powers else 0
        stages.append({
            'stage_number': i + 1,
            'start_sec': s['start_sec'],
            'end_sec': s['end_sec'],
            'mean_power': round(mean_pw, 1),
            'duration_sec': round(s['end_sec'] - s['start_sec'], 1),
        })

    warmup = override.get('warmup', [0, stages[0]['start_sec'] if stages else 600])
    ramp_start = warmup[1] if isinstance(warmup, (list, tuple)) else warmup
    ramp_end = stages[-1]['end_sec'] if stages else ramp_start + 1800

    max_effort = None
    if override.get('max_effort'):
        me = override['max_effort']
        max_effort = (me[0], me[1]) if isinstance(me, (list, tuple)) else None

    # Compute ramp peak and warmup power
    ramp_vals = [p for t, p in powers if ramp_start <= t <= ramp_end]
    ramp_peak = max(ramp_vals) if ramp_vals else 300
    warmup_power = _compute_warmup_power(powers)

    return {
        'status': 'ok',
        'warmup': (warmup[0], warmup[1]) if isinstance(warmup, (list, tuple)) else (0, ramp_start),
        'ramp_start': ramp_start,
        'stages': stages,
        'ramp_end': ramp_end,
        'ramp_peak_power': round(ramp_peak, 1),
        'recovery': (ramp_end, max_effort[0]) if max_effort else None,
        'max_effort': max_effort,
        'cooldown_start': override.get('cooldown_start'),
        'warmup_power': round(warmup_power, 1),
        'detection_method': 'manual',
        'warnings': ['Manual segment boundaries applied.'],
    }


def _build_timeline(powers, heart_rates, sample_sec=2):
    """Build compact power + HR timeline for session overview chart."""
    if not powers:
        return []

    max_t = max(t for t, _ in powers)
    timeline = []
    for t in np.arange(0, max_t, sample_sec):
        p = _interp_value(t, [x[0] for x in powers], [x[1] for x in powers])
        h = _interp_value(t, [x[0] for x in heart_rates],
                          [x[1] for x in heart_rates]) if heart_rates else None
        timeline.append({
            't': round(float(t), 1),
            'p': round(float(p), 1) if p else None,
            'h': round(float(h), 1) if h else None,
        })
    return timeline


def _build_timeline_run(heart_rates, segments, sample_sec=2):
    """Build HR-based timeline for run session chart (no power/speed)."""
    if not heart_rates:
        return []

    max_t = max(t for t, _ in heart_rates)
    hr_times = [x[0] for x in heart_rates]
    hr_vals = [x[1] for x in heart_rates]

    timeline = []
    for t in np.arange(0, max_t, sample_sec):
        h = _interp_value(t, hr_times, hr_vals)
        timeline.append({
            't': round(float(t), 1),
            'p': None,  # No reliable speed data
            'h': round(float(h), 1) if h else None,
        })
    return timeline


# ═══════════════════════════════════════════════════════════════════════════
# HISTORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def _load_history():
    """Load the shared history file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'athletes': {}}
    return {'athletes': {}}


def _save_history(history):
    """Write the history file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def save_ramp_test_result(athlete_name, result):
    """Save a ramp test result to athlete history under 'ramp_tests' key."""
    try:
        history = _load_history()
        name = athlete_name.strip()
        if name not in history['athletes']:
            history['athletes'][name] = {}
        athlete = history['athletes'][name]
        if 'ramp_tests' not in athlete:
            athlete['ramp_tests'] = []

        record = {
            'test_date': datetime.now().isoformat(timespec='seconds'),
            'hrvt1s_power': result.get('thresholds', {}).get('hrvt1s_power'),
            'hrvt1s_hr': result.get('thresholds', {}).get('hrvt1s_hr'),
            'hrvt1c_power': result.get('thresholds', {}).get('hrvt1c_power'),
            'hrvt1c_hr': result.get('thresholds', {}).get('hrvt1c_hr'),
            'a1_star': result.get('thresholds', {}).get('a1_star'),
            'hrvt2_power': result.get('thresholds', {}).get('hrvt2_power'),
            'hrvt2_hr': result.get('thresholds', {}).get('hrvt2_hr'),
            'hrvt2_extrapolated': result.get('thresholds', {}).get('hrvt2_extrapolated'),
            'regression_r2_power': result.get('regression_power', {}).get('r2'),
            'regression_r2_hr': result.get('regression_hr', {}).get('r2'),
            'stages_completed': result.get('ramp_validation', {}).get('stages_completed'),
            'ramp_status': result.get('ramp_validation', {}).get('overall_status'),
            'effort_status': result.get('effort_validation', {}).get('status'),
            'max_effort_power': result.get('effort_validation', {}).get('avg_power'),
            'archetype': result.get('archetype', {}).get('archetype'),
            'afu': result.get('archetype', {}).get('afu'),
            'anfu': result.get('archetype', {}).get('anfu'),
            'tsr': result.get('archetype', {}).get('tsr'),
            'atpr': result.get('archetype', {}).get('atpr'),
            'artifact_pct': result.get('artifact_pct'),
            'data_quality': result.get('data_quality', {}).get('overall_quality'),
            'protocol_type': result.get('protocol_type', 'bike'),
        }

        # Save weight and computed w/kg values
        weight_kg = result.get('weight_kg')
        if weight_kg and isinstance(weight_kg, (int, float)) and weight_kg > 0:
            record['weight_kg'] = round(float(weight_kg), 1)
            w = record['weight_kg']
            if record.get('hrvt1s_power'):
                record['hrvt1s_wkg'] = round(record['hrvt1s_power'] / w, 2)
            if record.get('hrvt1c_power'):
                record['hrvt1c_wkg'] = round(record['hrvt1c_power'] / w, 2)
            if record.get('hrvt2_power'):
                record['hrvt2_wkg'] = round(record['hrvt2_power'] / w, 2)
            if record.get('max_effort_power'):
                record['max_effort_wkg'] = round(record['max_effort_power'] / w, 2)

        # Store profile fields for autofill
        if result.get('hrmax_bike'):
            record['hrmax_bike'] = result['hrmax_bike']
        if result.get('hrmax_run'):
            record['hrmax_run'] = result['hrmax_run']
        if result.get('threshold_pace'):
            record['threshold_pace'] = result['threshold_pace']

        # Save full result to individual file for later recall
        result_id = _save_full_result(result)
        if result_id:
            record['result_id'] = result_id

        athlete['ramp_tests'].append(record)
        _save_history(history)
        return True
    except Exception:
        return False


def _save_full_result(result):
    """Save full analysis result to an individual JSON file. Returns the result_id."""
    import hashlib
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Build a unique ID from timestamp + protocol type
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        sport = result.get('protocol_type', 'bike')
        result_id = f'{ts}_{sport}'
        filepath = RESULTS_DIR / f'{result_id}.json'

        # Strip FIT binary data before saving (too large)
        save_data = {k: v for k, v in result.items() if k != 'fit_file_data'}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False)
        return result_id
    except Exception:
        return None


def load_full_result(result_id):
    """Load a full analysis result by result_id. Returns dict or None."""
    try:
        filepath = RESULTS_DIR / f'{result_id}.json'
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def get_ramp_test_history(athlete_name):
    """Return all ramp tests for an athlete, sorted by date ascending."""
    history = _load_history()
    name = athlete_name.strip()
    tests = history.get('athletes', {}).get(name, {}).get('ramp_tests', [])
    return sorted(tests, key=lambda t: t.get('test_date', ''))


def get_all_ramp_tests():
    """Return every ramp test across all athletes as a flat list.

    Each record is augmented with an 'athlete' key so the frontend
    can group and filter by athlete name.
    """
    history = _load_history()
    all_tests = []
    for name, data in history.get('athletes', {}).items():
        for test in data.get('ramp_tests', []):
            row = dict(test)
            row['athlete'] = name
            all_tests.append(row)
    return sorted(all_tests, key=lambda t: t.get('test_date', ''), reverse=True)


def delete_ramp_test_from_history(athlete_name, test_index):
    """Remove a specific ramp test by index."""
    try:
        history = _load_history()
        name = athlete_name.strip()
        tests = history.get('athletes', {}).get(name, {}).get('ramp_tests', [])
        if test_index < 0 or test_index >= len(tests):
            return False
        tests.pop(test_index)
        _save_history(history)
        return True
    except Exception:
        return False
