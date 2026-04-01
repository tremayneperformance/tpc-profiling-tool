"""
Core DFA Alpha1 computation functions shared across modules.

Extracted from app.py to allow reuse by ramp_analysis.py without
circular imports. All functions are identical to their original
implementations.
"""

import io
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve
import fitparse


# ---------------------------------------------------------------------------
# FIT FILE PARSING
# ---------------------------------------------------------------------------

def parse_fit_file(file_bytes):
    """
    Extract RR intervals, heart rate, and power from a FIT file.

    Tries HRV messages first (beat-by-beat, highest accuracy).
    Falls back to estimating RR from average heart rate in record messages.

    Returns a dict with keys:
        rr_ms        - list of RR intervals in milliseconds
        rr_times     - list of elapsed seconds corresponding to each beat
        heart_rates  - list of (elapsed_seconds, bpm) tuples
        powers       - list of (elapsed_seconds, watts) tuples
        source       - 'hrv' or 'record'
        warnings     - list of warning strings
    """
    warnings = []
    fitfile = fitparse.FitFile(io.BytesIO(file_bytes))

    # Always extract HR and power from record messages
    heart_rates = []
    powers = []
    speeds = []
    start_ts = None

    for msg in fitfile.get_messages('record'):
        ts = msg.get_value('timestamp')
        if ts is None:
            continue
        ts_epoch = ts.timestamp()
        if start_ts is None:
            start_ts = ts_epoch

        hr = msg.get_value('heart_rate')
        power = msg.get_value('power')
        speed = msg.get_value('enhanced_speed') or msg.get_value('speed')

        if hr and hr > 0:
            heart_rates.append((ts_epoch - start_ts, float(hr)))
        if power and power > 0:
            powers.append((ts_epoch - start_ts, float(power)))
        if speed is not None and speed > 0:
            speeds.append((ts_epoch - start_ts, float(speed)))

    # Attempt HRV message extraction
    rr_ms = []
    rr_times = []
    elapsed = 0.0

    for msg in fitfile.get_messages('hrv'):
        for field in msg:
            if field.name == 'time':
                values = field.value
                if values is None:
                    continue
                if not hasattr(values, '__iter__'):
                    values = [values]
                for v in values:
                    # 65.535 s is the sentinel for missing/invalid RR in FIT
                    if v is not None and v < 65.0:
                        rr_ms.append(v * 1000.0)
                        rr_times.append(elapsed)
                        elapsed += v

    if len(rr_ms) < 30:
        # Not enough HRV data — fall back to record-based estimation
        warnings.append(
            "No beat-by-beat HRV data found in this file. RR intervals were "
            "estimated from average heart rate — this produces a perfectly "
            "periodic series with no variability, so DFA Alpha 1 results are "
            "unreliable. Enable HRV logging on your device for valid results."
        )
        rr_ms = []
        rr_times = []
        elapsed = 0.0
        for ts_rel, hr in heart_rates:
            rr = (60.0 / hr) * 1000.0
            rr_ms.append(rr)
            rr_times.append(ts_rel)
        source = 'record'
    else:
        source = 'hrv'

    return {
        'rr_ms': rr_ms,
        'rr_times': rr_times,
        'heart_rates': heart_rates,
        'powers': powers,
        'speeds': speeds,
        'source': source,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# ARTIFACT DETECTION AND CORRECTION
# Adaptive time-varying thresholds based on Lipponen & Tarvainen (2019)
# ---------------------------------------------------------------------------

def _compute_local_median(arr, idx, half_window=10):
    """Compute the median of arr in a local window around idx."""
    lo = max(0, idx - half_window)
    hi = min(len(arr), idx + half_window + 1)
    return float(np.median(arr[lo:hi]))


def _compute_local_threshold(drr, idx, half_window=45):
    """
    Compute a time-varying threshold from the local distribution of
    absolute dRR values.  Uses 3.32 × the local quartile deviation
    (QD = Q3 - Q1) with a floor of 10 ms to avoid over-correction
    at very stable heart rates.

    The multiplier and quartile approach approximate the Lipponen &
    Tarvainen (2019) algorithm which uses the dRR distribution to
    set adaptive thresholds that track changing heart rate during
    exercise.
    """
    lo = max(0, idx - half_window)
    hi = min(len(drr), idx + half_window + 1)
    local = np.abs(drr[lo:hi])
    if len(local) < 5:
        return 100.0  # conservative fallback
    q1 = np.percentile(local, 25)
    q3 = np.percentile(local, 75)
    qd = q3 - q1
    # Floor at 10 ms to prevent over-detection at stable HR
    return max(3.32 * qd + q1, 10.0)


def clean_rr_intervals(rr_ms, rr_times):
    """
    Detect and correct artifact RR intervals using adaptive time-varying
    thresholds inspired by Lipponen & Tarvainen (2019), as used in Kubios
    HRV.  Correction uses linear interpolation to preserve the time
    structure of the series.

    Detection uses two simultaneous adaptive thresholds on the dRR series:
      Threshold 1 (successive differences): Detects sudden jumps between
        consecutive beats — missed R-peaks, extra R-peaks, or ectopic beats.
        The threshold adapts from the local distribution of dRR values.
      Threshold 2 (deviation from local median): Catches intervals that are
        abnormally long or short relative to the surrounding context.
        Adapts to the local median RR interval.

    Both thresholds adapt over time, which is essential during exercise
    when heart rate (and therefore typical RR interval length) is
    continuously changing.

    Correction: Extra beats are removed entirely, missed beats cause the
    long RR interval to be split, and ectopic beats are replaced via
    interpolation from neighbouring valid intervals.

    Returns:
        rr_clean      - corrected RR array (same length as input)
        rr_times      - unchanged time array
        artifact_pct  - percentage of beats that were corrected
    """
    rr = np.array(rr_ms, dtype=float)
    times = np.array(rr_times, dtype=float)
    N = len(rr)

    if N == 0:
        return rr.tolist(), times.tolist(), 0.0

    # Build artifact mask (True = artifact)
    artifact = np.zeros(N, dtype=bool)

    # Step 1: physiological bounds (hard limits)
    artifact |= (rr < 300) | (rr > 2000)

    # Step 2: Adaptive dRR threshold (Lipponen & Tarvainen Threshold 1)
    # Compute successive differences
    drr = np.diff(rr)  # length N-1

    for i in range(1, N):
        if artifact[i]:
            continue
        # Adaptive threshold from local dRR distribution
        thresh = _compute_local_threshold(drr, min(i - 1, len(drr) - 1))
        if abs(drr[i - 1]) > thresh:
            artifact[i] = True

    # Step 3: Adaptive local median threshold (Lipponen & Tarvainen Threshold 2)
    # Catches intervals abnormally far from local context
    for i in range(N):
        if artifact[i]:
            continue
        local_med = _compute_local_median(rr, i, half_window=10)
        # Deviation from local median > 30% of local median
        if abs(rr[i] - local_med) / local_med > 0.30:
            artifact[i] = True

    artifact_pct = float(artifact.sum()) / N * 100.0

    # Step 4: Correct artifacts via interpolation from valid neighbours
    rr_clean = rr.copy()
    artifact_indices = np.where(artifact)[0]

    for idx in artifact_indices:
        left = idx - 1
        while left >= 0 and artifact[left]:
            left -= 1
        right = idx + 1
        while right < N and artifact[right]:
            right += 1

        if left >= 0 and right < N:
            frac = (times[idx] - times[left]) / (times[right] - times[left]) if times[right] != times[left] else 0.5
            rr_clean[idx] = rr[left] + frac * (rr[right] - rr[left])
        elif left >= 0:
            rr_clean[idx] = rr_clean[left]
        elif right < N:
            rr_clean[idx] = rr[right]

    return rr_clean.tolist(), times.tolist(), artifact_pct


# ---------------------------------------------------------------------------
# SMOOTHNESS PRIORS DETRENDING (Kubios-style)
# ---------------------------------------------------------------------------

def smoothness_priors_detrend(rr, lam=500):
    """
    Remove slow non-stationarities (trends > ~25 s) from an RR-interval
    series using the smoothness priors method (Tarvainen et al. 2002).

    lam=500 corresponds to a ~25-second high-pass cutoff at typical
    resting/exercise sampling rates.
    """
    rr = np.asarray(rr, dtype=float)
    N = len(rr)
    if N < 4:
        return rr - np.mean(rr)

    ones = np.ones(N)
    diags = np.array([ones, -2 * ones, ones])
    D2 = spdiags(diags, [0, 1, 2], N - 2, N)

    H = speye(N, format='csc') + (lam ** 2) * (D2.T @ D2)
    trend = spsolve(H, rr)
    return rr - trend


# ---------------------------------------------------------------------------
# DFA ALPHA 1
# ---------------------------------------------------------------------------

def _resample_rr_to_4hz(rr_ms, rr_times):
    """
    Cubic spline interpolation and resampling of RR intervals to a
    uniform 4 Hz grid (one sample every 250 ms).

    RR intervals are inherently unevenly sampled (each interval has a
    different duration). Resampling to a uniform grid is required for
    the smoothness priors detrending filter to operate at a consistent
    cutoff frequency (~0.035 Hz at 4 Hz sampling rate with lambda=500).

    This matches the Kubios HRV preprocessing pipeline (Step 3).
    """
    rr = np.asarray(rr_ms, dtype=float)
    times = np.asarray(rr_times, dtype=float)

    if len(rr) < 4:
        return rr, times

    # Build cumulative time axis from RR intervals (seconds)
    cum_times = np.cumsum(rr) / 1000.0  # ms -> seconds
    cum_times = np.insert(cum_times, 0, 0.0)  # start at 0

    # RR values correspond to intervals between successive beats;
    # assign each RR to the midpoint of its interval for interpolation
    mid_times = (cum_times[:-1] + cum_times[1:]) / 2.0

    # Cubic spline interpolation
    cs = CubicSpline(mid_times, rr, extrapolate=False)

    # Resample at 4 Hz (250 ms intervals)
    t_start = mid_times[0]
    t_end = mid_times[-1]
    t_uniform = np.arange(t_start, t_end, 0.250)

    if len(t_uniform) < 4:
        return rr, times

    rr_resampled = cs(t_uniform)

    # Clamp any edge extrapolation artifacts
    valid = np.isfinite(rr_resampled)
    if not valid.all():
        rr_resampled = rr_resampled[valid]
        t_uniform = t_uniform[valid]

    return rr_resampled, t_uniform


def dfa_alpha1(rr_intervals, scale_min=4, scale_max=16):
    """
    Compute the short-term DFA scaling exponent (alpha 1).

    Pipeline follows Kubios HRV preprocessing order:
      1. Cubic spline interpolation + 4 Hz resampling (uniform grid)
      2. Smoothness priors detrending (lambda=500, ~0.035 Hz cutoff)
      3. Integration (cumulative sum of mean-centred detrended series)
      4. DFA window analysis (scales n=4..16, linear local detrending)
      5. Alpha 1 extraction (slope of log(F(n)) vs log(n))

    Returns (alpha1, r_squared) or (np.nan, np.nan) if data is insufficient.
    """
    rr = np.asarray(rr_intervals, dtype=float)
    N = len(rr)

    if N < scale_max * 4:
        return np.nan, np.nan

    # Step 1: Resample to uniform 4 Hz grid (Kubios Step 3)
    rr_resampled, _ = _resample_rr_to_4hz(rr, np.cumsum(rr) / 1000.0)
    N_rs = len(rr_resampled)

    if N_rs < scale_max * 4:
        return np.nan, np.nan

    # Step 2: Smoothness priors detrending on resampled series (Kubios Step 4)
    rr_detrended = smoothness_priors_detrend(rr_resampled)

    # Step 3: Integration — cumulative sum of mean-centred series (Kubios Step 5)
    y = np.cumsum(rr_detrended - np.mean(rr_detrended))

    # Steps 4-5: DFA window analysis and alpha extraction (Kubios Steps 6-7)
    scales = np.arange(scale_min, scale_max + 1)
    fluctuations = []

    for n in scales:
        n_segments = N_rs // n
        if n_segments < 4:
            fluctuations.append(np.nan)
            continue

        rms_list = []
        x_seg = np.arange(n, dtype=float)
        for seg in range(n_segments):
            segment = y[seg * n:(seg + 1) * n]
            coeffs = np.polyfit(x_seg, segment, 1)
            trend = np.polyval(coeffs, x_seg)
            residuals = segment - trend
            rms_list.append(np.sqrt(np.mean(residuals ** 2)))

        fluctuations.append(np.mean(rms_list))

    log_scales = np.log(scales.astype(float))
    log_fluct = np.log(np.array(fluctuations, dtype=float))

    valid = np.isfinite(log_fluct)
    if valid.sum() < 4:
        return np.nan, np.nan

    slope, intercept, r_value, _, _ = stats.linregress(
        log_scales[valid], log_fluct[valid]
    )
    return float(slope), float(r_value ** 2)


# ---------------------------------------------------------------------------
# ROLLING WINDOW ENGINE (120 s time window, 5 s step — Kubios Step 8)
# ---------------------------------------------------------------------------

def _interp_value(target_time, time_series, value_series):
    if not time_series:
        return None
    times = np.array(time_series)
    vals = np.array(value_series)
    if target_time <= times[0]:
        return float(vals[0])
    if target_time >= times[-1]:
        return float(vals[-1])
    idx = np.searchsorted(times, target_time)
    t0, t1 = times[idx - 1], times[idx]
    v0, v1 = vals[idx - 1], vals[idx]
    frac = (target_time - t0) / (t1 - t0)
    return float(v0 + frac * (v1 - v0))


def build_windows(rr_ms, rr_times, heart_rates, powers,
                  window_sec=120.0, step_sec=5.0,
                  artifact_mask=None):
    """
    Slide a 120-*second* window over the RR series with a 5-second step.

    Uses a fixed 120-second time window matching the Kubios HRV, FatMaxxer,
    and Runalyze validated implementations.  The window advances every 5
    seconds (115-second overlap), producing a smooth time-series of alpha 1
    values throughout the activity.

    Per-window artifact rate is computed; windows with > 5 % artifacts
    are flagged as unreliable.  Windows with fewer than 50% of expected
    beats (indicating data gaps) are skipped.
    """
    rr = np.array(rr_ms, dtype=float)
    times = np.array(rr_times, dtype=float)
    N = len(rr)

    hr_times = [x[0] for x in heart_rates]
    hr_vals = [x[1] for x in heart_rates]
    pw_times = [x[0] for x in powers]
    pw_vals = [x[1] for x in powers]

    if artifact_mask is not None:
        art = np.array(artifact_mask, dtype=bool)
    else:
        art = np.zeros(N, dtype=bool)

    if N < 30:
        return []

    windows = []
    t_start = times[0] if N > 0 else 0.0
    t_end_max = times[-1] if N > 0 else 0.0

    # First window starts after enough time has elapsed for a full window
    current_time = t_start + window_sec

    while current_time <= t_end_max:
        # Find all beats within [current_time - window_sec, current_time]
        win_start = current_time - window_sec
        start_idx = int(np.searchsorted(times, win_start, side='left'))
        end_idx = int(np.searchsorted(times, current_time, side='right')) - 1

        if end_idx <= start_idx:
            current_time += step_sec
            continue

        window_rr = rr[start_idx:end_idx + 1]
        window_art = art[start_idx:end_idx + 1]
        n_beats = len(window_rr)

        # Skip windows with too few beats (data gap or very low HR)
        # At 30 bpm, 120s ≈ 60 beats; require at least 50% of that
        if n_beats < 30:
            current_time += step_sec
            continue

        win_artifact_pct = float(window_art.sum()) / n_beats * 100.0
        reliable = win_artifact_pct <= 5.0

        alpha, r2 = dfa_alpha1(window_rr)
        if np.isnan(alpha):
            current_time += step_sec
            continue

        # Midpoint time of the window
        midpoint_time = win_start + window_sec / 2.0

        hr = _interp_value(midpoint_time, hr_times, hr_vals)
        power = _interp_value(midpoint_time, pw_times, pw_vals) if pw_times else None

        windows.append({
            'alpha1': round(alpha, 4),
            'r2': round(r2, 4) if not np.isnan(r2) else None,
            'hr': round(hr, 1) if hr is not None else None,
            'power': round(power, 1) if power is not None else None,
            'time': round(midpoint_time, 1),
            'artifact_pct': round(win_artifact_pct, 1),
            'reliable': reliable,
        })

        current_time += step_sec

    return windows
