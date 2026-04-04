"""
FTP & Run Test Profiling Module

Handles two field tests:
1. Cycling: 5-minute + 20-minute FTP test
2. Running: 1000m + 3000m time trial

Uses short:long effort ratios to classify athletes into metabolic archetypes
(High Anaerobic, Balanced, High Aerobic) and estimate FTP / critical speed.

Evidence base:
- Cycling boundaries (1.15 / 1.06) extrapolated from Allen-Coggan correction
  factor ranges, FasCat methodology, and Kolie Moore's WKO analysis.
- Running boundaries (1.18 / 1.10) extrapolated from Duffield et al. energy
  system data and critical speed research.
- These are coaching tools, not clinical diagnoses.
"""

import io
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import fitparse


# ---------------------------------------------------------------------------
# ENUMS (string-based for JSON serialisation)
# ---------------------------------------------------------------------------

PROFILE_HIGH_ANAEROBIC = "High Anaerobic"
PROFILE_BALANCED = "Balanced"
PROFILE_HIGH_AEROBIC = "High Aerobic"


# ---------------------------------------------------------------------------
# FIT FILE PARSING — RECORDS EXTRACTION
# ---------------------------------------------------------------------------

def _parse_fit_records(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extract second-by-second records from a FIT file.

    Returns dict with:
        records   - list of dicts with keys: timestamp, elapsed, power, hr,
                    cadence, speed, distance
        sport     - detected sport string or None
        warnings  - list of warning strings
    """
    fitfile = fitparse.FitFile(io.BytesIO(file_bytes))
    warnings = []

    # Detect sport from session/sport messages
    sport = None
    for msg in fitfile.get_messages('session'):
        s = msg.get_value('sport')
        if s:
            sport = str(s).lower()
            break
    if sport is None:
        for msg in fitfile.get_messages('sport'):
            s = msg.get_value('sport')
            if s:
                sport = str(s).lower()
                break

    records = []
    start_ts = None
    prev_distance = None

    for msg in fitfile.get_messages('record'):
        ts = msg.get_value('timestamp')
        if ts is None:
            continue
        ts_epoch = ts.timestamp()
        if start_ts is None:
            start_ts = ts_epoch

        elapsed = ts_epoch - start_ts
        hr = msg.get_value('heart_rate')
        power = msg.get_value('power')
        cadence = msg.get_value('cadence')
        speed = msg.get_value('speed') or msg.get_value('enhanced_speed')
        distance = msg.get_value('distance')

        rec = {
            'timestamp': ts_epoch,
            'elapsed': elapsed,
            'power': float(power) if power and power > 0 else None,
            'hr': float(hr) if hr and hr > 0 else None,
            'cadence': float(cadence) if cadence and cadence > 0 else None,
            'speed': float(speed) if speed and speed > 0 else None,
            'distance': float(distance) if distance is not None else None,
        }
        records.append(rec)

    return {'records': records, 'sport': sport, 'warnings': warnings}


# ---------------------------------------------------------------------------
# CYCLING — BEST 5-MIN AND 20-MIN POWER DETECTION
# ---------------------------------------------------------------------------

def _find_best_power_window(records: List[dict], target_sec: float,
                            tolerance_sec: float = 10.0,
                            exclude_range: Optional[Tuple[float, float]] = None
                            ) -> Optional[Dict[str, Any]]:
    """
    Find the window with the highest average power.

    Args:
        records: list of record dicts with 'elapsed' and 'power' keys
        target_sec: target window duration in seconds
        tolerance_sec: +/- tolerance for window duration
        exclude_range: (start_elapsed, end_elapsed) to exclude (non-overlap)

    Returns dict with effort data or None if not enough data.
    """
    power_records = [r for r in records if r['power'] is not None]
    if not power_records:
        return None

    min_dur = target_sec - tolerance_sec
    max_dur = target_sec + tolerance_sec

    best_avg = 0.0
    best_window = None

    for i in range(len(power_records)):
        start_t = power_records[i]['elapsed']

        # Skip if start falls in excluded range
        if exclude_range:
            if exclude_range[0] <= start_t <= exclude_range[1]:
                continue

        # Find end of window
        powers_in_window = []
        hrs_in_window = []
        cadences_in_window = []
        end_t = start_t

        for j in range(i, len(power_records)):
            t = power_records[j]['elapsed']
            dur = t - start_t

            if dur > max_dur:
                break

            # Skip if this record falls in excluded range
            if exclude_range and exclude_range[0] <= t <= exclude_range[1]:
                continue

            powers_in_window.append(power_records[j]['power'])
            if power_records[j]['hr'] is not None:
                hrs_in_window.append(power_records[j]['hr'])
            if power_records[j]['cadence'] is not None:
                cadences_in_window.append(power_records[j]['cadence'])
            end_t = t

        actual_dur = end_t - start_t
        if actual_dur < min_dur or len(powers_in_window) < 10:
            continue

        avg_power = np.mean(powers_in_window)

        if avg_power > best_avg:
            best_avg = avg_power

            # Normalised power (30s rolling average, then RMS)
            np_val = None
            if len(powers_in_window) >= 30:
                pw_arr = np.array(powers_in_window)
                rolling = np.convolve(pw_arr, np.ones(30) / 30, mode='valid')
                np_val = round(float((np.mean(rolling ** 4)) ** 0.25), 1)

            # Power CV
            power_cv = round(float(np.std(powers_in_window) / avg_power * 100), 1) if avg_power > 0 else None

            # HR drift for 20-min efforts
            hr_drift_pct = None
            if len(hrs_in_window) >= 20 and actual_dur >= 600:
                quarter = len(hrs_in_window) // 4
                first_quarter_hr = np.mean(hrs_in_window[:quarter])
                last_quarter_hr = np.mean(hrs_in_window[-quarter:])
                if first_quarter_hr > 0:
                    hr_drift_pct = round(float((last_quarter_hr - first_quarter_hr) / first_quarter_hr * 100), 1)

            # Pacing split: first quarter vs last quarter power
            quarter_pw = len(powers_in_window) // 4
            first_q_power = float(np.mean(powers_in_window[:quarter_pw])) if quarter_pw > 0 else None
            last_q_power = float(np.mean(powers_in_window[-quarter_pw:])) if quarter_pw > 0 else None

            # Power trend (for decay detection in 5-min)
            power_trend = None
            if len(powers_in_window) >= 20:
                x = np.arange(len(powers_in_window))
                try:
                    coeffs = np.polyfit(x, powers_in_window, 1)
                    power_trend = float(coeffs[0])  # watts per second
                except Exception:
                    power_trend = None

            best_window = {
                'avg_power': round(float(avg_power), 1),
                'avg_hr': round(float(np.mean(hrs_in_window)), 1) if hrs_in_window else None,
                'max_hr': round(float(max(hrs_in_window)), 0) if hrs_in_window else None,
                'avg_cadence': round(float(np.mean(cadences_in_window)), 0) if cadences_in_window else None,
                'normalised_power': np_val,
                'power_cv': power_cv,
                'hr_drift_pct': hr_drift_pct,
                'duration_seconds': round(actual_dur, 1),
                'start_elapsed': round(start_t, 1),
                'end_elapsed': round(end_t, 1),
                'power_series': [round(p, 1) for p in powers_in_window],
                'first_quarter_power': round(first_q_power, 1) if first_q_power else None,
                'last_quarter_power': round(last_q_power, 1) if last_q_power else None,
                'last_2min_avg_power': None,
                'power_trend': power_trend,
            }

            # Last 2 minutes average power
            if actual_dur >= 120:
                last_2min = [r['power'] for r in power_records
                             if r['power'] is not None
                             and r['elapsed'] >= end_t - 120
                             and r['elapsed'] <= end_t]
                if last_2min:
                    best_window['last_2min_avg_power'] = round(float(np.mean(last_2min)), 1)

    return best_window


def detect_cycling_efforts(records: List[dict]) -> Dict[str, Any]:
    """
    Detect best 5-min and 20-min power efforts from FIT file records.
    The windows must NOT overlap.
    """
    # Find best 20-min first
    twenty_min = _find_best_power_window(records, 1200.0, tolerance_sec=10.0)

    # Find best 5-min that doesn't overlap with 20-min
    exclude = None
    if twenty_min:
        exclude = (twenty_min['start_elapsed'], twenty_min['end_elapsed'])

    five_min = _find_best_power_window(records, 300.0, tolerance_sec=10.0,
                                       exclude_range=exclude)

    return {
        'five_min': five_min,
        'twenty_min': twenty_min,
        'sport': 'cycling',
    }


# ---------------------------------------------------------------------------
# RUNNING — FASTEST 1000M AND 3000M DETECTION
# ---------------------------------------------------------------------------

def _find_fastest_distance_window(records: List[dict], target_m: float,
                                   tolerance_m: float = 10.0,
                                   exclude_range: Optional[Tuple[float, float]] = None
                                   ) -> Optional[Dict[str, Any]]:
    """
    Find the segment with the highest average speed over a rolling distance window.
    """
    dist_records = [r for r in records
                    if r['distance'] is not None and r['speed'] is not None]
    if len(dist_records) < 2:
        return None

    min_dist = target_m - tolerance_m
    max_dist = target_m + tolerance_m

    best_speed = 0.0
    best_window = None

    for i in range(len(dist_records)):
        start_d = dist_records[i]['distance']
        start_t = dist_records[i]['elapsed']

        if exclude_range and exclude_range[0] <= start_t <= exclude_range[1]:
            continue

        hrs_in_window = []
        cadences_in_window = []
        speeds_in_window = []

        for j in range(i, len(dist_records)):
            r = dist_records[j]
            seg_dist = r['distance'] - start_d

            if seg_dist > max_dist:
                break

            if exclude_range and exclude_range[0] <= r['elapsed'] <= exclude_range[1]:
                continue

            if r['speed'] is not None:
                speeds_in_window.append(r['speed'])
            if r['hr'] is not None:
                hrs_in_window.append(r['hr'])
            if r['cadence'] is not None:
                cadences_in_window.append(r['cadence'])

            if seg_dist >= min_dist:
                actual_dist = seg_dist
                actual_dur = r['elapsed'] - start_t

                if actual_dur <= 0 or len(speeds_in_window) < 5:
                    continue

                avg_speed = actual_dist / actual_dur  # m/s

                if avg_speed > best_speed:
                    best_speed = avg_speed

                    pace_min_km = (1000.0 / avg_speed) / 60.0 if avg_speed > 0 else None
                    pace_str = None
                    if pace_min_km:
                        mins = int(pace_min_km)
                        secs = int((pace_min_km - mins) * 60)
                        pace_str = f"{mins}:{secs:02d}"

                    best_window = {
                        'avg_speed': round(avg_speed, 3),
                        'pace_min_km': round(pace_min_km, 2) if pace_min_km else None,
                        'pace_str': pace_str,
                        'avg_hr': round(float(np.mean(hrs_in_window)), 1) if hrs_in_window else None,
                        'max_hr': round(float(max(hrs_in_window)), 0) if hrs_in_window else None,
                        'avg_cadence': round(float(np.mean(cadences_in_window)), 0) if cadences_in_window else None,
                        'duration_seconds': round(actual_dur, 1),
                        'distance_metres': round(actual_dist, 1),
                        'start_elapsed': round(start_t, 1),
                        'end_elapsed': round(r['elapsed'], 1),
                        'speed_series': [round(s, 3) for s in speeds_in_window],
                    }
                break  # Found a valid segment at this start point

    return best_window


def detect_running_efforts(records: List[dict]) -> Dict[str, Any]:
    """
    Detect fastest 1000m and 3000m segments from FIT file records.
    The windows must NOT overlap.
    """
    three_km = _find_fastest_distance_window(records, 3000.0, tolerance_m=10.0)

    exclude = None
    if three_km:
        exclude = (three_km['start_elapsed'], three_km['end_elapsed'])

    one_km = _find_fastest_distance_window(records, 1000.0, tolerance_m=10.0,
                                            exclude_range=exclude)

    return {
        'one_km': one_km,
        'three_km': three_km,
        'sport': 'running',
    }


# ---------------------------------------------------------------------------
# SPORT AUTO-DETECTION
# ---------------------------------------------------------------------------

def detect_sport(parsed: Dict[str, Any]) -> str:
    """
    Auto-detect whether a FIT file is cycling or running.
    Uses the sport field first, then falls back to checking for power data.
    """
    sport = parsed.get('sport', '')
    if sport:
        sport_lower = sport.lower()
        if 'cycling' in sport_lower or 'bike' in sport_lower or 'biking' in sport_lower:
            return 'cycling'
        if 'running' in sport_lower or 'run' in sport_lower:
            return 'running'

    # Fallback: if power data exists, assume cycling
    has_power = any(r['power'] is not None for r in parsed['records'])
    return 'cycling' if has_power else 'running'


# ---------------------------------------------------------------------------
# FULL FIT FILE ANALYSIS (auto-detect + extract efforts)
# ---------------------------------------------------------------------------

def analyze_fit_file(file_bytes: bytes) -> Dict[str, Any]:
    """
    Parse a FIT file, auto-detect sport, and find best efforts.

    Returns dict with detected sport, efforts, and any warnings.
    """
    parsed = _parse_fit_records(file_bytes)
    warnings = parsed['warnings']

    if not parsed['records']:
        return {'status': 'error', 'message': 'No records found in FIT file.'}

    sport = detect_sport(parsed)

    if sport == 'cycling':
        efforts = detect_cycling_efforts(parsed['records'])
    else:
        efforts = detect_running_efforts(parsed['records'])

    result = {
        'status': 'ok',
        'sport': sport,
        'warnings': warnings,
    }
    result.update(efforts)

    # Format timestamps for display
    if sport == 'cycling':
        if efforts.get('five_min'):
            e = efforts['five_min']
            m, s = divmod(int(e['start_elapsed']), 60)
            e['start_time_str'] = f"{m}:{s:02d}"
        if efforts.get('twenty_min'):
            e = efforts['twenty_min']
            m, s = divmod(int(e['start_elapsed']), 60)
            e['start_time_str'] = f"{m}:{s:02d}"
    else:
        if efforts.get('one_km'):
            e = efforts['one_km']
            m, s = divmod(int(e['start_elapsed']), 60)
            e['start_time_str'] = f"{m}:{s:02d}"
        if efforts.get('three_km'):
            e = efforts['three_km']
            m, s = divmod(int(e['start_elapsed']), 60)
            e['start_time_str'] = f"{m}:{s:02d}"

    # Build lightweight timeline for charting (compact keys to reduce JSON size)
    timeline = []
    for r in parsed['records']:
        point = {'t': round(r['elapsed'], 1)}
        if r['power'] is not None:
            point['p'] = round(r['power'], 1)
        if r['speed'] is not None:
            point['s'] = round(r['speed'], 3)
        if r['hr'] is not None:
            point['h'] = round(r['hr'], 1)
        timeline.append(point)
    result['timeline'] = timeline

    return result


# ---------------------------------------------------------------------------
# CYCLING PROFILE CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_cycling_profile(ratio: float) -> Tuple[str, float]:
    """
    Classify cycling metabolic profile from 5-min:20-min power ratio.

    Returns (profile_label, correction_factor).
    """
    if ratio > 1.15:
        # HIGH ANAEROBIC: 0.93 at 1.15, down to 0.90 at 1.30+
        cf = max(0.90, 0.93 - ((ratio - 1.15) / 0.15) * 0.03)
        return PROFILE_HIGH_ANAEROBIC, round(cf, 3)
    elif ratio >= 1.06:
        # BALANCED: 0.95 at 1.06, down to 0.93 at 1.15
        cf = 0.95 - ((ratio - 1.06) / 0.09) * 0.02
        return PROFILE_BALANCED, round(cf, 3)
    else:
        # HIGH AEROBIC: 0.97 at 0.98, down to 0.95 at 1.06
        cf = min(0.97, 0.95 + ((1.06 - ratio) / 0.08) * 0.02)
        return PROFILE_HIGH_AEROBIC, round(cf, 3)


# ---------------------------------------------------------------------------
# RUNNING PROFILE CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_running_profile(ratio: float) -> str:
    """Classify running metabolic profile from 1000m:3000m speed ratio."""
    if ratio > 1.18:
        return PROFILE_HIGH_ANAEROBIC
    elif ratio >= 1.12:
        return PROFILE_BALANCED
    else:
        return PROFILE_HIGH_AEROBIC


# ---------------------------------------------------------------------------
# CYCLING PROFILE CALCULATION
# ---------------------------------------------------------------------------

def calculate_cycling_profile(five_min_power: float, twenty_min_power: float,
                               body_weight_kg: float,
                               five_min_hr: Optional[float] = None,
                               five_min_max_hr: Optional[float] = None,
                               five_min_cadence: Optional[float] = None,
                               twenty_min_hr: Optional[float] = None,
                               twenty_min_max_hr: Optional[float] = None,
                               twenty_min_cadence: Optional[float] = None,
                               known_hrmax: Optional[float] = None,
                               hr_drift_pct: Optional[float] = None,
                               power_cv: Optional[float] = None,
                               power_trend_5min: Optional[float] = None,
                               first_quarter_power_20min: Optional[float] = None,
                               last_quarter_power_20min: Optional[float] = None,
                               last_2min_avg_power_20min: Optional[float] = None,
                               ) -> Dict[str, Any]:
    """
    Calculate full cycling profile from 5-min and 20-min power data.
    """
    ratio = round(five_min_power / twenty_min_power, 3) if twenty_min_power > 0 else 0.0
    profile, correction_factor = classify_cycling_profile(ratio)
    estimated_ftp = round(twenty_min_power * correction_factor, 1)
    ftp_wkg = round(estimated_ftp / body_weight_kg, 2) if body_weight_kg > 0 else 0.0
    five_min_wkg = round(five_min_power / body_weight_kg, 2) if body_weight_kg > 0 else 0.0
    anaerobic_excess = round(five_min_power - twenty_min_power, 1)

    # HR analysis
    hr_analysis = _cycling_hr_analysis(
        five_min_hr, five_min_max_hr,
        twenty_min_hr, twenty_min_max_hr,
        known_hrmax, hr_drift_pct,
    )

    # Validity checks
    flags = _cycling_validity_checks(
        ratio, power_trend_5min, power_cv, hr_drift_pct,
        first_quarter_power_20min, last_quarter_power_20min,
        last_2min_avg_power_20min, twenty_min_power,
        five_min_max_hr, known_hrmax,
    )

    return {
        'sport': 'cycling',
        'five_min_power': five_min_power,
        'twenty_min_power': twenty_min_power,
        'five_min_hr': five_min_hr,
        'five_min_max_hr': five_min_max_hr,
        'five_min_cadence': five_min_cadence,
        'twenty_min_hr': twenty_min_hr,
        'twenty_min_max_hr': twenty_min_max_hr,
        'twenty_min_cadence': twenty_min_cadence,
        'ratio': ratio,
        'profile': profile,
        'correction_factor': correction_factor,
        'estimated_ftp': estimated_ftp,
        'ftp_wkg': ftp_wkg,
        'five_min_wkg': five_min_wkg,
        'anaerobic_excess_watts': anaerobic_excess,
        'body_weight_kg': body_weight_kg,
        'hr_analysis': hr_analysis,
        'validity_flags': flags,
    }


# ---------------------------------------------------------------------------
# RUNNING PROFILE CALCULATION
# ---------------------------------------------------------------------------

def calculate_running_profile(time_1000_seconds: float, time_3000_seconds: float,
                               body_weight_kg: float,
                               one_km_hr: Optional[float] = None,
                               one_km_max_hr: Optional[float] = None,
                               three_km_hr: Optional[float] = None,
                               three_km_max_hr: Optional[float] = None,
                               known_hrmax: Optional[float] = None,
                               ) -> Dict[str, Any]:
    """
    Calculate full running profile from 1000m and 3000m times.
    """
    speed_1000 = 1000.0 / time_1000_seconds if time_1000_seconds > 0 else 0.0
    speed_3000 = 3000.0 / time_3000_seconds if time_3000_seconds > 0 else 0.0
    ratio = round(speed_1000 / speed_3000, 3) if speed_3000 > 0 else 0.0
    profile = classify_running_profile(ratio)

    # Critical speed ~ 3000m speed (approx 99% MAS)
    estimated_cs = round(speed_3000, 3)

    # Threshold pace as min/km
    if estimated_cs > 0:
        pace_sec_per_km = 1000.0 / estimated_cs
        pace_min = int(pace_sec_per_km // 60)
        pace_sec = int(pace_sec_per_km % 60)
        threshold_pace_str = f"{pace_min}:{pace_sec:02d}"
    else:
        threshold_pace_str = "--:--"

    # D' estimate (rough, from 2 points)
    d_prime = None
    if speed_1000 > speed_3000:
        # d_prime ≈ distance_above_cs × (1 / (1 - cs/speed))
        # Simplified: d_prime = (speed_1000 - speed_3000) * time_1000_seconds
        d_prime = round((speed_1000 - speed_3000) * time_1000_seconds, 1)

    # Format times for display
    def fmt_time(sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}:{s:02d}"

    # Pace strings
    pace_1000 = 1000.0 / speed_1000 / 60.0 if speed_1000 > 0 else None
    pace_3000 = 1000.0 / speed_3000 / 60.0 if speed_3000 > 0 else None

    def fmt_pace(p):
        if p is None:
            return "--:--"
        m = int(p)
        s = int((p - m) * 60)
        return f"{m}:{s:02d}"

    # HR analysis
    hr_analysis = _running_hr_analysis(
        one_km_hr, one_km_max_hr,
        three_km_hr, three_km_max_hr,
        known_hrmax,
    )

    # Validity checks
    flags = _running_validity_checks(
        ratio, one_km_max_hr, three_km_hr, known_hrmax,
    )

    return {
        'sport': 'running',
        'time_1000': time_1000_seconds,
        'time_3000': time_3000_seconds,
        'time_1000_str': fmt_time(time_1000_seconds),
        'time_3000_str': fmt_time(time_3000_seconds),
        'speed_1000': round(speed_1000, 3),
        'speed_3000': round(speed_3000, 3),
        'pace_1000_str': fmt_pace(pace_1000),
        'pace_3000_str': fmt_pace(pace_3000),
        'ratio': ratio,
        'profile': profile,
        'estimated_critical_speed': estimated_cs,
        'estimated_threshold_pace': threshold_pace_str,
        'd_prime_estimate': d_prime,
        'body_weight_kg': body_weight_kg,
        'one_km_hr': one_km_hr,
        'one_km_max_hr': one_km_max_hr,
        'three_km_hr': three_km_hr,
        'three_km_max_hr': three_km_max_hr,
        'hr_analysis': hr_analysis,
        'validity_flags': flags,
    }


# ---------------------------------------------------------------------------
# HR ANALYSIS — CYCLING
# ---------------------------------------------------------------------------

def _cycling_hr_analysis(five_min_hr, five_min_max_hr,
                          twenty_min_hr, twenty_min_max_hr,
                          known_hrmax, hr_drift_pct) -> Dict[str, Any]:
    """Generate HR insights for cycling test."""
    analysis = {'insights': []}

    # Determine effective HRmax
    hrmax = known_hrmax
    if hrmax is None:
        candidates = [h for h in [five_min_max_hr, twenty_min_max_hr] if h]
        if candidates:
            hrmax = max(candidates)
            analysis['hrmax_source'] = 'observed'
            analysis['insights'].append(
                f"No known HRmax provided. Using observed max of {hrmax:.0f} bpm "
                "as proxy (may not be true max)."
            )
        else:
            analysis['hrmax_source'] = 'none'
            return analysis
    else:
        analysis['hrmax_source'] = 'known'

    analysis['hrmax_used'] = hrmax

    # 5-min HR analysis
    if five_min_max_hr:
        pct = round(five_min_max_hr / hrmax * 100, 1)
        analysis['five_min_max_hr_pct'] = pct
        diff = hrmax - five_min_max_hr
        if diff <= 5:
            assessment = "Near-maximal effort confirmed"
            analysis['insights'].append(
                f"5-min max HR reached {pct}% of HRmax ({five_min_max_hr:.0f}/{hrmax:.0f} bpm) "
                f"— confirms maximal effort."
            )
        elif diff > 15:
            assessment = "Effort may not have been maximal"
            analysis['insights'].append(
                f"5-min max HR was only {pct}% of HRmax ({five_min_max_hr:.0f}/{hrmax:.0f} bpm) "
                f"— effort may not have been maximal."
            )
        else:
            assessment = "Reasonable effort level"
            analysis['insights'].append(
                f"5-min max HR reached {pct}% of HRmax ({five_min_max_hr:.0f}/{hrmax:.0f} bpm)."
            )
        analysis['five_min_effort_assessment'] = assessment

    # 5-min avg HR
    if five_min_hr:
        analysis['five_min_avg_hr_pct'] = round(five_min_hr / hrmax * 100, 1)

    # 20-min HR analysis
    if twenty_min_hr:
        pct = round(twenty_min_hr / hrmax * 100, 1)
        analysis['twenty_min_avg_hr_pct'] = pct
        analysis['estimated_threshold_hr'] = round(twenty_min_hr, 0)
        analysis['insights'].append(
            f"20-min average HR of {twenty_min_hr:.0f} bpm represents an estimated "
            "threshold HR. Compare against DFA alpha1 HRVT2 for cross-validation."
        )

    if twenty_min_max_hr:
        analysis['twenty_min_max_hr_pct'] = round(twenty_min_max_hr / hrmax * 100, 1)

    # HR drift
    if hr_drift_pct is not None:
        analysis['hr_drift_pct'] = hr_drift_pct
        if abs(hr_drift_pct) < 5:
            drift_assessment = "Well-paced steady state"
        elif abs(hr_drift_pct) < 10:
            drift_assessment = "Moderate drift, acceptable"
        else:
            drift_assessment = "Significant decoupling, effort may have been above threshold"
        analysis['hr_drift_assessment'] = drift_assessment
        analysis['insights'].append(
            f"20-min HR drift was {hr_drift_pct:.1f}%. {drift_assessment}."
        )

    return analysis


# ---------------------------------------------------------------------------
# HR ANALYSIS — RUNNING
# ---------------------------------------------------------------------------

def _running_hr_analysis(one_km_hr, one_km_max_hr,
                          three_km_hr, three_km_max_hr,
                          known_hrmax) -> Dict[str, Any]:
    """Generate HR insights for running test."""
    analysis = {'insights': []}

    hrmax = known_hrmax
    if hrmax is None:
        candidates = [h for h in [one_km_max_hr, three_km_max_hr] if h]
        if candidates:
            hrmax = max(candidates)
            analysis['hrmax_source'] = 'observed'
            analysis['insights'].append(
                f"No known HRmax provided. Using observed max of {hrmax:.0f} bpm "
                "as proxy (may not be true max)."
            )
        else:
            analysis['hrmax_source'] = 'none'
            return analysis
    else:
        analysis['hrmax_source'] = 'known'

    analysis['hrmax_used'] = hrmax

    # 1000m HR analysis
    if one_km_max_hr:
        pct = round(one_km_max_hr / hrmax * 100, 1)
        analysis['one_km_max_hr_pct'] = pct
        diff = hrmax - one_km_max_hr
        if diff <= 5:
            assessment = "Maximal effort confirmed"
            analysis['insights'].append(
                f"1000m max HR was {one_km_max_hr:.0f} bpm ({pct}% of HRmax) "
                "— maximal effort confirmed."
            )
        elif diff > 15:
            assessment = "Effort may not have been maximal"
            analysis['insights'].append(
                f"1000m max HR was only {one_km_max_hr:.0f} bpm ({pct}% of HRmax) "
                "— effort may not have been maximal, which could invalidate the profile."
            )
        else:
            assessment = "Reasonable effort level"
            analysis['insights'].append(
                f"1000m max HR reached {pct}% of HRmax ({one_km_max_hr:.0f}/{hrmax:.0f} bpm)."
            )
        analysis['one_km_effort_assessment'] = assessment

    if one_km_hr:
        analysis['one_km_avg_hr_pct'] = round(one_km_hr / hrmax * 100, 1)

    # 3000m HR analysis
    if three_km_hr:
        pct = round(three_km_hr / hrmax * 100, 1)
        analysis['three_km_avg_hr_pct'] = pct
        analysis['estimated_vo2max_hr'] = round(three_km_hr, 0)
        analysis['insights'].append(
            f"3000m average HR of {three_km_hr:.0f} bpm at {pct}% of HRmax "
            "represents an estimated HR at VO2max/MAS."
        )

        # Typical range check
        if pct < 92:
            analysis['insights'].append(
                f"3000m avg HR ({pct}%) is below the typical 92-98% of HRmax range "
                "for a well-executed 3000m effort. Effort may have been submaximal."
            )

    if three_km_max_hr:
        analysis['three_km_max_hr_pct'] = round(three_km_max_hr / hrmax * 100, 1)

    return analysis


# ---------------------------------------------------------------------------
# VALIDITY CHECKS — CYCLING
# ---------------------------------------------------------------------------

def _cycling_validity_checks(ratio, power_trend, power_cv, hr_drift_pct,
                              first_q_power, last_q_power,
                              last_2min_power, twenty_min_power,
                              five_min_max_hr, known_hrmax) -> List[Dict[str, Any]]:
    """Run validity checks for cycling FTP test."""
    flags = []

    # Insufficient 5-min effort
    if ratio < 1.05:
        flags.append({
            'check': 'insufficient_5min',
            'severity': 'warning',
            'message': "5-min effort may not have been maximal. Profile classification has reduced confidence.",
            'value': ratio,
        })

    # Suspiciously low 5-min effort
    if ratio < 1.02:
        flags.append({
            'check': 'submaximal_5min',
            'severity': 'error',
            'message': "5-min effort appears submaximal. Do not use for profiling. FTP estimate may still be usable with standard 0.95 correction.",
            'value': ratio,
        })

    # No power decay in 5-min (flat or positive trend)
    if power_trend is not None and power_trend >= 0:
        flags.append({
            'check': 'no_power_decay_5min',
            'severity': 'error',
            'message': "A true all-out 5-min effort must show power decay. This effort was likely paced, not maximal.",
            'value': power_trend,
        })

    # Severe positive split (20-min)
    if first_q_power and last_q_power and twenty_min_power and twenty_min_power > 0:
        split_diff_pct = (first_q_power - last_q_power) / twenty_min_power * 100
        if split_diff_pct > 15:
            flags.append({
                'check': 'severe_positive_split',
                'severity': 'error',
                'message': "Athlete went out too hard and faded. 20-min average is not representative. Retest required.",
                'value': round(split_diff_pct, 1),
            })
        elif split_diff_pct > 8:
            flags.append({
                'check': 'moderate_positive_split',
                'severity': 'warning',
                'message': "Pacing was imperfect. Consider applying an additional 1-2% FTP reduction.",
                'value': round(split_diff_pct, 1),
            })

    # Excessive end surge (20-min)
    if last_2min_power and twenty_min_power and twenty_min_power > 0:
        surge_pct = (last_2min_power - twenty_min_power) / twenty_min_power * 100
        if surge_pct > 20:
            flags.append({
                'check': 'end_surge',
                'severity': 'warning',
                'message': "Athlete had significant energy remaining. FTP may be underestimated.",
                'value': round(surge_pct, 1),
            })

    # High power variability
    if power_cv is not None and power_cv > 12:
        flags.append({
            'check': 'high_variability',
            'severity': 'warning',
            'message': "Power output was erratic. Consider indoor retest.",
            'value': power_cv,
        })

    # HR drift
    if hr_drift_pct is not None and abs(hr_drift_pct) > 10:
        flags.append({
            'check': 'hr_drift',
            'severity': 'warning',
            'message': "Significant cardiac drift. Effort may have been above true threshold.",
            'value': hr_drift_pct,
        })

    # 5-min HR too low
    if five_min_max_hr and known_hrmax:
        pct = five_min_max_hr / known_hrmax * 100
        if pct < 90:
            flags.append({
                'check': '5min_hr_low',
                'severity': 'warning',
                'message': f"HR did not approach max during the 5-min effort ({pct:.0f}% of HRmax). Effort may not have been maximal.",
                'value': round(pct, 1),
            })

    return flags


# ---------------------------------------------------------------------------
# VALIDITY CHECKS — RUNNING
# ---------------------------------------------------------------------------

def _running_validity_checks(ratio, one_km_max_hr, three_km_hr,
                              known_hrmax) -> List[Dict[str, Any]]:
    """Run validity checks for running test."""
    flags = []

    # Insufficient 1000m effort
    if ratio < 1.05:
        flags.append({
            'check': 'insufficient_1km',
            'severity': 'warning',
            'message': "1000m effort may not have been maximal. Profile classification has reduced confidence.",
            'value': ratio,
        })

    # Suspiciously low 1000m effort
    if ratio < 1.02:
        flags.append({
            'check': 'submaximal_1km',
            'severity': 'error',
            'message': "1000m effort appears submaximal. Do not use for profiling.",
            'value': ratio,
        })

    # 1000m HR too low
    if one_km_max_hr and known_hrmax:
        pct = one_km_max_hr / known_hrmax * 100
        if pct < 90:
            flags.append({
                'check': '1km_hr_low',
                'severity': 'warning',
                'message': f"HR did not approach max during 1000m ({pct:.0f}% of HRmax). Effort may not have been maximal.",
                'value': round(pct, 1),
            })

    # 3000m HR too low
    if three_km_hr and known_hrmax:
        pct = three_km_hr / known_hrmax * 100
        if pct < 85:
            flags.append({
                'check': '3km_hr_low',
                'severity': 'warning',
                'message': f"3000m average HR unusually low ({pct:.0f}% of HRmax). Effort may have been submaximal.",
                'value': round(pct, 1),
            })

    return flags


# ---------------------------------------------------------------------------
# HISTORY — SAVE / LOAD FTP & RUN TEST RESULTS
# ---------------------------------------------------------------------------

def _get_history_path() -> Path:
    data_dir = Path.home() / '.dfatool'
    data_dir.mkdir(exist_ok=True)
    return data_dir / 'history.json'


def _load_history() -> dict:
    path = _get_history_path()
    if not path.exists():
        return {'athletes': {}}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'athletes' not in data:
            data = {'athletes': {}}
        return data
    except Exception:
        return {'athletes': {}}


def save_ftp_test_result(athlete_name: str, result: Dict[str, Any]) -> bool:
    """Save an FTP or run test result to athlete history."""
    history = _load_history()
    name = athlete_name.strip() or 'Unknown Athlete'
    if name not in history['athletes']:
        history['athletes'][name] = {'tests': [], 'ftp_tests': []}

    # Ensure ftp_tests key exists for older history files
    if 'ftp_tests' not in history['athletes'][name]:
        history['athletes'][name]['ftp_tests'] = []

    record = {
        'test_date': datetime.now().isoformat(timespec='seconds'),
        'sport': result.get('sport'),
        'profile': result.get('profile'),
        'ratio': result.get('ratio'),
    }

    if result.get('sport') == 'cycling':
        record.update({
            'five_min_power': result.get('five_min_power'),
            'twenty_min_power': result.get('twenty_min_power'),
            'estimated_ftp': result.get('estimated_ftp'),
            'correction_factor': result.get('correction_factor'),
            'ftp_wkg': result.get('ftp_wkg'),
            'body_weight_kg': result.get('body_weight_kg'),
        })
    else:
        record.update({
            'time_1000': result.get('time_1000'),
            'time_3000': result.get('time_3000'),
            'estimated_critical_speed': result.get('estimated_critical_speed'),
            'estimated_threshold_pace': result.get('estimated_threshold_pace'),
            'body_weight_kg': result.get('body_weight_kg'),
        })

    history['athletes'][name]['ftp_tests'].append(record)

    try:
        with open(_get_history_path(), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def get_previous_ftp_test(athlete_name: str, sport: str) -> Optional[dict]:
    """Return the most recent FTP test for an athlete/sport."""
    history = _load_history()
    name = athlete_name.strip()
    athlete = history.get('athletes', {}).get(name, {})
    ftp_tests = athlete.get('ftp_tests', [])
    sport_tests = [t for t in ftp_tests if t.get('sport') == sport]
    if sport_tests:
        return sorted(sport_tests, key=lambda t: t.get('test_date', ''))[-1]
    return None


def get_ftp_test_history(athlete_name: str,
                         sport: Optional[str] = None) -> List[dict]:
    """Return all FTP tests for an athlete, optionally filtered by sport, sorted by date."""
    history = _load_history()
    name = athlete_name.strip()
    tests = history.get('athletes', {}).get(name, {}).get('ftp_tests', [])
    if sport:
        tests = [t for t in tests if t.get('sport') == sport]
    return sorted(tests, key=lambda t: t.get('test_date', ''))


def update_ftp_test_in_history(athlete_name: str, test_index: int,
                                updates: dict) -> bool:
    """Patch fields of a specific FTP test by index. Returns True on success."""
    history = _load_history()
    name = athlete_name.strip()
    tests = history.get('athletes', {}).get(name, {}).get('ftp_tests', [])
    if test_index < 0 or test_index >= len(tests):
        return False
    tests[test_index].update(updates)
    try:
        with open(_get_history_path(), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def delete_ftp_test_from_history(athlete_name: str, test_index: int) -> bool:
    """Remove a specific FTP test by index. Returns True on success."""
    history = _load_history()
    name = athlete_name.strip()
    tests = history.get('athletes', {}).get(name, {}).get('ftp_tests', [])
    if test_index < 0 or test_index >= len(tests):
        return False
    tests.pop(test_index)
    try:
        with open(_get_history_path(), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def get_ftp_population_averages(sport: Optional[str] = None) -> dict:
    """
    Compute average normalised FTP/run profile across all athletes.
    Uses each athlete's most recent test only.
    """
    history = _load_history()
    ratios = []
    ftp_wkgs = []
    correction_factors = []
    critical_speeds = []

    for name, data in history.get('athletes', {}).items():
        tests = data.get('ftp_tests', [])
        if sport:
            tests = [t for t in tests if t.get('sport') == sport]
        if not tests:
            continue
        latest = sorted(tests, key=lambda t: t.get('test_date', ''))[-1]
        if latest.get('ratio') is not None:
            ratios.append(latest['ratio'])
        if latest.get('ftp_wkg') is not None:
            ftp_wkgs.append(latest['ftp_wkg'])
        if latest.get('correction_factor') is not None:
            correction_factors.append(latest['correction_factor'])
        if latest.get('estimated_critical_speed') is not None:
            critical_speeds.append(latest['estimated_critical_speed'])

    def _avg(vals, dp=3):
        return round(sum(vals) / len(vals), dp) if vals else None

    return {
        'avg_ratio': _avg(ratios),
        'avg_ftp_wkg': _avg(ftp_wkgs, 2),
        'avg_correction_factor': _avg(correction_factors),
        'avg_critical_speed': _avg(critical_speeds),
        'athlete_count': len(ratios),
    }
