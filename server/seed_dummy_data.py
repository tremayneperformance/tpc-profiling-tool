#!/usr/bin/env python3
"""
Seed Script — Create dummy athlete accounts with realistic test data.

Creates 3 dummy athletes for UI testing:
  - Alex Demo: bike (5 days ago) + run (3 days ago) — auto-combine pair
  - Jordan Demo: bike only (14 days ago)
  - Sam Demo: bike (30 days ago) + bike (2 days ago) — progression

All accounts use password "tpc" and will prompt for password change on login.

Usage:
    python server/seed_dummy_data.py
"""

import sys
import os
import json
import hashlib
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Add server directory to path so we can import server modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HISTORY_DIR = Path.home() / '.dfatool'
HISTORY_FILE = HISTORY_DIR / 'history.json'
RESULTS_DIR = HISTORY_DIR / 'results'


def _load_history():
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {'athletes': {}}


def _save_history(history):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _save_full_result(result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_id = str(uuid.uuid4())
    filepath = RESULTS_DIR / f'{result_id}.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, default=str)
    return result_id


def generate_bike_result(ftp, hrmax, weight, test_date, name):
    """Generate a realistic bike ramp test result."""
    rng = random.Random(name)  # deterministic per athlete name
    intensities = [0.60, 0.66, 0.71, 0.77, 0.82, 0.88, 0.93, 0.99, 1.04, 1.10]
    stage_duration = 180  # 3 min

    stage_data = []
    windows = []
    timeline = []

    base_hr = int(hrmax * 0.55)
    dfa_start = 1.10

    for i, pct in enumerate(intensities):
        target = round(ftp * pct)
        avg_power = target + round((rng.randint(0, 10 - 1)) - 5)
        hr = int(base_hr + (hrmax - base_hr) * (pct - 0.50) / 0.65)
        hr = min(hr, hrmax)
        dfa = max(0.30, dfa_start - (i * 0.085) + (rng.randint(0, 10 - 1)) / 100)

        stage_start = 1200 + i * stage_duration
        stage_end = stage_start + stage_duration

        stage_data.append({
            'stage_number': i + 1,
            'start_sec': stage_start,
            'end_sec': stage_end,
            'mean_power': avg_power,
            'mean_hr': hr,
            'duration_sec': stage_duration,
            'analysis_start': stage_start + 60,
            'analysis_end': stage_end,
            'power_target': target,
            'power_avg': avg_power,
            'hr_avg': hr,
            'dfa_alpha1': round(dfa, 3),
            'rmssd': round(max(5, 45 - i * 4.2), 1),
            'sdnn': round(max(4, 38 - i * 3.5), 1),
            'rr_count': 180 + (rng.randint(0, 20 - 1)),
            'artifact_pct': round(1.5 + (rng.randint(0, 30 - 1)) / 10, 1),
        })

        # Generate windows (every 30s within stage)
        for w in range(6):
            t = stage_start + w * 30
            w_dfa = dfa + (rng.randint(0, 20 - 1) - 10) / 100
            windows.append({
                'time': t,
                'center_time': t,
                'power': avg_power + (rng.randint(0, 20 - 1) - 10),
                'hr': hr + (rng.randint(0, 6 - 1) - 3),
                'dfa_alpha1': round(max(0.20, w_dfa), 3),
                'dfa_a1': round(max(0.20, w_dfa), 3),
                'rr_count': 30 + (rng.randint(0, 5 - 1)),
            })

        # Timeline entries (every 5s)
        for s in range(0, stage_duration, 5):
            t = stage_start + s
            timeline.append({
                'time': t,
                'power': avg_power + (rng.randint(0, 30 - 1) - 15),
                'hr': hr + (rng.randint(0, 8 - 1) - 4),
            })

    # Add warmup to timeline
    warmup_power = round(ftp * 0.45)
    warmup_timeline = []
    for t in range(0, 1200, 5):
        warmup_timeline.append({
            'time': float(t),
            'power': warmup_power + (rng.randint(0, 10 - 1) - 5),
            'hr': 105 + round(t / 150),
        })
    timeline = warmup_timeline + timeline

    # Calculate thresholds
    hrvt1s_power = hrvt1s_hr = hrvt1c_power = hrvt1c_hr = None
    hrvt2_power = hrvt2_hr = None

    for i in range(1, len(stage_data)):
        prev = stage_data[i - 1]
        curr = stage_data[i]
        if prev['dfa_alpha1'] >= 0.75 > curr['dfa_alpha1'] and hrvt1s_power is None:
            frac = (0.75 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt1s_power = round(curr['power_avg'] - frac * (curr['power_avg'] - prev['power_avg']))
            hrvt1s_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))
        if prev['dfa_alpha1'] >= 0.50 > curr['dfa_alpha1'] and hrvt2_power is None:
            frac = (0.50 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt2_power = round(curr['power_avg'] - frac * (curr['power_avg'] - prev['power_avg']))
            hrvt2_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))

    if hrvt1s_power is None:
        hrvt1s_power = round(ftp * 0.72)
        hrvt1s_hr = int(hrmax * 0.72)
    if hrvt2_power is None:
        hrvt2_power = round(ftp * 0.92)
        hrvt2_hr = int(hrmax * 0.87)

    # Individualised threshold
    a1_max = max(s['dfa_alpha1'] for s in stage_data)
    a1_star = round((a1_max + 0.50) / 2, 3)
    hrvt1c_power = round(hrvt1s_power * 1.02)
    hrvt1c_hr = (hrvt1s_hr + 2) if hrvt1s_hr else None

    map_power = round(ftp * 1.20)
    reg_r2 = round(0.92 + (rng.randint(0, 7 - 1)) / 100, 3)

    segments = {
        'warmup_power': warmup_power,
        'ramp_start': 1200,
        'ramp_end': 1200 + len(intensities) * stage_duration,
        'ramp_peak_power': stage_data[-1]['mean_power'],
        'stages': stage_data,
        'recovery_start': 1200 + len(intensities) * stage_duration,
        'recovery_end': 1200 + len(intensities) * stage_duration + 480,
    }

    thresholds = {
        'hrvt1s_power': hrvt1s_power,
        'hrvt1s_hr': hrvt1s_hr,
        'hrvt1c_power': hrvt1c_power,
        'hrvt1c_hr': hrvt1c_hr,
        'hrvt2_power': hrvt2_power,
        'hrvt2_hr': hrvt2_hr,
        'hrvt2_extrapolated': False,
        'a1_star': a1_star,
    }

    result = {
        'status': 'ok',
        'protocol_type': 'bike',
        'source': 'hrv',
        'test_date': test_date.isoformat(timespec='seconds'),
        'athlete_name': name,
        'duration_sec': timeline[-1]['time'] if timeline else 3600,
        'artifact_pct': round(sum(s['artifact_pct'] for s in stage_data) / len(stage_data), 1),
        'rr_count': sum(s['rr_count'] for s in stage_data),
        'weight_kg': weight,
        'hrmax_bike': hrmax,
        'stage_data': stage_data,
        'windows': windows,
        'timeline': timeline,
        'segments': segments,
        'thresholds': thresholds,
        'regression_power': {'slope': -0.004, 'intercept': 1.8, 'r2': reg_r2},
        'regression_hr': {'slope': -0.008, 'intercept': 2.1, 'r2': round(reg_r2 - 0.02, 3)},
        'ramp_validation': {
            'stages_completed': len(stage_data),
            'overall_status': 'VALID',
            'r2_status': 'VALID',
        },
        'effort_validation': {
            'status': 'VALID',
            'avg_power': map_power,
            'duration_sec': 180,
        },
        'archetype': {
            'archetype': 'Balanced',
            'afu': 0.65,
            'anfu': 0.72,
            'tsr': 0.78,
            'atpr': 0.80,
        },
        'data_quality': {
            'overall_quality': 'good',
            'artifact_status': 'VALID',
        },
        'warnings': [],
    }
    return result


def generate_run_result(threshold_pace_min, hrmax, weight, test_date, name):
    """Generate a realistic run ramp test result."""
    rng = random.Random(name)  # deterministic per athlete name
    intensities = [0.70, 0.74, 0.78, 0.82, 0.86, 0.91, 0.95, 0.99, 1.03, 1.07]
    stage_duration = 180

    stage_data = []
    windows = []
    timeline = []
    base_hr = int(hrmax * 0.58)
    dfa_start = 1.05

    for i, pct in enumerate(intensities):
        pace = threshold_pace_min / pct
        speed = 1000 / (pace * 60)  # m/s
        hr = int(base_hr + (hrmax - base_hr) * (pct - 0.60) / 0.55)
        hr = min(hr, hrmax)
        dfa = max(0.28, dfa_start - (i * 0.080) + (rng.randint(0, 10 - 1)) / 100)

        stage_start = 900 + i * stage_duration
        stage_end = stage_start + stage_duration

        stage_data.append({
            'stage_number': i + 1,
            'start_sec': stage_start,
            'end_sec': stage_end,
            'mean_speed': round(speed, 3),
            'mean_pace_sec': round(pace * 60, 1),
            'mean_hr': hr,
            'mean_power': None,
            'duration_sec': stage_duration,
            'analysis_start': stage_start + 60,
            'analysis_end': stage_end,
            'hr_avg': hr,
            'dfa_alpha1': round(dfa, 3),
            'rmssd': round(max(4, 42 - i * 3.8), 1),
            'sdnn': round(max(3, 35 - i * 3.2), 1),
            'rr_count': 175 + (rng.randint(0, 25 - 1)),
            'artifact_pct': round(2.0 + (rng.randint(0, 35 - 1)) / 10, 1),
        })

        for w in range(6):
            t = stage_start + w * 30
            w_dfa = dfa + (rng.randint(0, 20 - 1) - 10) / 100
            windows.append({
                'time': t,
                'center_time': t,
                'speed': round(speed, 3),
                'hr': hr + (rng.randint(0, 6 - 1) - 3),
                'dfa_alpha1': round(max(0.20, w_dfa), 3),
                'dfa_a1': round(max(0.20, w_dfa), 3),
                'rr_count': 30 + (rng.randint(0, 5 - 1)),
            })

        for s in range(0, stage_duration, 5):
            t = stage_start + s
            timeline.append({
                'time': t,
                'speed': round(speed + (rng.randint(0, 10 - 1) - 5) / 100, 3),
                'hr': hr + (rng.randint(0, 8 - 1) - 4),
            })

    # Thresholds
    hrvt1s_hr = hrvt1c_hr = hrvt2_hr = None
    hrvt1s_pace = hrvt1c_pace = hrvt2_pace = None

    for i in range(1, len(stage_data)):
        prev = stage_data[i - 1]
        curr = stage_data[i]
        if prev['dfa_alpha1'] >= 0.75 > curr['dfa_alpha1'] and hrvt1s_pace is None:
            frac = (0.75 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt1s_pace = round(curr['mean_pace_sec'] + frac * (prev['mean_pace_sec'] - curr['mean_pace_sec']), 1)
            hrvt1s_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))
        if prev['dfa_alpha1'] >= 0.50 > curr['dfa_alpha1'] and hrvt2_pace is None:
            frac = (0.50 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt2_pace = round(curr['mean_pace_sec'] + frac * (prev['mean_pace_sec'] - curr['mean_pace_sec']), 1)
            hrvt2_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))

    if hrvt1s_pace is None:
        hrvt1s_pace = round(threshold_pace_min * 60 / 0.78, 1)
        hrvt1s_hr = int(hrmax * 0.74)
    if hrvt2_pace is None:
        hrvt2_pace = round(threshold_pace_min * 60 / 0.95, 1)
        hrvt2_hr = int(hrmax * 0.88)

    hrvt1c_pace = round(hrvt1s_pace * 0.98, 1)
    hrvt1c_hr = (hrvt1s_hr + 2) if hrvt1s_hr else None

    def fmt_pace(pace_sec):
        mins = int(pace_sec // 60)
        secs = int(round(pace_sec % 60))
        if secs == 60:
            mins += 1
            secs = 0
        return f'{mins}:{secs:02d}'

    a1_max = max(s['dfa_alpha1'] for s in stage_data)
    a1_star = round((a1_max + 0.50) / 2, 3)
    reg_r2 = round(0.90 + (rng.randint(0, 8 - 1)) / 100, 3)

    segments = {
        'warmup_power': None,
        'ramp_start': 900,
        'ramp_end': 900 + len(intensities) * stage_duration,
        'stages': stage_data,
    }

    thresholds = {
        'hrvt1s_power': None,
        'hrvt1s_hr': hrvt1s_hr,
        'hrvt1c_power': None,
        'hrvt1c_hr': hrvt1c_hr,
        'hrvt2_power': None,
        'hrvt2_hr': hrvt2_hr,
        'hrvt2_extrapolated': False,
        'a1_star': a1_star,
    }

    result = {
        'status': 'ok',
        'protocol_type': 'run',
        'source': 'hrv',
        'test_date': test_date.isoformat(timespec='seconds'),
        'athlete_name': name,
        'duration_sec': timeline[-1]['time'] if timeline else 2700,
        'artifact_pct': round(sum(s['artifact_pct'] for s in stage_data) / len(stage_data), 1),
        'rr_count': sum(s['rr_count'] for s in stage_data),
        'weight_kg': weight,
        'hrmax_run': hrmax,
        'threshold_pace': fmt_pace(threshold_pace_min * 60),
        'stage_data': stage_data,
        'windows': windows,
        'timeline': timeline,
        'segments': segments,
        'thresholds': thresholds,
        'pace_data': {
            'hrvt1s_pace': fmt_pace(hrvt1s_pace),
            'hrvt1c_pace': fmt_pace(hrvt1c_pace),
            'hrvt2_pace': fmt_pace(hrvt2_pace),
        },
        'regression_power': {'slope': -0.005, 'intercept': 2.0, 'r2': reg_r2},
        'regression_hr': {'slope': -0.009, 'intercept': 2.3, 'r2': round(reg_r2 - 0.03, 3)},
        'ramp_validation': {
            'stages_completed': len(stage_data),
            'overall_status': 'VALID',
            'r2_status': 'VALID',
        },
        'effort_validation': {'status': 'ABSENT'},
        'archetype': {
            'archetype': 'High Aerobic',
            'afu': 0.82,
            'anfu': 0.55,
            'tsr': 0.88,
            'atpr': 0.86,
        },
        'data_quality': {
            'overall_quality': 'good',
            'artifact_status': 'VALID',
        },
        'warnings': [],
    }
    return result


def create_history_record(result, sport):
    """Create a history record from a full result, save full result to disk."""
    result_id = _save_full_result(result)

    thresholds = result.get('thresholds', {})
    arch = result.get('archetype', {})
    rv = result.get('ramp_validation', {})
    ev = result.get('effort_validation', {})
    dq = result.get('data_quality', {})
    pace = result.get('pace_data', {})

    record = {
        'test_date': result.get('test_date', datetime.utcnow().isoformat(timespec='seconds')),
        'protocol_type': sport,
        'hrvt1s_power': thresholds.get('hrvt1s_power'),
        'hrvt1s_hr': thresholds.get('hrvt1s_hr'),
        'hrvt1c_power': thresholds.get('hrvt1c_power'),
        'hrvt1c_hr': thresholds.get('hrvt1c_hr'),
        'hrvt2_power': thresholds.get('hrvt2_power'),
        'hrvt2_hr': thresholds.get('hrvt2_hr'),
        'hrvt2_extrapolated': thresholds.get('hrvt2_extrapolated'),
        'a1_star': thresholds.get('a1_star'),
        'regression_r2_power': result.get('regression_power', {}).get('r2'),
        'regression_r2_hr': result.get('regression_hr', {}).get('r2'),
        'stages_completed': rv.get('stages_completed'),
        'ramp_status': rv.get('overall_status'),
        'effort_status': ev.get('status'),
        'max_effort_power': ev.get('avg_power'),
        'archetype': arch.get('archetype'),
        'afu': arch.get('afu'),
        'anfu': arch.get('anfu'),
        'tsr': arch.get('tsr'),
        'atpr': arch.get('atpr'),
        'artifact_pct': result.get('artifact_pct'),
        'data_quality': dq.get('overall_quality'),
        'weight_kg': result.get('weight_kg'),
    }

    if sport == 'run':
        record['hrvt1s_pace'] = pace.get('hrvt1s_pace')
        record['hrvt1c_pace'] = pace.get('hrvt1c_pace')
        record['hrvt2_pace'] = pace.get('hrvt2_pace')
        record['hrmax_run'] = result.get('hrmax_run')
        record['threshold_pace'] = result.get('threshold_pace')
    else:
        record['hrmax_bike'] = result.get('hrmax_bike')

    if result_id:
        record['result_id'] = result_id

    return record


def seed_dummy_data():
    """Create dummy athletes and test data."""
    now = datetime.utcnow()
    history = _load_history()
    athletes = history.setdefault('athletes', {})

    print("Seeding dummy test data...")

    # --- Alex Demo: bike + run pair (within 10 days) ---
    name = 'Alex Demo'
    if name not in athletes:
        athletes[name] = {'ramp_tests': []}

    bike_result = generate_bike_result(
        ftp=220, hrmax=185, weight=72,
        test_date=now - timedelta(days=5), name=name,
    )
    run_result = generate_run_result(
        threshold_pace_min=4.5, hrmax=187, weight=72,
        test_date=now - timedelta(days=3), name=name,
    )

    athletes[name]['ramp_tests'] = [
        create_history_record(bike_result, 'bike'),
        create_history_record(run_result, 'run'),
    ]
    print(f"  {name}: bike (5 days ago) + run (3 days ago) -- auto-combine pair")

    # --- Jordan Demo: bike only ---
    name = 'Jordan Demo'
    if name not in athletes:
        athletes[name] = {'ramp_tests': []}

    bike_result = generate_bike_result(
        ftp=280, hrmax=190, weight=78,
        test_date=now - timedelta(days=14), name=name,
    )

    athletes[name]['ramp_tests'] = [
        create_history_record(bike_result, 'bike'),
    ]
    print(f"  {name}: bike (14 days ago)")

    # --- Sam Demo: two bike tests (progression) ---
    name = 'Sam Demo'
    if name not in athletes:
        athletes[name] = {'ramp_tests': []}

    bike_old = generate_bike_result(
        ftp=190, hrmax=178, weight=65,
        test_date=now - timedelta(days=30), name=name + '_old',
    )
    bike_new = generate_bike_result(
        ftp=200, hrmax=178, weight=64.5,
        test_date=now - timedelta(days=2), name=name + '_new',
    )

    athletes[name]['ramp_tests'] = [
        create_history_record(bike_old, 'bike'),
        create_history_record(bike_new, 'bike'),
    ]
    print(f"  {name}: bike (30 days ago) + bike (2 days ago) -- progression")

    _save_history(history)
    print(f"\nHistory saved to {HISTORY_FILE}")
    print(f"Full results saved to {RESULTS_DIR}/")

    # --- Create user accounts in database ---
    try:
        from app import create_app
        from auth import hash_password

        app = create_app()
        with app.app_context():
            from models import db, User

            dummy_users = [
                {'name': 'Alex Demo', 'email': 'alex@demo.tpc', 'sport': 'both',
                 'threshold_power': 220, 'threshold_pace': '4:30',
                 'hrmax_bike': 185, 'hrmax_run': 187, 'weight_kg': 72},
                {'name': 'Jordan Demo', 'email': 'jordan@demo.tpc', 'sport': 'bike',
                 'threshold_power': 280, 'hrmax_bike': 190, 'weight_kg': 78},
                {'name': 'Sam Demo', 'email': 'sam@demo.tpc', 'sport': 'bike',
                 'threshold_power': 200, 'hrmax_bike': 178, 'weight_kg': 64.5},
            ]

            for u in dummy_users:
                existing = User.query.filter_by(email=u['email']).first()
                if existing:
                    print(f"  User {u['email']} already exists, skipping.")
                    continue

                user = User(
                    email=u['email'],
                    name=u['name'],
                    role='athlete',
                    approved=True,
                    password_hash=hash_password('tpc'),
                    password_must_change=True,
                    sport=u.get('sport'),
                    threshold_power=u.get('threshold_power'),
                    threshold_pace=u.get('threshold_pace'),
                    hrmax_bike=u.get('hrmax_bike'),
                    hrmax_run=u.get('hrmax_run'),
                    weight_kg=u.get('weight_kg'),
                )
                db.session.add(user)
                print(f"  Created user: {u['name']} ({u['email']})")

            db.session.commit()
            print("\nDatabase users created. All passwords: 'tpc'")
    except Exception as e:
        print(f"\nNote: Could not create database users ({e}).")
        print("The test history data was still saved -- users can be created via the admin UI.")


if __name__ == '__main__':
    seed_dummy_data()
