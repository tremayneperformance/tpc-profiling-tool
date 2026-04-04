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
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path so we can import server modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HISTORY_DIR = Path.home() / '.dfatool'
HISTORY_FILE = HISTORY_DIR / 'history.json'
RESULTS_DIR = HISTORY_DIR / 'results'


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


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
    intensities = [0.60, 0.66, 0.71, 0.77, 0.82, 0.88, 0.93, 0.99, 1.04, 1.10]
    stage_duration = 180  # 3 min

    # Generate stage data with realistic DFA decline
    stage_data = []
    windows = []
    timeline = []

    base_hr = int(hrmax * 0.55)
    dfa_start = 1.10

    for i, pct in enumerate(intensities):
        target = round(ftp * pct)
        avg_power = target + round((hash(f"{name}{i}") % 10) - 5)
        hr = int(base_hr + (hrmax - base_hr) * (pct - 0.50) / 0.65)
        hr = min(hr, hrmax)
        dfa = max(0.30, dfa_start - (i * 0.085) + (hash(f"{name}dfa{i}") % 10) / 100)

        stage_data.append({
            'stage': i + 1,
            'power_target': target,
            'power_avg': avg_power,
            'hr_avg': hr,
            'dfa_alpha1': round(dfa, 3),
            'rmssd': round(max(5, 45 - i * 4.2), 1),
            'sdnn': round(max(4, 38 - i * 3.5), 1),
            'rr_count': 180 + (hash(f"{name}rr{i}") % 20),
            'artifact_pct': round(1.5 + (hash(f"{name}art{i}") % 30) / 10, 1),
        })

        # Generate windows (every 30s within stage)
        stage_start = 1200 + i * stage_duration  # warmup=20min offset
        for w in range(6):
            t = stage_start + w * 30
            w_dfa = dfa + (hash(f"{name}w{i}{w}") % 20 - 10) / 100
            windows.append({
                'time': t,
                'power': avg_power + (hash(f"{name}wp{i}{w}") % 20 - 10),
                'hr': hr + (hash(f"{name}wh{i}{w}") % 6 - 3),
                'dfa_alpha1': round(max(0.20, w_dfa), 3),
            })

        # Timeline entries
        for s in range(stage_duration):
            t = stage_start + s
            timeline.append({
                'time': t,
                'power': avg_power + (hash(f"{name}tp{i}{s}") % 30 - 15),
                'hr': hr + (hash(f"{name}th{i}{s}") % 8 - 4),
            })

    # Calculate thresholds via simple interpolation
    # Find where DFA crosses 0.75 (HRVT1)
    hrvt1_power = None
    hrvt1_hr = None
    hrvt2_power = None
    hrvt2_hr = None

    for i in range(1, len(stage_data)):
        prev = stage_data[i-1]
        curr = stage_data[i]
        # HRVT1 at DFA = 0.75
        if prev['dfa_alpha1'] >= 0.75 > curr['dfa_alpha1'] and hrvt1_power is None:
            frac = (0.75 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt1_power = round(curr['power_avg'] - frac * (curr['power_avg'] - prev['power_avg']))
            hrvt1_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))
        # HRVT2 at DFA = 0.50
        if prev['dfa_alpha1'] >= 0.50 > curr['dfa_alpha1'] and hrvt2_power is None:
            frac = (0.50 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt2_power = round(curr['power_avg'] - frac * (curr['power_avg'] - prev['power_avg']))
            hrvt2_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))

    # Fallback if thresholds not found
    if hrvt1_power is None:
        hrvt1_power = round(ftp * 0.72)
        hrvt1_hr = int(hrmax * 0.72)
    if hrvt2_power is None:
        hrvt2_power = round(ftp * 0.92)
        hrvt2_hr = int(hrmax * 0.87)

    # MAP estimation
    map_power = round(ftp * 1.20)

    result = {
        'protocol_type': 'bike',
        'test_date': test_date.isoformat(),
        'athlete_name': name,
        'ftp': ftp,
        'weight_kg': weight,
        'hrmax': hrmax,
        'stage_data': stage_data,
        'windows': windows,
        'timeline': timeline[:300],  # Keep reasonable size
        'segments': {
            'warmup': {'start': 0, 'end': 1200},
            'ramp': {'start': 1200, 'end': 4800},
            'recovery': {'start': 4800, 'end': 5280},
        },
        'hrvt1s_power': round(hrvt1_power * 0.98),
        'hrvt1s_hr': hrvt1_hr - 2 if hrvt1_hr else None,
        'hrvt1c_power': hrvt1_power,
        'hrvt1c_hr': hrvt1_hr,
        'hrvt2_power': hrvt2_power,
        'hrvt2_hr': hrvt2_hr,
        'map_power': map_power,
        'map_corrected': round(map_power * 0.95),
        'regression_r2_power': round(0.92 + (hash(f"{name}r2") % 7) / 100, 3),
        'data_quality': 'good',
        'archetype': 'balanced',
        'hrmax_bike': hrmax,
        'total_rr': sum(s['rr_count'] for s in stage_data),
        'artifact_pct': round(sum(s['artifact_pct'] for s in stage_data) / len(stage_data), 1),
    }
    return result


def generate_run_result(threshold_pace_min, hrmax, weight, test_date, name):
    """Generate a realistic run ramp test result."""
    intensities = [0.70, 0.74, 0.78, 0.82, 0.86, 0.91, 0.95, 0.99, 1.03, 1.07]
    stage_duration = 180

    stage_data = []
    base_hr = int(hrmax * 0.58)
    dfa_start = 1.05

    for i, pct in enumerate(intensities):
        pace = threshold_pace_min / pct  # slower than threshold
        hr = int(base_hr + (hrmax - base_hr) * (pct - 0.60) / 0.55)
        hr = min(hr, hrmax)
        dfa = max(0.28, dfa_start - (i * 0.080) + (hash(f"{name}rdfa{i}") % 10) / 100)

        stage_data.append({
            'stage': i + 1,
            'pace': round(pace, 2),
            'hr_avg': hr,
            'dfa_alpha1': round(dfa, 3),
            'rmssd': round(max(4, 42 - i * 3.8), 1),
            'sdnn': round(max(3, 35 - i * 3.2), 1),
            'rr_count': 175 + (hash(f"{name}rrr{i}") % 25),
            'artifact_pct': round(2.0 + (hash(f"{name}rart{i}") % 35) / 10, 1),
        })

    # Calculate thresholds
    hrvt1_pace = None
    hrvt1_hr = None
    hrvt2_pace = None
    hrvt2_hr = None

    for i in range(1, len(stage_data)):
        prev = stage_data[i-1]
        curr = stage_data[i]
        if prev['dfa_alpha1'] >= 0.75 > curr['dfa_alpha1'] and hrvt1_pace is None:
            frac = (0.75 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt1_pace = round(curr['pace'] + frac * (prev['pace'] - curr['pace']), 2)
            hrvt1_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))
        if prev['dfa_alpha1'] >= 0.50 > curr['dfa_alpha1'] and hrvt2_pace is None:
            frac = (0.50 - curr['dfa_alpha1']) / (prev['dfa_alpha1'] - curr['dfa_alpha1'])
            hrvt2_pace = round(curr['pace'] + frac * (prev['pace'] - curr['pace']), 2)
            hrvt2_hr = round(curr['hr_avg'] - frac * (curr['hr_avg'] - prev['hr_avg']))

    if hrvt1_pace is None:
        hrvt1_pace = round(threshold_pace_min / 0.78, 2)
        hrvt1_hr = int(hrmax * 0.74)
    if hrvt2_pace is None:
        hrvt2_pace = round(threshold_pace_min / 0.95, 2)
        hrvt2_hr = int(hrmax * 0.88)

    def fmt_pace(p):
        m = int(p)
        s = int((p - m) * 60)
        return f"{m}:{s:02d}"

    result = {
        'protocol_type': 'run',
        'test_date': test_date.isoformat(),
        'athlete_name': name,
        'threshold_pace': threshold_pace_min,
        'weight_kg': weight,
        'hrmax': hrmax,
        'stage_data': stage_data,
        'hrvt1s_pace': fmt_pace(round(hrvt1_pace * 1.02, 2)),
        'hrvt1s_hr': hrvt1_hr - 2 if hrvt1_hr else None,
        'hrvt1c_pace': fmt_pace(hrvt1_pace),
        'hrvt1c_hr': hrvt1_hr,
        'hrvt2_pace': fmt_pace(hrvt2_pace),
        'hrvt2_hr': hrvt2_hr,
        'regression_r2_power': round(0.90 + (hash(f"{name}rr2") % 8) / 100, 3),
        'data_quality': 'good',
        'archetype': 'balanced',
        'hrmax_run': hrmax,
        'threshold_pace': fmt_pace(threshold_pace_min),
        'total_rr': sum(s['rr_count'] for s in stage_data),
        'artifact_pct': round(sum(s['artifact_pct'] for s in stage_data) / len(stage_data), 1),
    }
    return result


def create_history_record(result, sport):
    """Create a history record from a full result, save full result to disk."""
    result_id = _save_full_result(result)

    record = {
        'test_date': result['test_date'],
        'protocol_type': sport,
        'data_quality': result.get('data_quality', 'unknown'),
        'archetype': result.get('archetype'),
        'regression_r2_power': result.get('regression_r2_power'),
        'total_rr': result.get('total_rr', 0),
        'artifact_pct': result.get('artifact_pct', 0),
    }

    if sport == 'bike':
        record['hrvt1s_power'] = result.get('hrvt1s_power')
        record['hrvt1s_hr'] = result.get('hrvt1s_hr')
        record['hrvt1c_power'] = result.get('hrvt1c_power')
        record['hrvt1c_hr'] = result.get('hrvt1c_hr')
        record['hrvt2_power'] = result.get('hrvt2_power')
        record['hrvt2_hr'] = result.get('hrvt2_hr')
        record['map_power'] = result.get('map_power')
        record['map_corrected'] = result.get('map_corrected')
        record['hrmax_bike'] = result.get('hrmax_bike')
    else:
        record['hrvt1s_pace'] = result.get('hrvt1s_pace')
        record['hrvt1s_hr'] = result.get('hrvt1s_hr')
        record['hrvt1c_pace'] = result.get('hrvt1c_pace')
        record['hrvt1c_hr'] = result.get('hrvt1c_hr')
        record['hrvt2_pace'] = result.get('hrvt2_pace')
        record['hrvt2_hr'] = result.get('hrvt2_hr')
        record['hrmax_run'] = result.get('hrmax_run')
        record['threshold_pace'] = result.get('threshold_pace')

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

    bike_result = generate_bike_result(ftp=220, hrmax=185, weight=72, test_date=now - timedelta(days=5), name=name)
    run_result = generate_run_result(threshold_pace_min=4.5, hrmax=187, weight=72, test_date=now - timedelta(days=3), name=name)

    athletes[name]['ramp_tests'] = [
        create_history_record(bike_result, 'bike'),
        create_history_record(run_result, 'run'),
    ]
    print(f"  {name}: bike (5 days ago) + run (3 days ago) — auto-combine pair")

    # --- Jordan Demo: bike only ---
    name = 'Jordan Demo'
    if name not in athletes:
        athletes[name] = {'ramp_tests': []}

    bike_result = generate_bike_result(ftp=280, hrmax=190, weight=78, test_date=now - timedelta(days=14), name=name)

    athletes[name]['ramp_tests'] = [
        create_history_record(bike_result, 'bike'),
    ]
    print(f"  {name}: bike (14 days ago)")

    # --- Sam Demo: two bike tests (progression) ---
    name = 'Sam Demo'
    if name not in athletes:
        athletes[name] = {'ramp_tests': []}

    bike_old = generate_bike_result(ftp=190, hrmax=178, weight=65, test_date=now - timedelta(days=30), name=name + '_old')
    bike_new = generate_bike_result(ftp=200, hrmax=178, weight=64.5, test_date=now - timedelta(days=2), name=name + '_new')

    athletes[name]['ramp_tests'] = [
        create_history_record(bike_old, 'bike'),
        create_history_record(bike_new, 'bike'),
    ]
    print(f"  {name}: bike (30 days ago) + bike (2 days ago) — progression")

    _save_history(history)
    print(f"\nHistory saved to {HISTORY_FILE}")
    print(f"Full results saved to {RESULTS_DIR}/")

    # --- Create user accounts in database ---
    try:
        from server.app import create_app
        app = create_app()
        with app.app_context():
            from server.models import db, User
            from server.auth import hash_password

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
        print("The test history data was still saved — users can be created via the admin UI.")


if __name__ == '__main__':
    seed_dummy_data()
