#!/usr/bin/env python3
"""
Seed script — creates dummy athletes, test sessions, and full analysis
history so the coach can click through the entire UI end-to-end.

Usage:
    cd server
    python seed_dummy_data.py

Creates:
  - 3 dummy athletes in the DB (with sessions + records + RR intervals)
  - Matching entries in ~/.dfatool/history.json with full result files
  - One athlete has a bike+run pair within 10 days (auto-combine test)

All passwords are "tpc" (default).  Safe to run multiple times — it
skips athletes whose email already exists.
"""

import os
import sys
import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta

# Ensure we can import server modules
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault('DATABASE_URL', 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'ramptool.db'))

from app import create_app
from models import db, User, TestSession, TestRecord, RRInterval
from auth import hash_password
from ramp_analysis import HISTORY_DIR, HISTORY_FILE, RESULTS_DIR

app = create_app()

# ──────────────────────────────────────────────────────────────────────
# ATHLETE PROFILES
# ──────────────────────────────────────────────────────────────────────

ATHLETES = [
    {
        'name': 'Alex Demo',
        'email': 'alex.demo@test.com',
        'weight_kg': 75.0,
        'hrmax_bike': 185,
        'hrmax_run': 190,
        'threshold_power': 260,
        'threshold_pace': '4:15',
        'tests': [
            # Bike test — 5 days ago
            {'sport': 'bike', 'days_ago': 5, 'ftp': 260, 'hrmax': 185},
            # Run test — 3 days ago (within 10 days → auto-combine)
            {'sport': 'run', 'days_ago': 3, 'ftp': 260, 'hrmax': 190},
        ],
    },
    {
        'name': 'Jordan Demo',
        'email': 'jordan.demo@test.com',
        'weight_kg': 68.0,
        'hrmax_bike': 178,
        'hrmax_run': 183,
        'threshold_power': 230,
        'threshold_pace': '4:35',
        'tests': [
            # Bike test only — 14 days ago
            {'sport': 'bike', 'days_ago': 14, 'ftp': 230, 'hrmax': 178},
        ],
    },
    {
        'name': 'Sam Demo',
        'email': 'sam.demo@test.com',
        'weight_kg': 82.0,
        'hrmax_bike': 192,
        'hrmax_run': 195,
        'threshold_power': 290,
        'threshold_pace': '4:00',
        'tests': [
            # Bike — 30 days ago
            {'sport': 'bike', 'days_ago': 30, 'ftp': 280, 'hrmax': 192},
            # Bike — 2 days ago (shows progression)
            {'sport': 'bike', 'days_ago': 2, 'ftp': 290, 'hrmax': 192},
        ],
    },
]

# ──────────────────────────────────────────────────────────────────────
# GENERATE REALISTIC DFA / RAMP DATA
# ──────────────────────────────────────────────────────────────────────

def generate_stage_data(sport, ftp, hrmax, n_stages=10):
    """Generate realistic per-stage DFA alpha1 data for a ramp test."""
    stages = []
    # Power starts at 60% FTP, increments ~5% per stage
    for i in range(n_stages):
        pct = 0.60 + i * 0.05
        power = int(ftp * pct)
        # HR rises roughly linearly
        hr_pct = 0.60 + i * 0.04
        hr = int(hrmax * hr_pct)
        # DFA alpha1 declines from ~1.1 to ~0.4
        a1 = max(0.35, 1.10 - i * 0.08 + random.gauss(0, 0.02))
        stages.append({
            'stage_number': i + 1,
            'mean_power': power if sport == 'bike' else round(1000 / (ftp * pct / 60), 2),  # approx speed m/s for run
            'mean_hr': hr,
            'a1_mean': round(a1, 3),
            'a1_sd': round(random.uniform(0.03, 0.08), 3),
            'duration_sec': 180,
        })
    return stages


def generate_windows(stage_data, sport):
    """Generate per-window DFA data points (every ~30s across the test)."""
    windows = []
    t = 0
    # Warmup windows (20 min)
    for _ in range(40):
        t += 30
        windows.append({
            'time': t,
            'power': stage_data[0]['mean_power'] * 0.75 if sport == 'bike' else stage_data[0]['mean_power'] * 0.75,
            'hr': int(stage_data[0]['mean_hr'] * 0.85),
            'alpha1': round(random.uniform(1.0, 1.2), 3),
            'reliable': True,
        })
    # Stage windows (6 per 3-min stage)
    for s in stage_data:
        for j in range(6):
            t += 30
            noise = random.gauss(0, 0.03)
            windows.append({
                'time': t,
                'power': s['mean_power'] + random.randint(-3, 3),
                'hr': s['mean_hr'] + random.randint(-2, 2),
                'alpha1': round(max(0.2, s['a1_mean'] + noise), 3),
                'reliable': True,
            })
    return windows


def generate_timeline(stage_data, sport, ftp, hrmax):
    """Generate per-second timeline for the session chart."""
    timeline = []
    t = 0
    # Warmup: 20 min @ 45% FTP
    warmup_power = int(ftp * 0.45)
    warmup_hr_start = int(hrmax * 0.55)
    for sec in range(1200):
        t += 1
        hr = warmup_hr_start + int((sec / 1200) * 15)
        timeline.append({'t': t, 'p': warmup_power + random.randint(-3, 3), 'h': hr + random.randint(-1, 1)})

    # Ramp stages: 10 × 3 min
    for s in stage_data:
        for sec in range(180):
            t += 1
            timeline.append({'t': t, 'p': s['mean_power'] + random.randint(-5, 5), 'h': s['mean_hr'] + random.randint(-2, 2)})

    # Recovery: 8 min
    rec_power = warmup_power
    rec_hr = stage_data[-1]['mean_hr']
    for sec in range(480):
        t += 1
        hr = rec_hr - int((sec / 480) * 30)
        timeline.append({'t': t, 'p': rec_power + random.randint(-3, 3), 'h': hr + random.randint(-1, 1)})

    return timeline


def generate_segments(stage_data):
    """Generate segment boundaries."""
    warmup_end = 1200
    stages = []
    t = warmup_end
    for s in stage_data:
        stages.append({
            'stage_number': s['stage_number'],
            'start_sec': t,
            'end_sec': t + 180,
            'duration_sec': 180,
            'mean_power': s['mean_power'],
        })
        t += 180
    ramp_end = t
    recovery_end = ramp_end + 480
    return {
        'warmup': [0, warmup_end],
        'ramp_start': warmup_end,
        'ramp_end': ramp_end,
        'warmup_power': stage_data[0]['mean_power'] * 0.75,
        'ramp_peak_power': stage_data[-1]['mean_power'],
        'stages': stages,
        'recovery': [ramp_end, recovery_end],
        'max_effort': [recovery_end, recovery_end + 180],
        'cooldown_start': recovery_end + 180,
    }


def compute_thresholds(stage_data, sport):
    """Compute realistic threshold values from stage data via linear regression."""
    # Simple linear fit: a1 = slope * power + intercept
    powers = [s['mean_power'] for s in stage_data]
    a1s = [s['a1_mean'] for s in stage_data]

    n = len(powers)
    sum_x = sum(powers)
    sum_y = sum(a1s)
    sum_xy = sum(p * a for p, a in zip(powers, a1s))
    sum_xx = sum(p * p for p in powers)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        slope, intercept = -0.002, 1.5
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

    # R² calculation
    y_mean = sum_y / n
    ss_tot = sum((a - y_mean) ** 2 for a in a1s)
    ss_res = sum((a - (slope * p + intercept)) ** 2 for p, a in zip(powers, a1s))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.95

    # Thresholds
    a1_max = max(a1s[:3])  # max from early stages
    a1_star = round((a1_max + 0.50) / 2, 3)

    def power_at_a1(target_a1):
        if slope == 0:
            return powers[n // 2]
        return round((target_a1 - intercept) / slope, 1)

    hrvt1s_power = power_at_a1(0.75)
    hrvt1c_power = power_at_a1(a1_star)
    hrvt2_power = power_at_a1(0.50)

    # HR at thresholds (linear interpolation)
    hrs = [s['mean_hr'] for s in stage_data]
    hr_slope = (hrs[-1] - hrs[0]) / (powers[-1] - powers[0]) if powers[-1] != powers[0] else 0.5

    def hr_at_power(p):
        return int(hrs[0] + hr_slope * (p - powers[0]))

    return {
        'hrvt1s_power': round(hrvt1s_power),
        'hrvt1s_hr': hr_at_power(hrvt1s_power),
        'hrvt1c_power': round(hrvt1c_power),
        'hrvt1c_hr': hr_at_power(hrvt1c_power),
        'hrvt2_power': round(hrvt2_power),
        'hrvt2_hr': hr_at_power(hrvt2_power),
        'a1_star': a1_star,
        'hrvt2_extrapolated': hrvt2_power > powers[-1],
    }, {
        'slope': round(slope, 6),
        'intercept': round(intercept, 4),
        'r2': round(r2, 4),
        'n': n,
    }


def generate_full_result(sport, ftp, hrmax, weight_kg):
    """Generate a complete analysis result object matching what the backend produces."""
    stage_data = generate_stage_data(sport, ftp, hrmax)
    windows = generate_windows(stage_data, sport)
    timeline = generate_timeline(stage_data, sport, ftp, hrmax)
    segments = generate_segments(stage_data)
    thresholds, regression_power = compute_thresholds(stage_data, sport)

    # HR regression (similar to power but vs HR)
    regression_hr = {
        'slope': round(regression_power['slope'] * 0.8, 6),
        'intercept': round(regression_power['intercept'] * 1.1, 4),
        'r2': round(regression_power['r2'] * 0.98, 4),
        'n': regression_power['n'],
    }

    # Archetype
    archetypes = ['Balanced', 'High Aerobic', 'High Anaerobic', 'Aerobic Ceiling Limited']
    archetype_pick = random.choice(archetypes)

    result = {
        'status': 'ok',
        'source': 'HRV',
        'protocol_type': sport,
        'stage_data': stage_data,
        'windows': windows,
        'timeline': timeline,
        'segments': segments,
        'thresholds': thresholds,
        'regression_power': regression_power,
        'regression_hr': regression_hr,
        'ramp_validation': {
            'overall_status': 'VALID',
            'stages_completed': len(stage_data),
            'monotonic_power': True,
            'adequate_duration': True,
        },
        'effort_validation': {
            'status': 'VALID',
            'avg_power': int(ftp * 1.15),
            'max_hr': hrmax - random.randint(2, 8),
            'duration_sec': 185,
        },
        'data_quality': {
            'overall_quality': 'good',
            'artifact_rate_ramp': round(random.uniform(1.0, 5.0), 1),
            'warmup_stable': True,
            'ramp_monotonic': True,
        },
        'artifact_pct': round(random.uniform(1.0, 5.0), 1),
        'archetype': {
            'archetype': archetype_pick,
            'confidence': 'high',
            'method': 'full',
            'afu': round(random.uniform(0.3, 0.8), 2),
            'anfu': round(random.uniform(0.2, 0.7), 2),
            'ar': round(random.uniform(0.15, 0.5), 2),
            'tsr': round(random.uniform(0.5, 0.9), 2),
            'atpr': round(random.uniform(0.4, 0.8), 2),
            'ceiling_limited': archetype_pick == 'Aerobic Ceiling Limited',
            'development_level': {
                'level': random.choice(['Intermediate', 'Advanced']),
                'note': 'Based on threshold power relative to body weight.',
            },
            'feedback': {
                'strengths': ['Good aerobic base', 'Consistent pacing'],
                'weaknesses': ['Anaerobic capacity could improve'],
            },
            'training_recommendations': [
                'Continue building aerobic volume at Zone 2',
                'Add threshold intervals 2x per week',
            ],
            'recommendation': 'Focus on polarised training with emphasis on Z2 volume.',
            'flags': [],
        },
        'warnings': [],
        'weight_kg': weight_kg,
    }

    # Add pace data for run
    if sport == 'run':
        result['pace_data'] = {
            'hrvt1s_pace': '4:45',
            'hrvt1c_pace': '4:25',
            'hrvt2_pace': '4:05',
        }

    return result


def generate_test_records(timeline, sport, ftp):
    """Generate TestRecord data from timeline (subsample to keep DB manageable)."""
    records = []
    for i, pt in enumerate(timeline):
        if i % 5 != 0:  # Every 5 seconds to keep DB size down
            continue
        elapsed = pt['t']
        # Determine phase
        if elapsed <= 1200:
            phase = 'warmup'
            stage = None
        elif elapsed <= 1200 + 1800:
            phase = 'ramp'
            stage = min(10, (elapsed - 1200) // 180 + 1)
        elif elapsed <= 1200 + 1800 + 480:
            phase = 'recovery'
            stage = None
        else:
            phase = 'cooldown'
            stage = None

        records.append({
            'elapsed_sec': elapsed,
            'power': pt['p'],
            'heart_rate': pt['h'],
            'cadence': random.randint(80, 95) if sport == 'bike' else None,
            'target_power': int(ftp * 0.45) if phase == 'warmup' else pt['p'],
            'phase': phase,
            'stage_num': stage,
        })
    return records


def generate_rr_intervals(timeline, hrmax):
    """Generate dummy RR interval data from HR timeline."""
    rr_data = []
    t_ms = 0
    for pt in timeline:
        if pt['t'] % 3 != 0:  # Every 3 seconds
            continue
        hr = pt['h'] or 120
        rr_ms = 60000 / max(hr, 40)
        # Add a few beats per timestamp
        for _ in range(3):
            rr = rr_ms + random.gauss(0, 15)
            rr_data.append({'timestamp_ms': int(t_ms), 'rr_ms': round(rr, 1)})
            t_ms += rr
    return rr_data


# ──────────────────────────────────────────────────────────────────────
# SEED THE DATABASE + HISTORY
# ──────────────────────────────────────────────────────────────────────

def seed():
    with app.app_context():
        # Load or init history
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = {'athletes': {}}

        for athlete_def in ATHLETES:
            email = athlete_def['email']
            name = athlete_def['name']

            # Check if athlete already exists
            existing = User.query.filter_by(email=email).first()
            if existing:
                print(f'  [SKIP] {name} ({email}) already exists (id={existing.id})')
                user = existing
            else:
                user = User(
                    email=email,
                    name=name,
                    role='athlete',
                    approved=True,
                    password_hash=hash_password('tpc'),
                    password_must_change=True,
                    weight_kg=athlete_def['weight_kg'],
                    hrmax_bike=athlete_def['hrmax_bike'],
                    hrmax_run=athlete_def['hrmax_run'],
                    threshold_power=athlete_def['threshold_power'],
                    threshold_pace=athlete_def['threshold_pace'],
                    last_login=datetime.utcnow() - timedelta(days=1),
                )
                db.session.add(user)
                db.session.flush()
                print(f'  [ADD]  {name} ({email}) → id={user.id}')

            # Ensure athlete has history entry
            if name not in history['athletes']:
                history['athletes'][name] = {'ramp_tests': []}

            # Generate tests
            for test_def in athlete_def['tests']:
                sport = test_def['sport']
                ftp = test_def['ftp']
                hrmax = test_def['hrmax']
                test_date = datetime.utcnow() - timedelta(days=test_def['days_ago'])

                # Check if session already exists for this date/sport
                existing_session = TestSession.query.filter_by(
                    athlete_id=user.id, sport=sport
                ).filter(
                    TestSession.test_date >= test_date - timedelta(hours=12),
                    TestSession.test_date <= test_date + timedelta(hours=12),
                ).first()

                if existing_session:
                    print(f'         [SKIP] {sport} test for {name} already exists (session id={existing_session.id})')
                    continue

                # Generate full result
                full_result = generate_full_result(sport, ftp, hrmax, athlete_def['weight_kg'])
                thresholds = full_result['thresholds']
                stage_data = full_result['stage_data']
                timeline = full_result['timeline']

                # Create DB session
                duration_sec = len(timeline)
                session = TestSession(
                    athlete_id=user.id,
                    sport=sport,
                    test_date=test_date,
                    duration_sec=duration_sec,
                    stages_completed=len(stage_data),
                    peak_power=int(ftp * 1.15),
                    peak_hr=hrmax - random.randint(2, 6),
                    artifact_pct=full_result['artifact_pct'],
                    quality='good',
                    threshold_value=ftp if sport == 'bike' else None,
                    hrmax=hrmax,
                    weight_kg=athlete_def['weight_kg'],
                    hrvt1s_power=thresholds['hrvt1s_power'],
                    hrvt1s_hr=thresholds['hrvt1s_hr'],
                    hrvt1c_power=thresholds['hrvt1c_power'],
                    hrvt1c_hr=thresholds['hrvt1c_hr'],
                    hrvt2_power=thresholds['hrvt2_power'],
                    hrvt2_hr=thresholds['hrvt2_hr'],
                    archetype=full_result['archetype']['archetype'],
                )
                db.session.add(session)
                db.session.flush()

                # Insert sample records (subsampled)
                records = generate_test_records(timeline, sport, ftp)
                for r in records:
                    db.session.add(TestRecord(
                        session_id=session.id,
                        elapsed_sec=r['elapsed_sec'],
                        power=r['power'],
                        heart_rate=r['heart_rate'],
                        cadence=r['cadence'],
                        target_power=r['target_power'],
                        phase=r['phase'],
                        stage_num=r['stage_num'],
                    ))

                # Insert RR intervals
                rr_data = generate_rr_intervals(timeline, hrmax)
                for rr in rr_data:
                    db.session.add(RRInterval(
                        session_id=session.id,
                        timestamp_ms=rr['timestamp_ms'],
                        rr_ms=rr['rr_ms'],
                    ))

                # Save full result to file
                ts_str = test_date.strftime('%Y%m%d_%H%M%S')
                result_id = f'{ts_str}_{sport}'
                result_path = RESULTS_DIR / f'{result_id}.json'
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(full_result, f, ensure_ascii=False)

                # Add to history.json
                record = {
                    'test_date': test_date.isoformat(timespec='seconds'),
                    'hrvt1s_power': thresholds['hrvt1s_power'],
                    'hrvt1s_hr': thresholds['hrvt1s_hr'],
                    'hrvt1c_power': thresholds['hrvt1c_power'],
                    'hrvt1c_hr': thresholds['hrvt1c_hr'],
                    'a1_star': thresholds['a1_star'],
                    'hrvt2_power': thresholds['hrvt2_power'],
                    'hrvt2_hr': thresholds['hrvt2_hr'],
                    'hrvt2_extrapolated': thresholds['hrvt2_extrapolated'],
                    'regression_r2_power': full_result['regression_power']['r2'],
                    'regression_r2_hr': full_result['regression_hr']['r2'],
                    'stages_completed': len(stage_data),
                    'ramp_status': 'VALID',
                    'effort_status': 'VALID',
                    'max_effort_power': int(ftp * 1.15),
                    'archetype': full_result['archetype']['archetype'],
                    'afu': full_result['archetype']['afu'],
                    'anfu': full_result['archetype']['anfu'],
                    'tsr': full_result['archetype']['tsr'],
                    'atpr': full_result['archetype']['atpr'],
                    'artifact_pct': full_result['artifact_pct'],
                    'data_quality': 'good',
                    'protocol_type': sport,
                    'weight_kg': athlete_def['weight_kg'],
                    'result_id': result_id,
                    'hrmax_bike': athlete_def['hrmax_bike'],
                    'hrmax_run': athlete_def['hrmax_run'],
                }
                if sport == 'run':
                    record['threshold_pace'] = athlete_def['threshold_pace']

                history['athletes'][name]['ramp_tests'].append(record)

                print(f'         [ADD]  {sport} test → session id={session.id}, result_id={result_id}')

        db.session.commit()

        # Write history file
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f'\nHistory saved to {HISTORY_FILE}')
        print(f'Full results saved to {RESULTS_DIR}/')
        print('\nDummy accounts (password: "tpc"):')
        for a in ATHLETES:
            print(f'  {a["email"]}  —  {a["name"]}')
        print('\nDone! Start the server with: python app.py')


if __name__ == '__main__':
    print('Seeding dummy data...\n')
    seed()
