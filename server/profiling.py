"""
Athlete Profiling & Training Prescription Module

This module takes DFA α1 threshold data (HRVT1/HRVT2) and athlete profile
information (HRmax) to generate a physiological profile with training recommendations.

IMPORTANT PHYSIOLOGICAL CAVEATS (do not remove these comments):

1. %HRmax ranges are GUIDELINES, not diagnostic criteria. Individual variation is
   large. A 65-year-old with HRmax 155 has different %HRmax distributions than a
   25-year-old with HRmax 200, even at equivalent fitness.

2. HRmax accuracy is the FOUNDATION. If HRmax is wrong, the entire profile is
   skewed. Estimated HRmax (e.g., 220-age) may be off by ±10-15 bpm. Always prefer
   a value from a true maximal test.

3. DFA α1 thresholds APPROXIMATE but do not perfectly equal lactate thresholds.
   HRVT1 ≈ VT1 ≈ LT1 (≈2 mmol/L lactate). HRVT2 ≈ VT2 ≈ MLSS ≈ LT2 (≈4 mmol/L).
   Typical agreement error: ±5-10W or ±3-5 bpm. Treat as estimates, not absolutes.

4. Power/pace thresholds are more ACTIONABLE for daily training prescription.
   HR thresholds are more STABLE for profiling because they are less affected by
   equipment calibration, temperature, caffeine, and day-to-day variation.

5. NORWEGIAN TRAINING MODEL CONTEXT: This tool is used within a coaching philosophy
   that emphasises sub-threshold volume. When in doubt, prescriptions lean toward
   "more easy volume" rather than "more intensity."

6. These results are DATA-INFORMED SUGGESTIONS for the coach to interpret.
   They are not medical advice and should never be presented as such.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# HISTORY STORAGE
# ---------------------------------------------------------------------------

def get_history_path() -> Path:
    """Return the path to the history file, creating the directory if needed."""
    data_dir = Path.home() / '.dfatool'
    data_dir.mkdir(exist_ok=True)
    return data_dir / 'history.json'


def load_history() -> dict:
    """Load the full history dict from disk. Returns empty structure on error."""
    path = get_history_path()
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


def save_test_to_history(athlete_name: str, hrmax: int, discipline: str,
                         lt1_hr: Optional[float], lt2_hr: Optional[float],
                         lt1_power: Optional[float] = None,
                         lt2_power: Optional[float] = None,
                         lt1_pct: Optional[float] = None,
                         lt2_pct: Optional[float] = None) -> bool:
    """
    Append a test result to the athlete's history. Returns True on success.
    """
    history = load_history()
    name = athlete_name.strip() or 'Unknown Athlete'
    if name not in history['athletes']:
        history['athletes'][name] = {'tests': []}

    record = {
        'test_date': datetime.now().isoformat(timespec='seconds'),
        'discipline': discipline,
        'hrmax': hrmax,
        'lt1_hr': lt1_hr,
        'lt2_hr': lt2_hr,
        'lt1_power': lt1_power,
        'lt2_power': lt2_power,
        'lt1_pct': lt1_pct,
        'lt2_pct': lt2_pct,
    }
    history['athletes'][name]['tests'].append(record)

    try:
        with open(get_history_path(), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def get_athlete_history(athlete_name: str,
                        discipline: Optional[str] = None) -> List[dict]:
    """
    Return all tests for an athlete, optionally filtered by discipline,
    sorted by date ascending (oldest first). The most recent test is last.
    """
    history = load_history()
    name = athlete_name.strip()
    tests = history.get('athletes', {}).get(name, {}).get('tests', [])
    if discipline:
        tests = [t for t in tests if t.get('discipline') == discipline]
    return sorted(tests, key=lambda t: t.get('test_date', ''))


def get_previous_test(athlete_name: str, discipline: str,
                      exclude_latest: bool = True) -> Optional[dict]:
    """
    Return the second-most-recent test for this athlete/discipline combination
    (i.e., the one before the current session), or None if none exists.
    """
    tests = get_athlete_history(athlete_name, discipline)
    if len(tests) < (2 if exclude_latest else 1):
        return None
    # Return second-to-last (the one before the most recent)
    return tests[-2] if exclude_latest else tests[-1]


def list_athletes() -> List[str]:
    """Return sorted list of known athlete names."""
    return sorted(load_history().get('athletes', {}).keys())


def update_test_in_history(athlete_name: str, test_index: int,
                           updates: dict) -> bool:
    """Patch fields of a specific DFA test by index. Returns True on success."""
    history = load_history()
    name = athlete_name.strip()
    tests = history.get('athletes', {}).get(name, {}).get('tests', [])
    if test_index < 0 or test_index >= len(tests):
        return False
    tests[test_index].update(updates)
    try:
        with open(get_history_path(), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def delete_test_from_history(athlete_name: str, test_index: int) -> bool:
    """Remove a specific DFA test by index. Returns True on success."""
    history = load_history()
    name = athlete_name.strip()
    tests = history.get('athletes', {}).get(name, {}).get('tests', [])
    if test_index < 0 or test_index >= len(tests):
        return False
    tests.pop(test_index)
    try:
        with open(get_history_path(), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def get_population_averages(discipline: Optional[str] = None) -> dict:
    """
    Compute average normalised DFA profile across all athletes.
    Uses each athlete's most recent test only to avoid over-weighting.
    Returns avg_lt1_pct, avg_lt2_pct, avg_gap_pct, athlete_count.
    """
    history = load_history()
    lt1_pcts = []
    lt2_pcts = []
    gap_pcts = []

    for name, data in history.get('athletes', {}).items():
        tests = data.get('tests', [])
        if discipline:
            tests = [t for t in tests if t.get('discipline') == discipline]
        if not tests:
            continue
        latest = sorted(tests, key=lambda t: t.get('test_date', ''))[-1]
        if latest.get('lt1_pct') is not None:
            lt1_pcts.append(latest['lt1_pct'])
        if latest.get('lt2_pct') is not None:
            lt2_pcts.append(latest['lt2_pct'])
        if latest.get('lt1_pct') is not None and latest.get('lt2_pct') is not None:
            gap_pcts.append(latest['lt2_pct'] - latest['lt1_pct'])

    def _avg(vals):
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        'avg_lt1_pct': _avg(lt1_pcts),
        'avg_lt2_pct': _avg(lt2_pcts),
        'avg_gap_pct': _avg(gap_pcts),
        'athlete_count': max(len(lt1_pcts), len(lt2_pcts)),
    }


# ---------------------------------------------------------------------------
# LT1 %HRMAX CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_lt1(lt1_pct: float) -> dict:
    """
    Classify LT1 as a percentage of HRmax.

    Ranges are approximate guidelines — individual variation is large.
    Always present as 'suggests' or 'indicates', not as a definitive diagnosis.
    """
    if lt1_pct < 65:
        return {
            'label': 'Underdeveloped',
            'css': 'underdeveloped',
            'meaning': (
                'Aerobic threshold crosses very early. The athlete may be undertrained, '
                'deconditioned, or has done too much intensity relative to volume. '
                'Very common in beginner triathletes or after a long off-season.'
            ),
        }
    elif lt1_pct < 72:
        return {
            'label': 'Developing',
            'css': 'developing',
            'meaning': (
                'Typical of recreational athletes or those early in a base phase. '
                'Solid foundation being built, but significant room for improvement '
                'through consistent sub-threshold volume.'
            ),
        }
    elif lt1_pct < 78:
        return {
            'label': 'Well-Developed',
            'css': 'well-developed',
            'meaning': (
                'Good aerobic engine. The athlete can sustain meaningful work while '
                'staying aerobic. Common in well-trained age-group triathletes.'
            ),
        }
    elif lt1_pct < 85:
        return {
            'label': 'Strong',
            'css': 'strong',
            'meaning': (
                'Excellent fat oxidation and mitochondrial density. Typically seen after '
                'sustained high-volume training blocks or in athletes with years of '
                'endurance background.'
            ),
        }
    else:
        return {
            'label': 'Elite',
            'css': 'elite',
            'meaning': (
                'Exceptional aerobic base. A very high proportion of HRmax is usable '
                'before crossing the aerobic threshold. Seen in elite/professional '
                'endurance athletes. Verify HRmax is from a true max test and rule out '
                'beta-blocker use.'
            ),
        }


# ---------------------------------------------------------------------------
# LT2 %HRMAX CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_lt2(lt2_pct: float) -> dict:
    if lt2_pct < 78:
        return {
            'label': 'Low Threshold',
            'css': 'low',
            'meaning': (
                'Lactate accumulates quickly above LT1. Limited ability to sustain '
                'moderate-hard efforts. Significant room for threshold development.'
            ),
        }
    elif lt2_pct < 84:
        return {
            'label': 'Developing Threshold',
            'css': 'developing',
            'meaning': (
                'Typical of recreational endurance athletes. Can hold tempo efforts '
                'but fatigues relatively quickly above threshold.'
            ),
        }
    elif lt2_pct < 88:
        return {
            'label': 'Good Threshold',
            'css': 'good',
            'meaning': (
                'Well-developed lactate clearance. Common in trained age-group athletes. '
                'Can sustain efforts close to threshold for meaningful durations.'
            ),
        }
    elif lt2_pct < 92:
        return {
            'label': 'Strong Threshold',
            'css': 'strong',
            'meaning': (
                'High-level endurance fitness. Large usable HR range for racing. '
                'Typical of competitive age-group and sub-elite athletes.'
            ),
        }
    else:
        return {
            'label': 'Elite Threshold',
            'css': 'elite',
            'meaning': (
                'Exceptional lactate clearance and buffering capacity. Seen in elite '
                'athletes. Very small gap remains between LT2 and HRmax.'
            ),
        }


# ---------------------------------------------------------------------------
# LT1–LT2 GAP CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_gap(gap_pct: float) -> dict:
    """
    The gap between LT1 and LT2 (%HRmax) is the most important diagnostic metric.
    In the Norwegian model, the goal is to raise LT1 toward LT2 through sub-threshold
    volume, narrowing the gap from below.
    """
    if gap_pct > 20:
        return {
            'label': 'Very Wide',
            'css': 'very-wide',
            'meaning': (
                'Very large moderate zone. LT1 may be underdeveloped, or the athlete '
                'has done lots of threshold work with insufficient easy volume — the '
                'classic "all Zone 3, no Zone 1" pattern. Also common in beginners.'
            ),
        }
    elif gap_pct > 15:
        return {
            'label': 'Wide',
            'css': 'wide',
            'meaning': (
                'Significant threshold separation. Typical early in a training cycle or '
                'in athletes transitioning from shorter-distance backgrounds. Room to '
                'raise LT1 through consistent aerobic volume.'
            ),
        }
    elif gap_pct > 10:
        return {
            'label': 'Moderate',
            'css': 'moderate',
            'meaning': (
                'Normal, healthy separation for trained endurance athletes. Both thresholds '
                'are well-developed. Most well-trained age-group athletes sit here.'
            ),
        }
    elif gap_pct > 5:
        return {
            'label': 'Narrow',
            'css': 'narrow',
            'meaning': (
                'Thresholds are close together. Indicates excellent aerobic development '
                'relative to threshold capacity. Either LT1 is very high, or LT2 may '
                'benefit from further development. Check individual metrics for context.'
            ),
        }
    else:
        return {
            'label': 'Very Narrow',
            'css': 'very-narrow',
            'meaning': (
                'Thresholds nearly converge. Either a well-developed elite aerobic profile, '
                'or a potential ceiling issue (verify HRmax). Could also indicate overtraining '
                'suppressing the gap. Investigate further.'
            ),
        }


# ---------------------------------------------------------------------------
# CEILING HEADROOM CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_headroom(headroom_pct: float) -> dict:
    if headroom_pct > 18:
        return {
            'label': 'Large',
            'css': 'large',
            'meaning': (
                'Significant room above LT2. VO2max development could raise the ceiling, '
                'which tends to pull the thresholds upward over time.'
            ),
        }
    elif headroom_pct > 12:
        return {
            'label': 'Moderate',
            'css': 'moderate',
            'meaning': (
                'Some headroom above LT2. VO2max work may help but threshold and base '
                'development are likely higher priorities at this stage.'
            ),
        }
    elif headroom_pct > 8:
        return {
            'label': 'Small',
            'css': 'small',
            'meaning': (
                'LT2 is already close to the ceiling. Good aerobic capacity utilisation. '
                'Further VO2max work has diminishing returns at this point.'
            ),
        }
    else:
        return {
            'label': 'Very Small',
            'css': 'very-small',
            'meaning': (
                'LT2 is near-maximal. This is an elite pattern OR HRmax may be '
                'underestimated. Verify HRmax from a true maximal effort test.'
            ),
        }


# ---------------------------------------------------------------------------
# LIMITER IDENTIFICATION
# ---------------------------------------------------------------------------

LIMITER_LABELS = {
    'aerobic_base':      'Aerobic Base Development',
    'threshold':         'Threshold Capacity',
    'vo2max':            'Aerobic Ceiling (VO2max)',
    'lactate_clearance': 'Lactate Clearance',
    'maintenance':       'Well-Balanced Profile',
}

_SEVERITY_RANK = {'critical': 2, 'moderate': 1, 'low': 0}


def identify_limiter(lt1_pct: float, lt2_pct: float,
                     gap_pct: float, headroom_pct: float) -> Tuple:
    """
    Identify the primary and secondary physiological limiters from %HRmax metrics.

    Returns ((primary_key, severity), secondary_or_None).

    NOTE: These rules are heuristics derived from population data and coaching
    experience. Individual variation is significant. Always present results as
    'suggests' or 'indicates', never as definitive diagnoses.
    """
    candidates = []

    # ── Rule 1: Low LT1 — almost always the first thing to fix ──────────────
    if lt1_pct < 68:
        candidates.append(('aerobic_base', 'critical'))
    elif lt1_pct < 73:
        candidates.append(('aerobic_base', 'moderate'))

    # ── Rule 2: Wide gap + decent LT2 = classic "too much intensity" pattern ─
    if gap_pct > 18 and lt2_pct >= 82:
        candidates.append(('aerobic_base', 'critical'))
    elif gap_pct > 15 and lt2_pct >= 80:
        candidates.append(('aerobic_base', 'moderate'))

    # ── Rule 3: Low LT2 but reasonable LT1 = threshold work needed ──────────
    if lt2_pct < 80 and lt1_pct >= 68:
        candidates.append(('threshold', 'critical'))
    elif lt2_pct < 84 and lt1_pct >= 72:
        candidates.append(('threshold', 'moderate'))

    # ── Rule 4: Large headroom + good thresholds = ceiling worth raising ─────
    if headroom_pct > 16 and lt2_pct >= 82:
        candidates.append(('vo2max', 'moderate'))
    elif headroom_pct > 20:
        candidates.append(('vo2max', 'moderate'))

    # ── Rule 5: Wide gap + moderate thresholds = lactate clearance ───────────
    if gap_pct > 15 and lt1_pct >= 68 and lt2_pct < 85:
        candidates.append(('lactate_clearance', 'moderate'))

    # ── Rule 6: Everything looks balanced ────────────────────────────────────
    if not candidates:
        candidates.append(('maintenance', 'low'))

    # De-duplicate: keep first-appearance order, escalate to highest severity
    best_sev: Dict[str, str] = {}
    order: List[str] = []
    for key, sev in candidates:
        if key not in best_sev:
            order.append(key)
            best_sev[key] = sev
        elif _SEVERITY_RANK[sev] > _SEVERITY_RANK[best_sev[key]]:
            best_sev[key] = sev

    unique = [(k, best_sev[k]) for k in order]
    primary   = unique[0] if unique else ('maintenance', 'low')
    secondary = unique[1] if len(unique) > 1 else None
    return primary, secondary


# ---------------------------------------------------------------------------
# TRAINING PRESCRIPTIONS
# ---------------------------------------------------------------------------

PRESCRIPTIONS: Dict[str, dict] = {
    'aerobic_base': {
        'title': 'Aerobic Base Development',
        'summary': (
            'LT1 is the primary limiter. The aerobic threshold needs to be raised '
            'through consistent sub-threshold volume. This is the highest-leverage '
            'intervention available.'
        ),
        'training_focus': [
            'Increase total weekly training volume by 10–15% (if recovery allows).',
            '80–85% of training time should be spent BELOW LT1 — DFA α1 > 0.75.',
            'Use real-time DFA α1 monitoring to enforce the Zone 1 ceiling. The '
            'temptation to push into Zone 2 is the main obstacle here.',
            'Long ride/run duration matters more than intensity. Prioritise '
            'extending the long session each week.',
            'Norwegian-style sub-threshold intervals (4×8–12 min at 95–100% of '
            'LT1 power/pace with short rest) can accelerate LT1 development while '
            'keeping intensity truly aerobic.',
            'Minimise time in the moderate zone (LT1–LT2). This zone is fatiguing '
            'but provides poor aerobic stimulus when base is the limiter.',
            'Expect 4–8 weeks of focused base work before retesting shows a '
            'meaningful LT1 shift.',
        ],
        'what_to_monitor': (
            'Retest every 4–6 weeks. Track LT1 %HRmax and the LT1–LT2 gap. '
            'Success = LT1 rising while LT2 stays stable or also rises.'
        ),
        'what_to_avoid': (
            'Excessive threshold/tempo work, frequent race-pace sessions, or '
            'high-intensity intervals. These develop LT2 and VO2max but do not '
            'efficiently raise LT1.'
        ),
    },

    'threshold': {
        'title': 'Threshold Capacity Development',
        'summary': (
            'LT2 is the primary limiter. The athlete has a reasonable aerobic base '
            'but limited ability to sustain higher intensities. Threshold-specific '
            'work will yield the greatest gains.'
        ),
        'training_focus': [
            'Maintain current easy volume — do not reduce base work.',
            'Add 1–2 threshold-focused sessions per week.',
            'Classic Norwegian 4×8 min or 5×6 min intervals at LT2 HR/power '
            '(DFA α1 hovering around 0.50–0.55).',
            'Tempo runs/rides at 85–95% of LT2 power/pace for 20–40 min '
            'continuous efforts.',
            'Sweet spot work on the bike (88–93% FTP) provides threshold stimulus '
            'with manageable fatigue cost.',
            'Progressive overload: start with shorter intervals, build duration '
            'over weeks before increasing intensity.',
            'Race-specific sessions that simulate sustained efforts near LT2.',
        ],
        'what_to_monitor': (
            'Retest every 4–6 weeks. Track LT2 %HRmax and LT2 power/pace. '
            'Success = LT2 rising while LT1 remains stable.'
        ),
        'what_to_avoid': (
            'Reducing easy volume to make room for intensity — the base must be '
            'maintained. Also avoid too much VO2max work. Raise the threshold '
            'before chasing the ceiling.'
        ),
    },

    'vo2max': {
        'title': 'VO2max / Aerobic Ceiling Development',
        'summary': (
            'Both thresholds are well-developed but significant headroom exists '
            'between LT2 and HRmax. Raising the ceiling can create downstream '
            'improvements in both thresholds over time.'
        ),
        'training_focus': [
            'Maintain base volume and threshold work — these are already strengths.',
            'Add 1 VO2max-focused session per week (2 maximum, with full '
            'recovery between sessions).',
            'Classic VO2max intervals: 4–6 × 3–5 min at 95–105% of current '
            'VO2max power/pace (well above LT2), with equal or slightly shorter '
            'recovery.',
            'Hill repeats (bike or run) provide excellent VO2max stimulus with '
            'lower injury risk.',
            'Short-course racing or time trials (5K runs, 10-mile TTs) give '
            'race-specific VO2max stimulus.',
            'Expect a 6–8 week focused block before meaningful ceiling changes appear.',
        ],
        'what_to_monitor': (
            'Retest every 6–8 weeks. Watch for LT2 creeping upward (the "rising '
            'tide" effect). If LT1 drops while doing VO2max work, the intensity '
            'load is too high — back off and rebuild base.'
        ),
        'what_to_avoid': (
            'Neglecting easy volume. VO2max blocks are fatiguing and it is easy '
            'to let overall volume drop, which erodes the aerobic base.'
        ),
    },

    'lactate_clearance': {
        'title': 'Lactate Clearance & Zone Integration',
        'summary': (
            'Both thresholds are moderate but the gap between them is wide. The '
            'athlete needs to develop more efficient lactate processing across the '
            'moderate intensity range. Work both ends.'
        ),
        'training_focus': [
            'This is a "rising tide lifts all boats" situation — develop both '
            'ends of the threshold curve simultaneously.',
            'Priority 1: Continue building aerobic base with high easy volume '
            '(raise LT1 from below).',
            'Priority 2: Add tempo/threshold work 1–2× per week to push LT2 higher.',
            'Norwegian-style sub-threshold intervals are ideal: long efforts just '
            'below LT1 build the aerobic engine while teaching lactate clearance.',
            'Fartlek sessions oscillating between just below and just above LT1 '
            'can effectively develop the moderate zone.',
            'Gradual progression: run a 4–6 week base block first, then introduce '
            'threshold work on top of the volume.',
        ],
        'what_to_monitor': (
            'Track the LT1–LT2 gap over time. Success = gap narrowing because '
            'LT1 is rising, NOT because LT2 is dropping.'
        ),
        'what_to_avoid': (
            'Spending lots of training time inside the gap (Zone 3 / no-man\'s-land). '
            'This intensity is fatiguing but does not efficiently improve either threshold.'
        ),
    },

    'maintenance': {
        'title': 'Profile Well-Balanced — Maintain & Refine',
        'summary': (
            'No single limiter stands out. The athlete\'s profile is well-balanced. '
            'Focus on maintaining current fitness and making targeted improvements '
            'based on specific race goals.'
        ),
        'training_focus': [
            'Continue current training distribution — it is working.',
            'Focus on race-specific preparation: what does the target event demand?',
            'For longer events (half/full Ironman): emphasise durability — can the '
            'athlete hold these thresholds deep into a race?',
            'For shorter events (sprint/Olympic): sharpen VO2max and neuromuscular power.',
            'Consider sport-specific limiters: swim fitness, transitions, nutrition, pacing.',
            'Periodise: use this balanced phase to target small gains in whichever '
            'area has the most race-day impact.',
        ],
        'what_to_monitor': (
            'Retest every 6–8 weeks to ensure no regression. Watch for early signs '
            'of overtraining (DFA α1 baseline suppression, LT1 dropping).'
        ),
        'what_to_avoid': (
            'Chasing marginal gains in one area at the expense of overall balance. '
            'Do not fix what is not broken.'
        ),
    },
}


# ---------------------------------------------------------------------------
# CONTEXTUAL FLAGS
# ---------------------------------------------------------------------------

def check_flags(lt1_hr: float, lt2_hr: Optional[float],
                hrmax: int, lt1_pct: float,
                lt2_pct: Optional[float] = None,
                headroom_pct: Optional[float] = None,
                bike_lt1_hr: Optional[float] = None,
                run_lt1_hr: Optional[float] = None) -> List[dict]:
    """
    Generate contextual flags for a test result.

    Single-discipline flags are always evaluated.
    Cross-discipline flags require both bike_lt1_hr and run_lt1_hr to be provided.
    """
    flags = []

    # ERROR: LT1 >= LT2 — physiologically impossible
    if lt2_hr is not None and lt1_hr >= lt2_hr:
        flags.append({
            'key': 'lt1_higher_than_lt2',
            'severity': 'error',
            'title': 'Testing Error: LT1 ≥ LT2',
            'message': (
                'LT1 heart rate is equal to or higher than LT2. This is physiologically '
                'impossible and indicates a testing error, excessive artifact contamination, '
                'or incorrect threshold detection. Do not use this test for profiling. '
                'Retest when fresh with verified equipment.'
            ),
        })

    # WARNING: LT2 very close to HRmax
    if headroom_pct is not None and headroom_pct < 5:
        flags.append({
            'key': 'lt2_very_close_to_hrmax',
            'severity': 'warning',
            'title': 'LT2 Near HRmax — Verify HRmax',
            'message': (
                f'LT2 is within 5% of the recorded HRmax ({hrmax} bpm). This may indicate '
                'that HRmax is underestimated (was it from a true maximal test?), or the '
                'athlete is exceptionally well-trained. Verify HRmax before drawing conclusions.'
            ),
        })

    # WARNING: LT1 suspiciously high
    if lt1_pct > 85:
        flags.append({
            'key': 'lt1_suspiciously_high',
            'severity': 'warning',
            'title': 'LT1 > 85% HRmax — Verify Inputs',
            'message': (
                f'LT1 is at {lt1_pct:.1f}% of HRmax. This is possible in elite athletes '
                'but uncommon in recreational/age-group athletes. Check for: beta-blocker '
                'use, incorrect HRmax, cardiac drift during test, or DFA α1 artifact '
                'causing delayed threshold detection.'
            ),
        })

    # INFO/WARNING: cross-sport comparison
    if bike_lt1_hr is not None and run_lt1_hr is not None:
        diff = run_lt1_hr - bike_lt1_hr  # signed (positive = run higher = normal)
        abs_diff = abs(diff)

        if abs_diff > 12:
            flags.append({
                'key': 'large_bike_run_discrepancy',
                'severity': 'info',
                'title': f'Large Run/Bike LT1 Difference ({abs_diff:.0f} bpm)',
                'message': (
                    f'LT1 differs by {abs_diff:.0f} bpm between disciplines. A 3–8 bpm '
                    'difference (run higher) is normal due to greater muscle mass recruited '
                    'in running. A larger gap may indicate: running-specific deconditioning, '
                    'DFA α1 running impact artifact, or different testing conditions. Use '
                    'discipline-specific thresholds for training.'
                ),
            })

        if diff < 0:  # run LT1 lower than bike LT1
            flags.append({
                'key': 'run_lt1_lower_than_bike',
                'severity': 'warning',
                'title': 'Running LT1 Lower Than Cycling LT1',
                'message': (
                    'Running LT1 is lower than cycling LT1. Running normally produces '
                    'higher HR at equivalent internal intensity. Possible causes: DFA α1 '
                    'impact artifact (premature threshold detection), chest strap movement '
                    'during running, or genuinely poor running economy. Consider retesting '
                    'on a treadmill with extra strap security, or default to the cycling '
                    'profile for zone setting.'
                ),
            })

    return flags


# ---------------------------------------------------------------------------
# DELTA COMPARISON (historical)
# ---------------------------------------------------------------------------

def _delta_arrow(current: Optional[float],
                 previous: Optional[float],
                 threshold: float = 1.5) -> str:
    """Return ↑, ↓, or → based on change, or '' if comparison not possible."""
    if current is None or previous is None:
        return ''
    diff = current - previous
    if diff > threshold:
        return '↑'
    elif diff < -threshold:
        return '↓'
    return '→'


def build_delta(current_test: dict, previous_test: Optional[dict]) -> dict:
    """
    Build a delta comparison dict for display in the history section.
    Keys: lt1_hr_delta, lt2_hr_delta, lt1_power_delta, lt2_power_delta,
          lt1_pct_delta, lt2_pct_delta — each is {arrow, diff, prev_value}.
    """
    if previous_test is None:
        return {}

    def _d(key: str, fmt: str = '.0f') -> dict:
        cur = current_test.get(key)
        prev = previous_test.get(key)
        if cur is None or prev is None:
            return {}
        diff = cur - prev
        arrow = _delta_arrow(cur, prev)
        diff_str = f'+{diff:{fmt}}' if diff > 0 else f'{diff:{fmt}}'
        return {
            'arrow': arrow,
            'diff': diff_str,
            'prev': f'{prev:{fmt}}',
        }

    return {
        'lt1_hr':    _d('lt1_hr'),
        'lt2_hr':    _d('lt2_hr'),
        'lt1_power': _d('lt1_power'),
        'lt2_power': _d('lt2_power'),
        'lt1_pct':   _d('lt1_pct', '.1f'),
        'lt2_pct':   _d('lt2_pct', '.1f'),
        'prev_date': previous_test.get('test_date', '')[:10],
    }


# ---------------------------------------------------------------------------
# MAIN PROFILE GENERATION
# ---------------------------------------------------------------------------

def generate_profile(hrmax: int,
                     lt1_hr: float,
                     lt2_hr: Optional[float],
                     discipline: str = 'bike',
                     lt1_power: Optional[float] = None,
                     lt2_power: Optional[float] = None,
                     # Cross-sport context (optional, only when both are available)
                     bike_lt1_hr: Optional[float] = None,
                     run_lt1_hr: Optional[float] = None,
                     # History context (optional)
                     athlete_name: Optional[str] = None) -> dict:
    """
    Generate a complete physiological profile from threshold data and HRmax.

    DISCLAIMER: All outputs are data-informed suggestions for the coach to
    interpret. They are not medical advice and should never be presented as such.

    Returns a dict suitable for JSON serialisation and frontend rendering.
    """
    # ── Core %HRmax metrics ────────────────────────────────────────────────
    lt1_pct = round((lt1_hr / hrmax) * 100, 1)
    lt2_pct = round((lt2_hr / hrmax) * 100, 1) if lt2_hr is not None else None

    gap_pct   = round(lt2_pct - lt1_pct, 1) if lt2_pct is not None else None
    gap_hr    = round(lt2_hr - lt1_hr, 1)   if lt2_hr  is not None else None
    gap_power = (round(lt2_power - lt1_power, 1)
                 if (lt1_power is not None and lt2_power is not None) else None)

    headroom_pct = round(100 - lt2_pct, 1) if lt2_pct is not None else None
    headroom_hr  = round(hrmax - lt2_hr, 1) if lt2_hr  is not None else None

    # ── Classifications ────────────────────────────────────────────────────
    lt1_class       = classify_lt1(lt1_pct)
    lt2_class       = classify_lt2(lt2_pct)      if lt2_pct       is not None else None
    gap_class       = classify_gap(gap_pct)       if gap_pct       is not None else None
    headroom_class  = classify_headroom(headroom_pct) if headroom_pct is not None else None

    # ── Flags ──────────────────────────────────────────────────────────────
    flags = check_flags(
        lt1_hr=lt1_hr,
        lt2_hr=lt2_hr,
        hrmax=hrmax,
        lt1_pct=lt1_pct,
        lt2_pct=lt2_pct,
        headroom_pct=headroom_pct,
        bike_lt1_hr=bike_lt1_hr,
        run_lt1_hr=run_lt1_hr,
    )

    has_error = any(f['severity'] == 'error' for f in flags)

    profile: Dict[str, Any] = {
        'hrmax':       hrmax,
        'discipline':  discipline,
        'valid':       not has_error,
        'metrics': {
            'lt1_pct_hrmax':  lt1_pct,
            'lt2_pct_hrmax':  lt2_pct,
            'gap_hr':         gap_hr,
            'gap_pct':        gap_pct,
            'gap_power':      gap_power,
            'headroom_hr':    headroom_hr,
            'headroom_pct':   headroom_pct,
            'lt1_classification':      lt1_class,
            'lt2_classification':      lt2_class,
            'gap_classification':      gap_class,
            'headroom_classification': headroom_class,
        },
        'flags': flags,
        'limiter':              None,
        'secondary_limiter':    None,
        'prescription':         None,
        'secondary_prescription': None,
        'delta':                {},
    }

    if has_error:
        return profile

    # ── Limiter identification ─────────────────────────────────────────────
    if lt2_pct is not None and gap_pct is not None and headroom_pct is not None:
        primary, secondary = identify_limiter(lt1_pct, lt2_pct, gap_pct, headroom_pct)
    else:
        # Only LT1 available — base-only assessment
        faux_l2   = lt1_pct + 15     # conservative assumption
        faux_gap  = 15
        faux_head = 20
        primary, secondary = identify_limiter(lt1_pct, faux_l2, faux_gap, faux_head)
        secondary = None             # secondary requires full data

    profile['limiter'] = {
        'key':      primary[0],
        'label':    LIMITER_LABELS[primary[0]],
        'severity': primary[1],
    }
    profile['prescription'] = PRESCRIPTIONS.get(primary[0])

    if secondary:
        profile['secondary_limiter'] = {
            'key':      secondary[0],
            'label':    LIMITER_LABELS[secondary[0]],
            'severity': secondary[1],
        }
        sec_presc = PRESCRIPTIONS.get(secondary[0], {})
        profile['secondary_prescription'] = {
            'title':   sec_presc.get('title', ''),
            'summary': sec_presc.get('summary', ''),
        }

    # ── Historical delta ───────────────────────────────────────────────────
    if athlete_name:
        prev = get_previous_test(athlete_name, discipline, exclude_latest=False)
        if prev:
            current_snapshot = {
                'lt1_hr': lt1_hr, 'lt2_hr': lt2_hr,
                'lt1_power': lt1_power, 'lt2_power': lt2_power,
                'lt1_pct': lt1_pct, 'lt2_pct': lt2_pct,
            }
            profile['delta'] = build_delta(current_snapshot, prev)

    return profile
