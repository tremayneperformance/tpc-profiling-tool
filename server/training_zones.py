"""
Training Zone Prescription Module

Maps athlete profiles (A, B, C) to intensity distribution percentages
and calculates weekly time-at-intensity from volume inputs.

Profile mapping from limiter identification:
  Profile A  ->  aerobic_base or lactate_clearance limiter
  Profile B  ->  threshold or vo2max limiter
  Profile C  ->  maintenance (well-balanced)
"""

from typing import Optional, Dict, List, Tuple


# ---------------------------------------------------------------------------
# INTENSITY DISTRIBUTIONS BY PROFILE
# ---------------------------------------------------------------------------
# Each zone has a (min%, max%) tuple. Midpoint is used for weekly-minutes calc.

PROFILE_A_ZONES: List[Dict] = [
    {'zone': 'Zone 1',         'range': (50, 55)},
    {'zone': 'Targeted LT1',   'range': (18, 22)},
    {'zone': 'Sub-threshold',  'range': (5, 8)},
    {'zone': 'LT2',            'range': (10, 13)},
    {'zone': 'Supra-threshold','range': (3, 5)},
    {'zone': 'VO2max',         'range': (2, 4)},
]

PROFILE_B_ZONES: List[Dict] = [
    {'zone': 'Zone 1-2',       'range': (78, 82)},
    {'zone': 'Sub-threshold',  'range': (4, 6)},
    {'zone': 'LT2',            'range': (8, 12)},
    {'zone': 'Supra-threshold','range': (2, 3)},
    {'zone': 'VO2max',         'range': (2, 3)},
]

PROFILE_C_ZONES: List[Dict] = [
    {'zone': 'Zone 1-2',       'range': (85, 92)},
    {'zone': 'Sub-threshold',  'range': (3, 5)},
    {'zone': 'LT2',            'range': (4, 8)},
    {'zone': 'Supra-threshold','range': (1, 2)},
    {'zone': 'VO2max',         'range': (0, 2)},
]

PROFILES = {
    'A': PROFILE_A_ZONES,
    'B': PROFILE_B_ZONES,
    'C': PROFILE_C_ZONES,
}

PROFILE_DESCRIPTIONS = {
    'A': (
        'Profile A indicates the aerobic base is the primary limiter. '
        'Training distribution emphasises targeted LT1 work alongside a '
        'substantial easy volume foundation to raise the aerobic threshold.'
    ),
    'B': (
        'Profile B reflects a solid aerobic base with threshold capacity as '
        'the main area for development. Distribution follows a polarised '
        'model with the majority of volume in Zone 1-2 and structured '
        'threshold and VO2max sessions.'
    ),
    'C': (
        'Profile C represents a well-balanced physiological profile. '
        'Distribution is heavily aerobic with minimal high-intensity work, '
        'maintaining the existing fitness base while allowing targeted '
        'refinement.'
    ),
}


# ---------------------------------------------------------------------------
# LIMITER -> PROFILE MAPPING
# ---------------------------------------------------------------------------

def get_profile_letter(limiter_key: str) -> str:
    """Map a limiter key from profiling.py to a training profile letter."""
    mapping = {
        'aerobic_base':      'A',
        'lactate_clearance': 'A',
        'threshold':         'B',
        'vo2max':            'B',
        'maintenance':       'C',
    }
    return mapping.get(limiter_key, 'B')


# ---------------------------------------------------------------------------
# VOLUME TIERS
# ---------------------------------------------------------------------------

def get_volume_tier(discipline: str, volume: float) -> str:
    """
    Return 'standard', 'higher', or 'above_range' based on discipline and volume.

    Cycling volume is in hours. Running volume is in km.

    Tiers:
      Cycling:  4-8 hrs = standard  |  8-14 hrs = higher  |  >14 hrs = above_range
      Running:  40-60 km = standard |  60-100 km = higher  |  >100 km = above_range
    """
    if discipline == 'bike':
        if volume <= 8:
            return 'standard'
        elif volume <= 14:
            return 'higher'
        else:
            return 'above_range'
    else:  # run
        if volume <= 60:
            return 'standard'
        elif volume <= 100:
            return 'higher'
        else:
            return 'above_range'


# ---------------------------------------------------------------------------
# COACHING NOTES
# ---------------------------------------------------------------------------

COACHING_NOTES_HIGHER = {
    'A': (
        'At this volume, targeted LT1 work and threshold sessions carry '
        'more weight — ensure recovery between intensity blocks is adequate.'
    ),
    'B': (
        'At this volume, progressive aerobic overload is highly effective. '
        'Keep intensity proportions stable — resist adding more threshold '
        'work as volume grows.'
    ),
    'C': (
        'At this volume, the aerobic base stimulus is substantial. Reduce '
        'threshold and supra-threshold percentages toward the lower end of '
        'the prescribed ranges.'
    ),
}

COACHING_NOTE_ABOVE_RANGE = (
    'Volume entered exceeds typical age-group range — recommendations are '
    'calibrated for up to 14 hrs / 100 km. Apply distributions as a guide only.'
)


def get_coaching_note(profile_letter: str, tier: str) -> Optional[str]:
    """Return the coaching note for a profile/tier combination, or None."""
    if tier == 'standard':
        return None
    elif tier == 'higher':
        return COACHING_NOTES_HIGHER.get(profile_letter)
    elif tier == 'above_range':
        return COACHING_NOTE_ABOVE_RANGE
    return None


# ---------------------------------------------------------------------------
# ZONE TABLE GENERATION
# ---------------------------------------------------------------------------

def get_zone_table(profile_letter: str,
                   weekly_volume: Optional[float] = None,
                   discipline: str = 'bike') -> Dict:
    """
    Build the zone prescription table for a given profile.

    Parameters:
        profile_letter: 'A', 'B', or 'C'
        weekly_volume:  hours (bike) or km (run), or None for percentages only
        discipline:     'bike' or 'run'

    Returns dict with:
        profile:      letter
        description:  profile description text
        zones:        list of zone dicts with keys: zone, min_pct, max_pct, weekly_minutes (or None)
        volume_unit:  'hrs' or 'km'
        coaching_note: string or None
        tier:         'standard', 'higher', 'above_range', or None
    """
    zones_def = PROFILES.get(profile_letter, PROFILE_B_ZONES)
    volume_unit = 'hrs' if discipline == 'bike' else 'km'

    tier = None
    coaching_note = None
    zones = []

    for z in zones_def:
        entry = {
            'zone': z['zone'],
            'min_pct': z['range'][0],
            'max_pct': z['range'][1],
            'weekly_minutes': None,
        }

        if weekly_volume is not None:
            midpoint_pct = (z['range'][0] + z['range'][1]) / 2.0
            if discipline == 'bike':
                total_minutes = weekly_volume * 60
            else:
                # For running, convert km to approximate minutes using 6 min/km
                total_minutes = weekly_volume * 6
            entry['weekly_minutes'] = round(total_minutes * midpoint_pct / 100)

        zones.append(entry)

    if weekly_volume is not None:
        tier = get_volume_tier(discipline, weekly_volume)
        coaching_note = get_coaching_note(profile_letter, tier)

    return {
        'profile': profile_letter,
        'description': PROFILE_DESCRIPTIONS.get(profile_letter, ''),
        'zones': zones,
        'volume_unit': volume_unit,
        'coaching_note': coaching_note,
        'tier': tier,
    }
