"""
Combined DFA Ramp Tool — Flask Application

Routes:
  /                    → Athlete test tool (PWA login + test execution)
  /analysis            → Coach analysis dashboard (protected)
  /api/auth/*          → Authentication endpoints
  /api/test/*          → Test data submission (athlete, authenticated)
  /api/analysis/*      → Analysis endpoints (coach only)
  /api/admin/*         → Athlete management (coach only)
"""

import io
import os
import gzip
import json
import hashlib
from datetime import datetime
from collections import OrderedDict

import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    make_response, send_from_directory, g
)

from models import db, User, TestSession, TestRecord, RRInterval
from auth import (
    generate_pin, set_pin_for_user, validate_pin,
    hash_password, verify_password,
    create_token, decode_token, get_current_user,
    login_required, coach_required,
    send_pin_email,
)

# Analysis modules (from existing DFA Tool)
import ftp_run_profiling
import ramp_analysis
import report_generator
from dfa_core import (
    parse_fit_file, clean_rr_intervals, smoothness_priors_detrend,
    dfa_alpha1, build_windows
)


def create_app():
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')

    # --- Config ---
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(os.path.dirname(__file__), 'ramptool.db')
    )
    # Fix Render/Railway postgres:// → postgresql://
    if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
        app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace(
            'postgres://', 'postgresql://', 1
        )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

    db.init_app(app)

    # In-memory cache for FIT files between analyze and save (max 20 entries)
    fit_cache = OrderedDict()
    FIT_CACHE_MAX = 20

    def cache_fit_file(file_bytes, filename):
        """Store FIT file bytes, return cache key."""
        key = hashlib.sha256(file_bytes).hexdigest()[:16]
        compressed = gzip.compress(file_bytes)
        fit_cache[key] = {'data': compressed, 'name': filename}
        # Evict oldest if over limit
        while len(fit_cache) > FIT_CACHE_MAX:
            fit_cache.popitem(last=False)
        return key

    def pop_fit_file(key):
        """Retrieve and remove FIT file from cache."""
        return fit_cache.pop(key, None)

    with app.app_context():
        db.create_all()
        # Add columns that create_all() won't add to existing tables
        _migrate_add_columns(app)
        _ensure_coach_account(app)
        _seed_dummy_athletes(app)

    # -----------------------------------------------------------------------
    # STATIC FILE SERVING — PWA client files
    # -----------------------------------------------------------------------
    client_dir = os.path.join(os.path.dirname(__file__), '..', 'client')

    @app.route('/')
    def index():
        """Serve the athlete test tool (PWA)."""
        return send_from_directory(client_dir, 'index.html')

    @app.route('/css/<path:filename>')
    def client_css(filename):
        return send_from_directory(os.path.join(client_dir, 'css'), filename)

    @app.route('/js/<path:filename>')
    def client_js(filename):
        return send_from_directory(os.path.join(client_dir, 'js'), filename)

    @app.route('/manifest.json')
    def manifest():
        return send_from_directory(client_dir, 'manifest.json')

    @app.route('/sw.js')
    def service_worker():
        return send_from_directory(client_dir, 'sw.js')

    @app.route('/tpc_logo.png')
    def tpc_logo():
        return send_from_directory(client_dir, 'tpc_logo.png')

    # -----------------------------------------------------------------------
    # COACH ANALYSIS DASHBOARD
    # -----------------------------------------------------------------------

    @app.route('/analysis')
    def analysis_dashboard():
        """Serve the coach-only analysis dashboard."""
        # Auth check is done client-side via JWT — the HTML loads,
        # but all API calls require coach token
        return render_template('analysis.html')

    # -----------------------------------------------------------------------
    # AUTH ROUTES
    # -----------------------------------------------------------------------

    @app.route('/api/auth/athlete-login', methods=['POST'])
    def athlete_login():
        """Athlete logs in with email + password."""
        data = request.get_json(silent=True) or {}
        email = str(data.get('email', '')).strip().lower()
        password = str(data.get('password', ''))

        if not email or not password:
            return jsonify({'status': 'error', 'message': 'Email and password required.'}), 400

        user = User.query.filter_by(email=email).first()

        if not user:
            return jsonify({
                'status': 'error',
                'message': 'This email is not registered. Contact your coach to be added.'
            }), 404

        if not user.approved:
            return jsonify({
                'status': 'error',
                'message': 'Your account is pending approval. Contact your coach.'
            }), 403

        if not verify_password(user, password):
            return jsonify({'status': 'error', 'message': 'Incorrect password.'}), 401

        user.last_login = datetime.utcnow()
        db.session.commit()

        token = create_token(user)
        return jsonify({
            'status': 'ok',
            'token': token,
            'user': user.to_dict(),
        })

    @app.route('/api/auth/set-password', methods=['POST'])
    @login_required
    def set_password():
        """Athlete sets a new password (first login or password change)."""
        data = request.get_json(silent=True) or {}
        new_password = str(data.get('new_password', ''))

        if len(new_password) < 4:
            return jsonify({'status': 'error', 'message': 'Password must be at least 4 characters.'}), 400

        user = g.current_user
        user.password_hash = hash_password(new_password)
        user.password_must_change = False
        db.session.commit()

        return jsonify({
            'status': 'ok',
            'message': 'Password updated.',
            'user': user.to_dict(),
        })

    @app.route('/api/auth/coach-login', methods=['POST'])
    def coach_login():
        """Coach logs in with email + password."""
        data = request.get_json(silent=True) or {}
        email = str(data.get('email', '')).strip().lower()
        password = str(data.get('password', ''))

        if not email or not password:
            return jsonify({'status': 'error', 'message': 'Email and password required.'}), 400

        user = User.query.filter_by(email=email, role='coach').first()
        if not user or not verify_password(user, password):
            return jsonify({'status': 'error', 'message': 'Invalid credentials.'}), 401

        user.last_login = datetime.utcnow()
        db.session.commit()

        token = create_token(user)
        return jsonify({
            'status': 'ok',
            'token': token,
            'user': user.to_dict(),
        })

    @app.route('/api/auth/me', methods=['GET'])
    @login_required
    def auth_me():
        """Return current user info."""
        return jsonify({'status': 'ok', 'user': g.current_user.to_dict()})

    # -----------------------------------------------------------------------
    # ADMIN ROUTES (coach only) — Athlete management
    # -----------------------------------------------------------------------

    @app.route('/api/admin/athletes', methods=['GET'])
    @coach_required
    def list_athletes():
        """List all athletes."""
        athletes = User.query.filter_by(role='athlete').order_by(User.name).all()
        return jsonify({
            'status': 'ok',
            'athletes': [a.to_dict() for a in athletes],
        })

    @app.route('/api/admin/athletes', methods=['POST'])
    @coach_required
    def add_athlete():
        """Add a new athlete to the approved list."""
        data = request.get_json(silent=True) or {}
        email = str(data.get('email', '')).strip().lower()
        name = str(data.get('name', '')).strip()

        if not email or not name:
            return jsonify({'status': 'error', 'message': 'Email and name required.'}), 400

        existing = User.query.filter_by(email=email).first()
        if existing:
            return jsonify({'status': 'error', 'message': 'Email already registered.'}), 409

        user = User(
            email=email,
            name=name,
            role='athlete',
            approved=True,
            password_hash=hash_password('tpc'),
            password_must_change=True,
        )
        # Optional profile fields on creation
        if data.get('sport'):
            user.sport = data['sport']
        if data.get('weight_kg') is not None:
            user.weight_kg = data['weight_kg']
        if data.get('hrmax_bike') is not None:
            user.hrmax_bike = data['hrmax_bike']
        if data.get('hrmax_run') is not None:
            user.hrmax_run = data['hrmax_run']
        if data.get('threshold_power') is not None:
            user.threshold_power = data['threshold_power']
        if data.get('threshold_pace') is not None:
            user.threshold_pace = data['threshold_pace']

        db.session.add(user)
        db.session.commit()

        return jsonify({'status': 'ok', 'athlete': user.to_dict()})

    @app.route('/api/admin/athletes/<int:athlete_id>', methods=['PUT'])
    @coach_required
    def update_athlete(athlete_id):
        """Update athlete details."""
        user = User.query.get(athlete_id)
        if not user or user.role != 'athlete':
            return jsonify({'status': 'error', 'message': 'Athlete not found.'}), 404

        data = request.get_json(silent=True) or {}
        if 'name' in data:
            user.name = data['name']
        if 'email' in data:
            new_email = str(data['email']).strip().lower()
            if new_email and new_email != user.email:
                conflict = User.query.filter_by(email=new_email).first()
                if conflict:
                    return jsonify({'status': 'error', 'message': 'Email already in use.'}), 409
                user.email = new_email
        if 'sport' in data:
            user.sport = data['sport']
        if 'approved' in data:
            user.approved = bool(data['approved'])
        if 'weight_kg' in data:
            user.weight_kg = data['weight_kg']
        if 'hrmax_bike' in data:
            user.hrmax_bike = data['hrmax_bike']
        if 'hrmax_run' in data:
            user.hrmax_run = data['hrmax_run']
        if 'threshold_power' in data:
            user.threshold_power = data['threshold_power']
        if 'threshold_pace' in data:
            user.threshold_pace = data['threshold_pace']
        if data.get('password_reset'):
            user.password_hash = hash_password('tpc')
            user.password_must_change = True

        db.session.commit()
        return jsonify({'status': 'ok', 'athlete': user.to_dict()})

    @app.route('/api/admin/athletes/<int:athlete_id>', methods=['DELETE'])
    @coach_required
    def remove_athlete(athlete_id):
        """Remove an athlete."""
        user = User.query.get(athlete_id)
        if not user or user.role != 'athlete':
            return jsonify({'status': 'error', 'message': 'Athlete not found.'}), 404

        db.session.delete(user)
        db.session.commit()
        return jsonify({'status': 'ok'})

    # -----------------------------------------------------------------------
    # TEST DATA SUBMISSION (athlete, authenticated)
    # -----------------------------------------------------------------------

    @app.route('/api/test/submit', methods=['POST'])
    @login_required
    def submit_test():
        """
        Receive completed test data from PWA.

        Expects JSON with:
          metadata:    { sport, thresholdValue, hrMax, weight }
          records:     [ { elapsed, power, heartRate, cadence, targetPower, phase, stageNum } ]
          rrIntervals: [ { time, rr } ]
          summary:     { duration, peakPower, peakHR, artifactPct, stagesCompleted }
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided.'}), 400

        user = g.current_user
        meta = data.get('metadata', {})
        summary = data.get('summary', {})

        # Determine quality
        artifact_pct = summary.get('artifactPct', 0)
        if artifact_pct < 5:
            quality = 'good'
        elif artifact_pct < 15:
            quality = 'moderate'
        else:
            quality = 'poor'

        # Early ramp termination metadata
        early_ramp_json = None
        if meta.get('earlyRampEnd'):
            early_ramp_json = json.dumps(meta['earlyRampEnd'])

        # Create test session
        session = TestSession(
            athlete_id=user.id,
            sport=meta.get('sport', 'bike'),
            duration_sec=summary.get('duration'),
            stages_completed=summary.get('stagesCompleted'),
            peak_power=summary.get('peakPower'),
            peak_hr=summary.get('peakHR'),
            artifact_pct=artifact_pct,
            quality=quality,
            threshold_value=meta.get('thresholdValue'),
            hrmax=meta.get('hrMax'),
            weight_kg=meta.get('weight'),
            early_ramp_end=early_ramp_json,
        )
        db.session.add(session)
        db.session.flush()  # Get session.id

        # Insert records in bulk
        records_data = data.get('records', [])
        if records_data:
            record_objects = []
            for r in records_data:
                record_objects.append(TestRecord(
                    session_id=session.id,
                    elapsed_sec=r.get('elapsed', 0),
                    power=r.get('power'),
                    heart_rate=r.get('heartRate'),
                    cadence=r.get('cadence'),
                    target_power=r.get('targetPower'),
                    phase=r.get('phase'),
                    stage_num=r.get('stageNum'),
                ))
            db.session.bulk_save_objects(record_objects)

        # Insert RR intervals in bulk
        rr_data = data.get('rrIntervals', [])
        if rr_data:
            rr_objects = []
            for rr in rr_data:
                rr_objects.append(RRInterval(
                    session_id=session.id,
                    timestamp_ms=rr.get('time', 0),
                    rr_ms=rr.get('rr', 0),
                ))
            db.session.bulk_save_objects(rr_objects)

        # Update athlete profile with latest test config
        if meta.get('weight'):
            user.weight_kg = meta['weight']
        if meta.get('hrMax'):
            if meta.get('sport') == 'bike':
                user.hrmax_bike = meta['hrMax']
            else:
                user.hrmax_run = meta['hrMax']
        if meta.get('thresholdValue'):
            if meta.get('sport') == 'bike':
                user.threshold_power = int(meta['thresholdValue'])
            else:
                user.threshold_pace = str(meta['thresholdValue'])

        db.session.commit()

        return jsonify({
            'status': 'ok',
            'message': 'Test data saved successfully.',
            'session_id': session.id,
        })

    @app.route('/api/test/sessions', methods=['GET'])
    @login_required
    def list_my_sessions():
        """List test sessions for the current user."""
        sessions = TestSession.query.filter_by(
            athlete_id=g.current_user.id
        ).order_by(TestSession.test_date.desc()).all()
        return jsonify({
            'status': 'ok',
            'sessions': [s.to_dict() for s in sessions],
        })

    # -----------------------------------------------------------------------
    # ANALYSIS ROUTES (coach only)
    # -----------------------------------------------------------------------

    def _float_or_none(val):
        if val is None or val == '':
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @app.route('/api/analysis/sessions', methods=['GET'])
    @coach_required
    def all_sessions():
        """List all test sessions across all athletes."""
        athlete_id = request.args.get('athlete_id')
        query = TestSession.query
        if athlete_id:
            try:
                query = query.filter_by(athlete_id=int(athlete_id))
            except (ValueError, TypeError):
                return jsonify({'status': 'error', 'message': 'Invalid athlete_id.'}), 400
        sessions = query.order_by(TestSession.test_date.desc()).all()

        results = []
        for s in sessions:
            d = s.to_dict()
            d['athlete_name'] = s.athlete.name
            d['athlete_email'] = s.athlete.email
            d['rr_count'] = RRInterval.query.filter_by(session_id=s.id).count()
            results.append(d)

        return jsonify({'status': 'ok', 'sessions': results})

    @app.route('/api/analysis/session/<int:session_id>/fit', methods=['GET'])
    @coach_required
    def export_session_fit(session_id):
        """Export a stored test session as a FIT file for analysis."""
        session = TestSession.query.get(session_id)
        if not session:
            return jsonify({'status': 'error', 'message': 'Session not found.'}), 404

        # Build FIT-compatible data from stored records
        records = TestRecord.query.filter_by(session_id=session_id).order_by(
            TestRecord.elapsed_sec
        ).all()
        rr_intervals = RRInterval.query.filter_by(session_id=session_id).order_by(
            RRInterval.timestamp_ms
        ).all()

        # Convert to the format expected by ramp_analysis
        fit_data = {
            'rr_ms': [rr.rr_ms for rr in rr_intervals],
            'rr_times': [],
            'heart_rates': [],
            'powers': [],
            'speeds': [],
            'source': 'hrv' if len(rr_intervals) > 30 else 'record',
            'warnings': [],
        }

        # Build rr_times from cumulative RR
        elapsed = 0.0
        for rr in rr_intervals:
            fit_data['rr_times'].append(elapsed)
            elapsed += rr.rr_ms / 1000.0

        # Build HR and power time series from records
        for r in records:
            if r.heart_rate is not None:
                fit_data['heart_rates'].append((r.elapsed_sec, float(r.heart_rate)))
            if r.power is not None:
                fit_data['powers'].append((r.elapsed_sec, float(r.power)))

        return jsonify({'status': 'ok', 'fit_data': fit_data, 'session': session.to_dict()})

    @app.route('/api/analysis/analyze-session/<int:session_id>', methods=['POST'])
    @coach_required
    def analyze_session(session_id):
        """Run full DFA ramp analysis on a stored test session."""
        session_obj = TestSession.query.get(session_id)
        if not session_obj:
            return jsonify({'status': 'error', 'message': 'Session not found.'}), 404

        # Get the stored data
        records = TestRecord.query.filter_by(session_id=session_id).order_by(
            TestRecord.elapsed_sec
        ).all()
        rr_intervals = RRInterval.query.filter_by(session_id=session_id).order_by(
            RRInterval.timestamp_ms
        ).all()

        if not rr_intervals:
            return jsonify({'status': 'error', 'message': 'No RR interval data for this session.'}), 400

        # Build data structures matching what parse_fit_file returns
        rr_ms = [rr.rr_ms for rr in rr_intervals]
        rr_times_raw = []
        elapsed = 0.0
        for rr in rr_intervals:
            rr_times_raw.append(elapsed)
            elapsed += rr.rr_ms / 1000.0

        heart_rates = [(r.elapsed_sec, float(r.heart_rate)) for r in records if r.heart_rate]
        powers = [(r.elapsed_sec, float(r.power)) for r in records if r.power]

        # Clean RR intervals and build windows
        rr_clean, times_clean, artifact_pct = clean_rr_intervals(rr_ms, rr_times_raw)

        rr_orig = np.array(rr_ms)
        rr_fixed = np.array(rr_clean)
        artifact_mask = (np.abs(rr_orig - rr_fixed) > 0.01).tolist()

        windows = build_windows(rr_clean, times_clean, heart_rates, powers,
                                artifact_mask=artifact_mask)

        # Construct dfa_result dict compatible with analyze_ramp_test
        dfa_result = {
            'status': 'ok',
            'parsed': {
                'powers': powers,
                'heart_rates': heart_rates,
                'speeds': [],
                'rr_ms': rr_ms,
                'rr_times': rr_times_raw,
                'warnings': [],
                'source': 'database',
            },
            'rr_clean': rr_clean,
            'rr_times': times_clean,
            'artifact_pct': round(artifact_pct, 2),
            'artifact_mask': artifact_mask,
            'windows': windows,
        }

        # Get optional parameters from request
        body = request.get_json(silent=True) or {}
        sport = body.get('sport', session_obj.sport or 'bike')
        threshold_pace_sec = body.get('threshold_pace_sec')

        # Run the full analysis pipeline
        result = ramp_analysis.analyze_ramp_test(
            dfa_result=dfa_result,
            protocol_type=sport,
            threshold_pace_sec=threshold_pace_sec,
        )

        if result.get('status') == 'ok':
            # Store analysis results on the session
            session_obj.artifact_pct = result.get('artifact_pct')
            thresholds = result.get('thresholds', {})
            session_obj.hrvt1s_power = thresholds.get('hrvt1s_power')
            session_obj.hrvt1s_hr = thresholds.get('hrvt1s_hr')
            session_obj.hrvt1c_power = thresholds.get('hrvt1c_power')
            session_obj.hrvt1c_hr = thresholds.get('hrvt1c_hr')
            session_obj.hrvt2_power = thresholds.get('hrvt2_power')
            session_obj.hrvt2_hr = thresholds.get('hrvt2_hr')
            session_obj.archetype = result.get('archetype', {}).get('label')
            session_obj.analysis_json = json.dumps(result)
            db.session.commit()

        return jsonify({'status': result.get('status', 'error'), 'result': result})

    # -----------------------------------------------------------------------
    # EXISTING ANALYSIS ROUTES (FIT file upload — preserved for coach)
    # -----------------------------------------------------------------------

    @app.route('/api/analysis/analyze-ramp', methods=['POST'])
    @coach_required
    def analyze_ramp():
        """Accept a FIT file upload and run full ramp test analysis."""
        if 'fit_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded.'}), 400

        f = request.files['fit_file']
        fname = f.filename.lower()
        if not (fname.endswith('.fit') or fname.endswith('.fit.gz') or fname.endswith('.gz')):
            return jsonify({'status': 'error', 'message': 'File must be .FIT or .FIT.GZ'}), 400

        file_bytes = f.read()
        if fname.endswith('.gz'):
            try:
                file_bytes = gzip.decompress(file_bytes)
            except Exception:
                return jsonify({'status': 'error', 'message': 'Could not decompress file.'}), 400

        segments_override = None
        seg_json = request.form.get('segments_override', '').strip()
        if seg_json:
            try:
                segments_override = json.loads(seg_json)
            except json.JSONDecodeError:
                pass

        protocol_type = request.form.get('protocol_type', 'bike').strip()
        threshold_pace_sec = None
        pace_raw = request.form.get('threshold_pace_sec', '').strip()
        if pace_raw:
            try:
                threshold_pace_sec = float(pace_raw)
            except ValueError:
                pass

        tte_duration_sec = None
        tte_raw = request.form.get('tte_duration_sec', '').strip()
        if tte_raw:
            try:
                tte_duration_sec = float(tte_raw)
            except ValueError:
                pass

        skip_tte = request.form.get('skip_tte', '').strip().lower() == 'true'

        try:
            result = ramp_analysis.analyze_ramp_test(
                file_bytes, segments_override,
                protocol_type=protocol_type,
                threshold_pace_sec=threshold_pace_sec,
                tte_duration_sec=-1 if skip_tte else tte_duration_sec,
            )
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Analysis failed: {e}'}), 400

        return jsonify(result)

    @app.route('/api/analysis/analyze-ftp', methods=['POST'])
    @coach_required
    def analyze_ftp_test():
        """Accept FIT file for FTP/run test profiling."""
        if 'fit_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded.'}), 400

        f = request.files['fit_file']
        file_bytes = f.read()
        fname = f.filename.lower()
        if fname.endswith('.gz'):
            try:
                file_bytes = gzip.decompress(file_bytes)
            except Exception:
                return jsonify({'status': 'error', 'message': 'Could not decompress file.'}), 400

        try:
            result = ftp_run_profiling.analyze_fit_file(file_bytes)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Could not parse FIT file: {e}'}), 400

        return jsonify(result)

    @app.route('/api/analysis/calculate-ftp', methods=['POST'])
    @coach_required
    def calculate_ftp_profile():
        """Calculate FTP/run profile from manual or auto-detected data."""
        data = request.get_json(silent=True) or {}
        sport = data.get('sport', 'cycling')

        try:
            body_weight = float(data.get('body_weight_kg', 0))
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'body_weight_kg is required.'}), 400

        if body_weight <= 0:
            return jsonify({'status': 'error', 'message': 'body_weight_kg must be positive.'}), 400

        known_hrmax = None
        if data.get('known_hrmax'):
            try:
                known_hrmax = float(data['known_hrmax'])
            except (ValueError, TypeError):
                pass

        if sport == 'cycling':
            try:
                five_min_power = float(data['five_min_power'])
                twenty_min_power = float(data['twenty_min_power'])
            except (KeyError, ValueError, TypeError):
                return jsonify({'status': 'error', 'message': '5-min and 20-min power required.'}), 400

            result = ftp_run_profiling.calculate_cycling_profile(
                five_min_power=five_min_power,
                twenty_min_power=twenty_min_power,
                body_weight_kg=body_weight,
                five_min_hr=_float_or_none(data.get('five_min_hr')),
                five_min_max_hr=_float_or_none(data.get('five_min_max_hr')),
                five_min_cadence=_float_or_none(data.get('five_min_cadence')),
                twenty_min_hr=_float_or_none(data.get('twenty_min_hr')),
                twenty_min_max_hr=_float_or_none(data.get('twenty_min_max_hr')),
                twenty_min_cadence=_float_or_none(data.get('twenty_min_cadence')),
                known_hrmax=known_hrmax,
                hr_drift_pct=_float_or_none(data.get('hr_drift_pct')),
                power_cv=_float_or_none(data.get('power_cv')),
                power_trend_5min=_float_or_none(data.get('power_trend_5min')),
                first_quarter_power_20min=_float_or_none(data.get('first_quarter_power_20min')),
                last_quarter_power_20min=_float_or_none(data.get('last_quarter_power_20min')),
                last_2min_avg_power_20min=_float_or_none(data.get('last_2min_avg_power_20min')),
            )
        else:
            try:
                time_1000 = float(data['time_1000_seconds'])
                time_3000 = float(data['time_3000_seconds'])
            except (KeyError, ValueError, TypeError):
                return jsonify({'status': 'error', 'message': '1000m and 3000m times required.'}), 400

            result = ftp_run_profiling.calculate_running_profile(
                time_1000_seconds=time_1000,
                time_3000_seconds=time_3000,
                body_weight_kg=body_weight,
                one_km_hr=_float_or_none(data.get('one_km_hr')),
                one_km_max_hr=_float_or_none(data.get('one_km_max_hr')),
                three_km_hr=_float_or_none(data.get('three_km_hr')),
                three_km_max_hr=_float_or_none(data.get('three_km_max_hr')),
                known_hrmax=known_hrmax,
            )

        return jsonify({'status': 'ok', **result})

    @app.route('/api/analysis/save-ramp', methods=['POST'])
    @coach_required
    def save_ramp_test():
        """Save a ramp test to athlete history (legacy file-based)."""
        data = request.get_json(silent=True) or {}
        athlete_name = str(data.get('athlete_name', '')).strip()
        if not athlete_name:
            return jsonify({'status': 'error', 'message': 'athlete_name required.'}), 400

        result = data.get('result', {})
        weight_kg = data.get('weight_kg')
        if weight_kg is not None:
            result['weight_kg'] = weight_kg
        if data.get('hrmax_bike') is not None:
            result['hrmax_bike'] = data['hrmax_bike']
        if data.get('hrmax_run') is not None:
            result['hrmax_run'] = data['hrmax_run']
        if data.get('threshold_pace') is not None:
            result['threshold_pace'] = data['threshold_pace']

        ok = ramp_analysis.save_ramp_test_result(athlete_name, result)
        if ok:
            return jsonify({'status': 'ok', 'message': f'Ramp test saved for {athlete_name}.'})
        return jsonify({'status': 'error', 'message': 'Could not write history file.'}), 500

    @app.route('/api/analysis/save-ftp', methods=['POST'])
    @coach_required
    def save_ftp_test():
        """Save FTP test to athlete history."""
        data = request.get_json(silent=True) or {}
        athlete_name = str(data.get('athlete_name', '')).strip()
        if not athlete_name:
            return jsonify({'status': 'error', 'message': 'athlete_name required.'}), 400
        result = data.get('result', {})
        ok = ftp_run_profiling.save_ftp_test_result(athlete_name, result)
        if ok:
            return jsonify({'status': 'ok', 'message': f'Test saved for {athlete_name}.'})
        return jsonify({'status': 'error', 'message': 'Could not write history file.'}), 500

    @app.route('/api/analysis/ramp-history', methods=['GET'])
    @coach_required
    def ramp_test_history():
        athlete = request.args.get('athlete', '').strip()
        if not athlete:
            return jsonify({'status': 'error', 'message': 'athlete required.'}), 400
        tests = ramp_analysis.get_ramp_test_history(athlete)
        for t in tests:
            has_fit = bool(t.get('fit_file_data'))
            t.pop('fit_file_data', None)
            t.pop('fit_file_name', None)
            t['has_fit_file'] = has_fit
        return jsonify({'status': 'ok', 'athlete': athlete, 'tests': tests})

    @app.route('/api/analysis/ftp-history', methods=['GET'])
    @coach_required
    def ftp_test_history():
        athlete = request.args.get('athlete', '').strip()
        sport = request.args.get('sport', '').strip() or None
        if not athlete:
            return jsonify({'status': 'error', 'message': 'athlete required.'}), 400
        tests = ftp_run_profiling.get_ftp_test_history(athlete, sport)
        return jsonify({'status': 'ok', 'athlete': athlete, 'tests': tests})

    @app.route('/api/analysis/athlete-list', methods=['GET'])
    @coach_required
    def athlete_list():
        history = ramp_analysis._load_history()
        names = sorted(history.get('athletes', {}).keys())
        return jsonify({'status': 'ok', 'athletes': names})

    @app.route('/api/analysis/athlete-profile', methods=['GET'])
    @coach_required
    def athlete_profile():
        name = request.args.get('name', '').strip()
        if not name:
            return jsonify({'status': 'error', 'message': 'name required.'}), 400
        history = ramp_analysis._load_history()
        athlete = history.get('athletes', {}).get(name, {})
        tests = athlete.get('ramp_tests', [])
        if not tests:
            return jsonify({'status': 'ok', 'profile': {}})
        sorted_tests = sorted(tests, key=lambda t: t.get('test_date', ''), reverse=True)
        profile = {}
        for t in sorted_tests:
            if 'hrmax_bike' not in profile and t.get('hrmax_bike'):
                profile['hrmax_bike'] = t['hrmax_bike']
            if 'hrmax_run' not in profile and t.get('hrmax_run'):
                profile['hrmax_run'] = t['hrmax_run']
            if 'weight_kg' not in profile and t.get('weight_kg'):
                profile['weight_kg'] = t['weight_kg']
            if 'threshold_pace' not in profile and t.get('threshold_pace'):
                profile['threshold_pace'] = t['threshold_pace']
            if all(k in profile for k in ('hrmax_bike', 'hrmax_run', 'weight_kg', 'threshold_pace')):
                break
        return jsonify({'status': 'ok', 'profile': profile})

    @app.route('/api/analysis/all-ramp-tests', methods=['GET'])
    @coach_required
    def all_ramp_tests():
        tests = ramp_analysis.get_all_ramp_tests()
        for t in tests:
            t.pop('fit_file_data', None)
            t.pop('fit_file_name', None)
        return jsonify({'status': 'ok', 'tests': tests})

    @app.route('/api/analysis/generate-report', methods=['POST'])
    @coach_required
    def generate_report():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'status': 'error', 'message': 'No data provided.'}), 400

            result = data.get('result', {})
            athlete_name = data.get('athlete_name', 'Athlete')
            sport = data.get('sport', 'bike')
            comments = data.get('comments', '')
            previous_test = data.get('previous_test')
            weight_kg = _float_or_none(data.get('weight_kg'))
            hrmax = _float_or_none(data.get('hrmax'))

            pdf_bytes = report_generator.generate_ramp_report(
                result_data=result,
                athlete_name=athlete_name,
                sport=sport,
                coach_comments=comments,
                previous_test=previous_test,
                weight_kg=weight_kg,
                hrmax=hrmax,
            )

            response = make_response(pdf_bytes)
            safe_name = athlete_name.replace(' ', '_')
            filename = f'{safe_name}_{sport}_report_{datetime.now().strftime("%Y%m%d")}.pdf'
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response

        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Report generation failed: {e}'}), 500

    # -----------------------------------------------------------------------
    # ADMIN DATA BROWSER (coach only)
    # -----------------------------------------------------------------------

    @app.route('/admin')
    def admin_dashboard():
        """Serve the admin data browser."""
        return render_template('admin.html')

    @app.route('/api/admin/all-users', methods=['GET'])
    @coach_required
    def all_users():
        """List all users (athletes and coaches)."""
        users = User.query.order_by(User.role, User.name).all()
        return jsonify({'status': 'ok', 'users': [u.to_dict() for u in users]})

    @app.route('/api/admin/all-sessions', methods=['GET'])
    @coach_required
    def all_sessions_admin():
        """List all test sessions with athlete info."""
        sessions = TestSession.query.order_by(TestSession.test_date.desc()).all()
        results = []
        for s in sessions:
            d = s.to_dict()
            d['athlete_name'] = s.athlete.name if s.athlete else 'Unknown'
            d['athlete_email'] = s.athlete.email if s.athlete else ''
            d['record_count'] = TestRecord.query.filter_by(session_id=s.id).count()
            d['rr_count'] = RRInterval.query.filter_by(session_id=s.id).count()
            results.append(d)
        return jsonify({'status': 'ok', 'sessions': results})

    @app.route('/api/admin/session/<int:session_id>/detail', methods=['GET'])
    @coach_required
    def session_detail(session_id):
        """Get full detail for a single session including record and RR counts."""
        session = TestSession.query.get(session_id)
        if not session:
            return jsonify({'status': 'error', 'message': 'Session not found.'}), 404
        d = session.to_dict()
        d['athlete_name'] = session.athlete.name if session.athlete else 'Unknown'
        d['athlete_email'] = session.athlete.email if session.athlete else ''
        d['record_count'] = TestRecord.query.filter_by(session_id=session_id).count()
        d['rr_count'] = RRInterval.query.filter_by(session_id=session_id).count()
        d['early_ramp_end'] = session.early_ramp_end
        d['analysis_json'] = session.analysis_json
        # First 50 records as sample
        records = TestRecord.query.filter_by(session_id=session_id).order_by(
            TestRecord.elapsed_sec).limit(50).all()
        d['sample_records'] = [{
            'elapsed_sec': r.elapsed_sec,
            'power': r.power,
            'heart_rate': r.heart_rate,
            'cadence': r.cadence,
            'phase': r.phase,
            'stage_num': r.stage_num,
        } for r in records]
        return jsonify({'status': 'ok', 'session': d})

    @app.route('/api/admin/stats', methods=['GET'])
    @coach_required
    def admin_stats():
        """Dashboard summary stats."""
        total_users = User.query.count()
        total_athletes = User.query.filter_by(role='athlete').count()
        approved_athletes = User.query.filter_by(role='athlete', approved=True).count()
        total_sessions = TestSession.query.count()
        total_records = TestRecord.query.count()
        total_rr = RRInterval.query.count()
        return jsonify({
            'status': 'ok',
            'stats': {
                'total_users': total_users,
                'total_athletes': total_athletes,
                'approved_athletes': approved_athletes,
                'total_sessions': total_sessions,
                'total_records': total_records,
                'total_rr_intervals': total_rr,
            }
        })

    @app.route('/api/admin/fit-download/<athlete_name>/<int:test_index>', methods=['GET'])
    @coach_required
    def download_fit_file(athlete_name, test_index):
        """Download a stored FIT file for a saved ramp test."""
        tests = ramp_analysis.get_ramp_test_history(athlete_name)
        if test_index < 0 or test_index >= len(tests):
            return jsonify({'status': 'error', 'message': 'Test not found.'}), 404
        test = tests[test_index]
        fit_hex = test.get('fit_file_data')
        if not fit_hex:
            return jsonify({'status': 'error', 'message': 'No FIT file stored for this test.'}), 404
        try:
            fit_compressed = bytes.fromhex(fit_hex)
            fit_bytes = gzip.decompress(fit_compressed)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Stored FIT file is corrupt.'}), 400
        filename = test.get('fit_file_name', f'{athlete_name}_ramp.fit')
        if not filename.lower().endswith('.fit'):
            filename = filename + '.fit'
        response = make_response(fit_bytes)
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    @app.route('/api/admin/reanalyze/<athlete_name>/<int:test_index>', methods=['POST'])
    @coach_required
    def reanalyze_test(athlete_name, test_index):
        """Re-run ramp analysis on a stored FIT file."""
        tests = ramp_analysis.get_ramp_test_history(athlete_name)
        if test_index < 0 or test_index >= len(tests):
            return jsonify({'status': 'error', 'message': 'Test not found.'}), 404
        test = tests[test_index]
        fit_hex = test.get('fit_file_data')
        if not fit_hex:
            return jsonify({'status': 'error', 'message': 'No FIT file stored for this test.'}), 404
        try:
            fit_compressed = bytes.fromhex(fit_hex)
            file_bytes = gzip.decompress(fit_compressed)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Stored FIT file is corrupt.'}), 400

        protocol_type = test.get('protocol_type', 'bike')
        threshold_pace_sec = _float_or_none(test.get('threshold_pace_sec'))
        try:
            result = ramp_analysis.analyze_ramp_test(
                file_bytes, None,
                protocol_type=protocol_type,
                threshold_pace_sec=threshold_pace_sec,
            )
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Re-analysis failed: {e}'}), 400

        # Cache the FIT file again for potential re-save
        fit_key = cache_fit_file(file_bytes, test.get('fit_file_name', 'reanalysis.fit'))
        result['fit_cache_key'] = fit_key

        return jsonify(result)

    # -----------------------------------------------------------------------
    # PRIMARY ROUTES — used by analysis dashboard templates.
    # All require coach auth.
    # -----------------------------------------------------------------------

    @app.route('/athlete_list', methods=['GET'])
    @coach_required
    def legacy_athlete_list():
        history = ramp_analysis._load_history()
        names = sorted(history.get('athletes', {}).keys())
        return jsonify({'status': 'ok', 'athletes': names})

    @app.route('/athlete_profile', methods=['GET'])
    @coach_required
    def legacy_athlete_profile():
        name = request.args.get('name', '').strip()
        if not name:
            return jsonify({'status': 'ok', 'profile': {}})
        history = ramp_analysis._load_history()
        athlete = history.get('athletes', {}).get(name, {})
        tests = athlete.get('ramp_tests', [])
        if not tests:
            return jsonify({'status': 'ok', 'profile': {}})
        sorted_tests = sorted(tests, key=lambda t: t.get('test_date', ''), reverse=True)
        profile = {}
        for t in sorted_tests:
            for k in ('hrmax_bike', 'hrmax_run', 'weight_kg', 'threshold_pace'):
                if k not in profile and t.get(k):
                    profile[k] = t[k]
        return jsonify({'status': 'ok', 'profile': profile})

    @app.route('/analyze_ramp', methods=['POST'])
    @coach_required
    def legacy_analyze_ramp():
        if 'fit_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded.'}), 400
        f = request.files['fit_file']
        original_filename = f.filename
        file_bytes = f.read()
        fname = f.filename.lower()
        if fname.endswith('.gz'):
            try:
                file_bytes = gzip.decompress(file_bytes)
            except Exception:
                return jsonify({'status': 'error', 'message': 'Could not decompress file.'}), 400
        segments_override = None
        seg_json = request.form.get('segments_override', '').strip()
        if seg_json:
            try:
                segments_override = json.loads(seg_json)
            except json.JSONDecodeError:
                pass
        protocol_type = request.form.get('protocol_type', 'bike').strip()
        threshold_pace_sec = _float_or_none(request.form.get('threshold_pace_sec', '').strip())
        tte_duration_sec = _float_or_none(request.form.get('tte_duration_sec', '').strip())
        skip_tte = request.form.get('skip_tte', '').strip().lower() == 'true'
        try:
            result = ramp_analysis.analyze_ramp_test(
                file_bytes, segments_override,
                protocol_type=protocol_type,
                threshold_pace_sec=threshold_pace_sec,
                tte_duration_sec=-1 if skip_tte else tte_duration_sec,
            )
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Analysis failed: {e}'}), 400

        # Cache FIT file for later save
        fit_key = cache_fit_file(file_bytes, original_filename)
        result['fit_cache_key'] = fit_key

        return jsonify(result)

    @app.route('/analyze_ftp_test', methods=['POST'])
    @coach_required
    def legacy_analyze_ftp():
        if 'fit_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded.'}), 400
        f = request.files['fit_file']
        file_bytes = f.read()
        fname = f.filename.lower()
        if fname.endswith('.gz'):
            try:
                file_bytes = gzip.decompress(file_bytes)
            except Exception:
                return jsonify({'status': 'error', 'message': 'Could not decompress.'}), 400
        try:
            result = ftp_run_profiling.analyze_fit_file(file_bytes)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Parse failed: {e}'}), 400
        return jsonify(result)

    @app.route('/calculate_ftp_profile', methods=['POST'])
    @coach_required
    def legacy_calculate_ftp():
        data = request.get_json(silent=True) or {}
        sport = data.get('sport', 'cycling')
        try:
            body_weight = float(data.get('body_weight_kg', 0))
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'body_weight_kg required.'}), 400
        if body_weight <= 0:
            return jsonify({'status': 'error', 'message': 'body_weight_kg must be positive.'}), 400
        known_hrmax = _float_or_none(data.get('known_hrmax'))
        if sport == 'cycling':
            try:
                result = ftp_run_profiling.calculate_cycling_profile(
                    five_min_power=float(data['five_min_power']),
                    twenty_min_power=float(data['twenty_min_power']),
                    body_weight_kg=body_weight,
                    five_min_hr=_float_or_none(data.get('five_min_hr')),
                    five_min_max_hr=_float_or_none(data.get('five_min_max_hr')),
                    five_min_cadence=_float_or_none(data.get('five_min_cadence')),
                    twenty_min_hr=_float_or_none(data.get('twenty_min_hr')),
                    twenty_min_max_hr=_float_or_none(data.get('twenty_min_max_hr')),
                    twenty_min_cadence=_float_or_none(data.get('twenty_min_cadence')),
                    known_hrmax=known_hrmax,
                    hr_drift_pct=_float_or_none(data.get('hr_drift_pct')),
                    power_cv=_float_or_none(data.get('power_cv')),
                    power_trend_5min=_float_or_none(data.get('power_trend_5min')),
                    first_quarter_power_20min=_float_or_none(data.get('first_quarter_power_20min')),
                    last_quarter_power_20min=_float_or_none(data.get('last_quarter_power_20min')),
                    last_2min_avg_power_20min=_float_or_none(data.get('last_2min_avg_power_20min')),
                )
            except (KeyError, ValueError, TypeError):
                return jsonify({'status': 'error', 'message': '5-min and 20-min power required.'}), 400
        else:
            try:
                result = ftp_run_profiling.calculate_running_profile(
                    time_1000_seconds=float(data['time_1000_seconds']),
                    time_3000_seconds=float(data['time_3000_seconds']),
                    body_weight_kg=body_weight,
                    one_km_hr=_float_or_none(data.get('one_km_hr')),
                    one_km_max_hr=_float_or_none(data.get('one_km_max_hr')),
                    three_km_hr=_float_or_none(data.get('three_km_hr')),
                    three_km_max_hr=_float_or_none(data.get('three_km_max_hr')),
                    known_hrmax=known_hrmax,
                )
            except (KeyError, ValueError, TypeError):
                return jsonify({'status': 'error', 'message': '1000m and 3000m times required.'}), 400
        return jsonify({'status': 'ok', **result})

    @app.route('/save_ramp_test', methods=['POST'])
    @coach_required
    def legacy_save_ramp():
        data = request.get_json(silent=True) or {}
        athlete_name = str(data.get('athlete_name', '')).strip()
        if not athlete_name:
            return jsonify({'status': 'error', 'message': 'athlete_name required.'}), 400
        result = data.get('result', {})
        for k in ('weight_kg', 'hrmax_bike', 'hrmax_run', 'threshold_pace'):
            if data.get(k) is not None:
                result[k] = data[k]

        # Persist the FIT file from cache if available
        fit_key = data.get('fit_cache_key') or result.get('fit_cache_key')
        fit_entry = pop_fit_file(fit_key) if fit_key else None
        if fit_entry:
            result['fit_file_data'] = fit_entry['data'].hex()  # Store as hex in JSON
            result['fit_file_name'] = fit_entry['name']

        ok = ramp_analysis.save_ramp_test_result(athlete_name, result)
        if ok:
            return jsonify({'status': 'ok', 'message': f'Ramp test saved for {athlete_name}.'})
        return jsonify({'status': 'error', 'message': 'Could not write history file.'}), 500

    @app.route('/save_ftp_test', methods=['POST'])
    @coach_required
    def legacy_save_ftp():
        data = request.get_json(silent=True) or {}
        athlete_name = str(data.get('athlete_name', '')).strip()
        if not athlete_name:
            return jsonify({'status': 'error', 'message': 'athlete_name required.'}), 400
        ok = ftp_run_profiling.save_ftp_test_result(athlete_name, data.get('result', {}))
        if ok:
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Could not write history file.'}), 500

    @app.route('/ramp_test_history', methods=['GET'])
    @coach_required
    def legacy_ramp_history():
        athlete = request.args.get('athlete', '').strip()
        if not athlete:
            return jsonify({'status': 'error', 'message': 'athlete required.'}), 400
        tests = ramp_analysis.get_ramp_test_history(athlete)
        # Strip large FIT blob from response, replace with boolean flag
        for t in tests:
            has_fit = bool(t.get('fit_file_data'))
            t.pop('fit_file_data', None)
            t.pop('fit_file_name', None)
            t['has_fit_file'] = has_fit
        return jsonify({'status': 'ok', 'athlete': athlete, 'tests': tests})

    @app.route('/ftp_test_history', methods=['GET'])
    @coach_required
    def legacy_ftp_history():
        athlete = request.args.get('athlete', '').strip()
        sport = request.args.get('sport', '').strip() or None
        if not athlete:
            return jsonify({'status': 'error', 'message': 'athlete required.'}), 400
        tests = ftp_run_profiling.get_ftp_test_history(athlete, sport)
        return jsonify({'status': 'ok', 'athlete': athlete, 'tests': tests})

    @app.route('/delete_ramp_test', methods=['POST'])
    @coach_required
    def legacy_delete_ramp():
        data = request.get_json(silent=True) or {}
        athlete_name = str(data.get('athlete_name', '')).strip()
        if not athlete_name:
            return jsonify({'status': 'error', 'message': 'athlete_name required.'}), 400
        try:
            test_index = int(data['test_index'])
        except (KeyError, ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'test_index required.'}), 400
        ok = ramp_analysis.delete_ramp_test_from_history(athlete_name, test_index)
        if ok:
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Could not delete test.'}), 400

    @app.route('/delete_ftp_test', methods=['POST'])
    @coach_required
    def legacy_delete_ftp():
        data = request.get_json(silent=True) or {}
        athlete_name = str(data.get('athlete_name', '')).strip()
        if not athlete_name:
            return jsonify({'status': 'error', 'message': 'athlete_name required.'}), 400
        try:
            test_index = int(data['test_index'])
        except (KeyError, ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'test_index required.'}), 400
        ok = ftp_run_profiling.delete_ftp_test_from_history(athlete_name, test_index)
        if ok:
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Could not delete test.'}), 400

    @app.route('/all_ramp_tests', methods=['GET'])
    @coach_required
    def legacy_all_ramp():
        tests = ramp_analysis.get_all_ramp_tests()
        for t in tests:
            t.pop('fit_file_data', None)
            t.pop('fit_file_name', None)
        return jsonify({'status': 'ok', 'tests': tests})

    @app.route('/generate_report', methods=['POST'])
    @coach_required
    def legacy_generate_report():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'status': 'error', 'message': 'No data provided.'}), 400
            pdf_bytes = report_generator.generate_ramp_report(
                result_data=data.get('result', {}),
                athlete_name=data.get('athlete_name', 'Athlete'),
                sport=data.get('sport', 'bike'),
                coach_comments=data.get('comments', ''),
                previous_test=data.get('previous_test'),
                weight_kg=_float_or_none(data.get('weight_kg')),
                hrmax=_float_or_none(data.get('hrmax')),
            )
            response = make_response(pdf_bytes)
            safe_name = data.get('athlete_name', 'Athlete').replace(' ', '_')
            filename = f'{safe_name}_report_{datetime.now().strftime("%Y%m%d")}.pdf'
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Report failed: {e}'}), 500

    @app.route('/ftp_population_averages', methods=['GET'])
    @coach_required
    def legacy_ftp_pop():
        sport = request.args.get('sport', '').strip() or None
        averages = ftp_run_profiling.get_ftp_population_averages(sport)
        return jsonify({'status': 'ok', **averages})

    @app.route('/ramp_test_full_result', methods=['GET'])
    @coach_required
    def ramp_test_full_result():
        """Return a full saved analysis result by result_id."""
        result_id = request.args.get('result_id', '').strip()
        if not result_id:
            return jsonify({'status': 'error', 'message': 'result_id required.'}), 400
        result = ramp_analysis.load_full_result(result_id)
        if result is None:
            return jsonify({'status': 'error', 'message': 'Result not found.'}), 404
        return jsonify({'status': 'ok', 'result': result})

    @app.route('/heartbeat', methods=['POST'])
    def legacy_heartbeat():
        return jsonify({'status': 'ok'})

    @app.route('/shutdown', methods=['POST'])
    def legacy_shutdown():
        return jsonify({'status': 'ok'})

    return app


# ---------------------------------------------------------------------------
# DATABASE MIGRATION — add columns that create_all() won't add to existing tables
# ---------------------------------------------------------------------------

def _migrate_add_columns(app):
    """Safely add new columns to existing tables (idempotent)."""
    from sqlalchemy import text, inspect
    insp = inspect(db.engine)

    # Users: sport column
    if 'users' in insp.get_table_names():
        existing_user_cols = {c['name'] for c in insp.get_columns('users')}
        with db.engine.begin() as conn:
            if 'sport' not in existing_user_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN sport VARCHAR(10)"))
                print('[MIGRATE] Added sport to users')

    # TestSession: fit_file_name, fit_file_data
    if 'test_sessions' in insp.get_table_names():
        existing = {c['name'] for c in insp.get_columns('test_sessions')}
        with db.engine.begin() as conn:
            if 'fit_file_name' not in existing:
                conn.execute(text('ALTER TABLE test_sessions ADD COLUMN fit_file_name VARCHAR(255)'))
                print('[MIGRATE] Added fit_file_name to test_sessions')
            if 'fit_file_data' not in existing:
                # Use BLOB for SQLite, BYTEA for PostgreSQL
                col_type = 'BYTEA' if 'postgresql' in str(db.engine.url) else 'BLOB'
                conn.execute(text(f'ALTER TABLE test_sessions ADD COLUMN fit_file_data {col_type}'))
                print('[MIGRATE] Added fit_file_data to test_sessions')


# ---------------------------------------------------------------------------
# COACH ACCOUNT BOOTSTRAP
# ---------------------------------------------------------------------------

def _ensure_coach_account(app):
    """Create the coach account on first run if it doesn't exist."""
    coach_email = os.environ.get('COACH_EMAIL', 'coach@tremayneperformance.com')
    coach_password = os.environ.get('COACH_PASSWORD', 'changeme')
    coach_name = os.environ.get('COACH_NAME', 'Kyle Tremayne')

    existing = User.query.filter_by(email=coach_email, role='coach').first()
    if not existing:
        coach = User(
            email=coach_email,
            name=coach_name,
            role='coach',
            approved=True,
            password_hash=hash_password(coach_password),
        )
        db.session.add(coach)
        db.session.commit()
        print(f'[SETUP] Coach account created: {coach_email}')


def _seed_dummy_athletes(app):
    """Create dummy test athletes on first run if they don't exist."""
    dummy_users = [
        {'name': 'Alex Demo', 'email': 'alex@demo.tpc', 'sport': 'both',
         'threshold_power': 220, 'threshold_pace': '4:30',
         'hrmax_bike': 185, 'hrmax_run': 187, 'weight_kg': 72},
        {'name': 'Jordan Demo', 'email': 'jordan@demo.tpc', 'sport': 'bike',
         'threshold_power': 280, 'hrmax_bike': 190, 'weight_kg': 78},
        {'name': 'Sam Demo', 'email': 'sam@demo.tpc', 'sport': 'bike',
         'threshold_power': 200, 'hrmax_bike': 178, 'weight_kg': 64.5},
    ]
    created = 0
    for u in dummy_users:
        if User.query.filter_by(email=u['email']).first():
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
        created += 1
    if created:
        db.session.commit()
        print(f'[SETUP] Created {created} dummy athlete account(s)')


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=True, host='0.0.0.0', port=port)
