"""
Database Models — SQLAlchemy ORM for the combined DFA Ramp Tool.

Tables:
  users         — Coach and athlete accounts (email + PIN auth)
  test_sessions — Completed ramp test metadata
  test_records  — Per-second power/HR/cadence data points
  rr_intervals  — Raw RR interval data for DFA analysis
"""

import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='athlete')  # 'athlete' or 'coach'
    approved = db.Column(db.Boolean, default=False)
    pin_hash = db.Column(db.String(128), nullable=True)
    pin_expires = db.Column(db.DateTime, nullable=True)
    # Coach-only: password hash for persistent login
    password_hash = db.Column(db.String(128), nullable=True)
    password_must_change = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Athlete profile fields
    sport = db.Column(db.String(10), nullable=True)  # 'bike', 'run', or 'both'
    weight_kg = db.Column(db.Float, nullable=True)
    hrmax_bike = db.Column(db.Integer, nullable=True)
    hrmax_run = db.Column(db.Integer, nullable=True)
    threshold_power = db.Column(db.Integer, nullable=True)
    threshold_pace = db.Column(db.String(10), nullable=True)  # "4:30" format

    test_sessions = db.relationship('TestSession', backref='athlete', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'approved': self.approved,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'sport': self.sport,
            'weight_kg': self.weight_kg,
            'hrmax_bike': self.hrmax_bike,
            'hrmax_run': self.hrmax_run,
            'threshold_power': self.threshold_power,
            'threshold_pace': self.threshold_pace,
            'password_must_change': self.password_must_change or False,
        }


class TestSession(db.Model):
    __tablename__ = 'test_sessions'

    id = db.Column(db.Integer, primary_key=True)
    athlete_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    sport = db.Column(db.String(10), nullable=False)  # 'bike' or 'run'
    test_date = db.Column(db.DateTime, default=datetime.utcnow)
    duration_sec = db.Column(db.Float, nullable=True)
    stages_completed = db.Column(db.Integer, nullable=True)
    peak_power = db.Column(db.Integer, nullable=True)
    peak_hr = db.Column(db.Integer, nullable=True)
    artifact_pct = db.Column(db.Float, nullable=True)
    quality = db.Column(db.String(20), nullable=True)  # 'good', 'moderate', 'poor'

    # Athlete config at time of test
    threshold_value = db.Column(db.Float, nullable=True)  # watts or min/km
    hrmax = db.Column(db.Integer, nullable=True)
    weight_kg = db.Column(db.Float, nullable=True)

    # Early ramp termination metadata (JSON string)
    early_ramp_end = db.Column(db.Text, nullable=True)

    # Analysis results (populated after coach runs analysis)
    hrvt1s_power = db.Column(db.Float, nullable=True)
    hrvt1s_hr = db.Column(db.Float, nullable=True)
    hrvt1c_power = db.Column(db.Float, nullable=True)
    hrvt1c_hr = db.Column(db.Float, nullable=True)
    hrvt2_power = db.Column(db.Float, nullable=True)
    hrvt2_hr = db.Column(db.Float, nullable=True)
    archetype = db.Column(db.String(50), nullable=True)
    analysis_json = db.Column(db.Text, nullable=True)  # Full analysis result as JSON

    # Original FIT file (gzip-compressed bytes)
    fit_file_name = db.Column(db.String(255), nullable=True)
    fit_file_data = db.Column(db.LargeBinary, nullable=True)

    records = db.relationship('TestRecord', backref='session', lazy='dynamic',
                              cascade='all, delete-orphan')
    rr_data = db.relationship('RRInterval', backref='session', lazy='dynamic',
                              cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'athlete_id': self.athlete_id,
            'sport': self.sport,
            'test_date': self.test_date.isoformat() if self.test_date else None,
            'duration_sec': self.duration_sec,
            'stages_completed': self.stages_completed,
            'peak_power': self.peak_power,
            'peak_hr': self.peak_hr,
            'artifact_pct': self.artifact_pct,
            'quality': self.quality,
            'threshold_value': self.threshold_value,
            'hrmax': self.hrmax,
            'weight_kg': self.weight_kg,
            'hrvt1s_power': self.hrvt1s_power,
            'hrvt1s_hr': self.hrvt1s_hr,
            'hrvt1c_power': self.hrvt1c_power,
            'hrvt1c_hr': self.hrvt1c_hr,
            'hrvt2_power': self.hrvt2_power,
            'hrvt2_hr': self.hrvt2_hr,
            'archetype': self.archetype,
        }


class TestRecord(db.Model):
    """Per-second data point during a test."""
    __tablename__ = 'test_records'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('test_sessions.id'), nullable=False, index=True)
    elapsed_sec = db.Column(db.Float, nullable=False)
    power = db.Column(db.Integer, nullable=True)
    heart_rate = db.Column(db.Integer, nullable=True)
    cadence = db.Column(db.Integer, nullable=True)
    target_power = db.Column(db.Integer, nullable=True)
    phase = db.Column(db.String(20), nullable=True)
    stage_num = db.Column(db.Integer, nullable=True)


class RRInterval(db.Model):
    """Individual RR interval from HR monitor."""
    __tablename__ = 'rr_intervals'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('test_sessions.id'), nullable=False, index=True)
    timestamp_ms = db.Column(db.BigInteger, nullable=False)
    rr_ms = db.Column(db.Float, nullable=False)
