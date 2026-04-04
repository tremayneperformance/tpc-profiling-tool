"""
Authentication Module — Email + PIN for athletes, password for coach.

Flow:
  1. Athlete enters email at /test
  2. Server checks approved list → generates 6-digit PIN → sends email
  3. Athlete enters PIN → server validates → issues JWT session token
  4. Token required for all /api/* routes

Coach:
  - Logs in with email + password
  - Gets JWT with role='coach'
  - Can access /analysis/* routes
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import request, jsonify, g

from models import db, User


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
JWT_SECRET = os.environ.get('JWT_SECRET', 'dev-secret-change-in-production')
JWT_EXPIRY_HOURS = 24
PIN_EXPIRY_MINUTES = 10
PIN_LENGTH = 6


# ---------------------------------------------------------------------------
# PIN GENERATION & VALIDATION
# ---------------------------------------------------------------------------

def generate_pin():
    """Generate a random 6-digit numeric PIN."""
    return ''.join([str(secrets.randbelow(10)) for _ in range(PIN_LENGTH)])


def hash_pin(pin):
    """Hash a PIN for storage."""
    return hashlib.sha256(pin.encode()).hexdigest()


def set_pin_for_user(user):
    """Generate a PIN, store its hash, set expiry. Returns the plain PIN."""
    pin = generate_pin()
    user.pin_hash = hash_pin(pin)
    user.pin_expires = datetime.now(timezone.utc) + timedelta(minutes=PIN_EXPIRY_MINUTES)
    db.session.commit()
    return pin


def validate_pin(user, pin):
    """Check if PIN matches and hasn't expired."""
    if not user.pin_hash or not user.pin_expires:
        return False
    if datetime.now(timezone.utc) > user.pin_expires:
        return False
    return user.pin_hash == hash_pin(pin)


# ---------------------------------------------------------------------------
# PASSWORD (COACH ONLY)
# ---------------------------------------------------------------------------

def hash_password(password):
    """Simple password hash. Use bcrypt in production."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(user, password):
    """Check password against stored hash."""
    if not user.password_hash:
        return False
    return user.password_hash == hash_password(password)


# ---------------------------------------------------------------------------
# JWT TOKENS
# ---------------------------------------------------------------------------

def create_token(user):
    """Create a JWT token for an authenticated user."""
    payload = {
        'user_id': user.id,
        'email': user.email,
        'name': user.name,
        'role': user.role,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        'iat': datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')


def decode_token(token):
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ---------------------------------------------------------------------------
# ROUTE DECORATORS
# ---------------------------------------------------------------------------

def get_current_user():
    """Extract user from Authorization header or cookie."""
    token = None

    # Check Authorization header
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]

    # Check cookie fallback
    if not token:
        token = request.cookies.get('auth_token')

    if not token:
        return None

    payload = decode_token(token)
    if not payload:
        return None

    return User.query.get(payload['user_id'])


def login_required(f):
    """Require any authenticated user (athlete or coach)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def coach_required(f):
    """Require authenticated coach user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401
        if user.role != 'coach':
            return jsonify({'status': 'error', 'message': 'Coach access required.'}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# EMAIL SERVICE (pluggable — Resend, SendGrid, or SMTP)
# ---------------------------------------------------------------------------

def send_pin_email(email, pin):
    """
    Send a PIN to the athlete's email.

    Uses Resend if RESEND_API_KEY is set, otherwise logs to console.
    """
    resend_key = os.environ.get('RESEND_API_KEY')
    from_email = os.environ.get('FROM_EMAIL', 'testing@tremayneperformance.com')

    if resend_key:
        try:
            import requests
            resp = requests.post(
                'https://api.resend.com/emails',
                headers={
                    'Authorization': f'Bearer {resend_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'from': from_email,
                    'to': [email],
                    'subject': 'Your Ramp Test Access PIN',
                    'text': (
                        f'Your access PIN for the DFA Ramp Test is: {pin}\n\n'
                        f'This PIN expires in {PIN_EXPIRY_MINUTES} minutes.\n\n'
                        f'— Tremayne Performance Coaching'
                    ),
                },
            )
            return resp.status_code == 200
        except Exception as e:
            print(f'[EMAIL ERROR] Failed to send PIN to {email}: {e}')
            return False
    else:
        # Development fallback — print PIN to console
        print(f'\n{"="*50}')
        print(f'  PIN for {email}: {pin}')
        print(f'  (Set RESEND_API_KEY to enable email delivery)')
        print(f'{"="*50}\n')
        return True
