# TPC Profiling Tool — DFA A1 Ramp Test

## What This Is
A web-based athlete profiling tool for Tremayne Performance Coaching. Athletes perform structured ramp tests on a bike trainer or treadmill while the app collects heart rate variability (HRV) data via Bluetooth. The coach then analyses results to identify aerobic/anaerobic thresholds using DFA Alpha1 methodology.

## Architecture
- **Frontend (PWA)**: Vanilla JS in `client/` — connects to smart trainers and HR monitors via Web Bluetooth, controls ERG mode, collects per-second data
- **Backend (Flask)**: Python in `server/` — auth, test data storage, DFA/ramp analysis, PDF reports, coach dashboard
- **Database**: PostgreSQL on Railway (falls back to SQLite locally)
- **Deployment**: Railway via Procfile (`gunicorn`), custom domain `app.tremayneperformance.com` behind Cloudflare

## Key Files

### Client (PWA)
| File | Purpose |
|------|---------|
| `client/index.html` | Single-page app — login, setup, test execution, results |
| `client/js/ble.js` | Web Bluetooth manager — FTMS/FE-C trainer control, HRS heart rate + RR intervals |
| `client/js/app.js` | Main controller — auth, test flow, ERG control, phase transitions, data upload |
| `client/js/protocol.js` | Test protocol engine — phases, timing, power targets |
| `client/js/data.js` | Data collection, HRV metrics (RMSSD, SDNN, DFA Alpha1), FIT/CSV/JSON export |
| `client/css/style.css` | UI styling with mobile responsive breakpoints |
| `client/sw.js` | Service worker — network-first caching. **Bump `CACHE_NAME` version on every deploy.** |

### Server
| File | Purpose |
|------|---------|
| `server/app.py` | Flask app — routes, auth, test submission, analysis endpoints, admin |
| `server/auth.py` | JWT auth, password hashing, login/token management |
| `server/models.py` | SQLAlchemy models — User, TestSession, TestRecord, RRInterval |
| `server/ramp_analysis.py` | Core analysis — segment detection, DFA regression, threshold calculation, MAP estimation |
| `server/dfa_core.py` | DFA Alpha1 computation, RR interval cleaning, smoothness priors detrending |
| `server/templates/analysis.html` | Coach analysis dashboard (large file ~3000 lines, JS-heavy) |
| `server/templates/admin.html` | Coach admin/data management page |

## Test Protocol (Bike)
```
Warmup:     20 min @ 45% FTP
DFA Ramp:   10 × 3 min stages (60% → 110% FTP)
Recovery:    8 min @ 45% FTP (ERG stays on)
MAP Ramp:   6 × 1 min steps (110% → 135% FTP, ERG on, auto-terminates on power drop)
Cooldown:   10 min @ 45% FTP
```

The DFA ramp identifies HRVT1/HRVT2 thresholds. The MAP ramp estimates maximal aerobic power (corrected by 5% for ramp overestimation).

## Auth System
- Single login form for both athletes and coach (email + password)
- Coach account bootstrapped from env vars (`COACH_EMAIL`, `COACH_PASSWORD`, `COACH_NAME`)
- Athletes added by coach in admin UI — default password `tpc`, forced to change on first login
- JWT tokens, 24-hour expiry

## Key Thresholds
- **HRVT1s**: DFA Alpha1 = 0.75 (standard)
- **HRVT1c**: Individualised — a1_star = (a1_max + 0.50) / 2
- **HRVT2**: DFA Alpha1 = 0.51 (conservative, triggers early ramp termination in client)
- **MAP correction**: 0.95 (ramp overestimation offset by prior fatigue)

## Environment Variables (Railway)
| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string (`${{Postgres.DATABASE_URL}}`) |
| `COACH_EMAIL` | Coach account email |
| `COACH_PASSWORD` | Coach account password |
| `COACH_NAME` | Coach display name |
| `SECRET_KEY` | Flask session secret |
| `JWT_SECRET` | JWT signing key (defaults to `dev-secret-change-in-production`) |

## Important Patterns
- **Service worker caching**: The SW uses network-first strategy but must have `CACHE_NAME` bumped on every client-side change or users get stale code
- **Bluefy (iOS)**: Web Bluetooth on iOS requires Bluefy browser. Custom 128-bit UUIDs in `requestDevice` filters break Bluefy — keep FE-C UUID in `optionalServices` only
- **ERG mode**: Trainer power control via FTMS Set Target Power (opcode 0x05) or FE-C Data Page 49
- **DFA early termination**: Client monitors DFA Alpha1 per stage; if < 0.51 for a full stage, allows one more stage then skips to recovery
- **MAP auto-termination**: During MAP ramp, if power drops below 30% of target for 8 seconds, skips to cooldown

## Git Practices
- **Commit and push to `main` immediately** — do not batch changes on feature branches for later merging
- **One change at a time** — each commit should be a single logical change so it's easy to identify what broke if something goes wrong
- Railway auto-deploys on push to main
- Always bump `client/sw.js` `CACHE_NAME` when changing client files
- **Never modify `client/js/ble.js`** unless explicitly asked — BLE connection code is fragile and proven working
