/* ============================================
   App Controller — DFA A1 Ramp Test
   With authentication (email + PIN login)
   ============================================ */

const App = (() => {

    // --- Auth State ---
    let currentUser = null;
    let authToken = null;

    // --- State ---
    let sport = 'bike';
    let phases = [];
    let isRunning = false;
    let isPaused = false;
    let startTimestamp = 0;
    let pausedDuration = 0;
    let pauseStart = 0;
    let elapsedSec = 0;
    let tickInterval = null;
    let currentPhaseIndex = -1;

    let latestPower = null;
    let latestHR = null;
    let latestCadence = null;

    let trainerConnected = false;
    let hrmConnected = false;

    let preflightActive = false;
    let preflightRRCount = 0;
    let preflightRRData = [];

    // Early ramp termination tracking
    const HRVT2_THRESHOLD = 0.51;       // DFA Alpha1 below this = HRVT2 reached
    let stageStartTimestamp = null;      // When current ramp stage started (ms)
    let hrvt2StageCompleted = false;     // Has a full stage below HRVT2 been completed?
    let lastCompletedStageDFA = null;    // DFA of the most recently completed stage

    let chart = null;
    let chartData = { labels: [], power: [], hr: [], target: [] };
    const MAX_CHART_POINTS = 600;
    let latestDFA = null;  // Most recent DFA alpha1 value for live chart
    let noPedalingSeconds = 0;  // Counter for consecutive seconds with no/very low power

    let audioCtx = null;

    // MAP ramp auto-termination: detect when athlete stops pedalling
    const MAP_POWER_DROP_THRESHOLD = 0.30; // below 30% of target = stopped
    const MAP_POWER_DROP_DURATION = 8;     // 8 seconds of no power = auto-terminate
    let mapPowerDropSec = 0;               // consecutive seconds below threshold

    // --- Init ---
    function init() {
        bindAuthEvents();
        bindEvents();
        updateSportUI();
        updateProtocolPreview();
        updateStartButton();

        BLE.onTrainerData = handleTrainerData;
        BLE.onHRMData = handleHRMData;
        BLE.onConnectionChange = handleConnectionChange;

        // Check for existing session
        checkExistingSession();
    }

    // --- Auth ---
    async function checkExistingSession() {
        const token = localStorage.getItem('auth_token');
        const userJson = localStorage.getItem('auth_user');
        if (!token || !userJson) return;

        try {
            // Validate token against the server
            const resp = await fetch('/api/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` },
            });
            if (resp.ok) {
                const data = await resp.json();
                currentUser = data.user;
                authToken = token;
                // Update stored user with latest server data
                localStorage.setItem('auth_user', JSON.stringify(data.user));
                onLoginSuccess();
            } else {
                // Token expired or invalid — clear and show login
                localStorage.removeItem('auth_token');
                localStorage.removeItem('auth_user');
            }
        } catch {
            // Network error — try with cached user (offline test support)
            try {
                currentUser = JSON.parse(userJson);
                authToken = token;
                onLoginSuccess();
            } catch {
                localStorage.removeItem('auth_token');
                localStorage.removeItem('auth_user');
            }
        }
    }

    function bindAuthEvents() {
        document.getElementById('btn-athlete-login').addEventListener('click', athleteLogin);
        document.getElementById('btn-set-password').addEventListener('click', setPassword);
        document.getElementById('btn-logout').addEventListener('click', logout);

        // Enter key handlers
        document.getElementById('login-password').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') athleteLogin();
        });
        document.getElementById('login-confirm-password').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') setPassword();
        });
    }

    async function athleteLogin() {
        const email = document.getElementById('login-email').value.trim();
        const password = document.getElementById('login-password').value;

        if (!email) {
            showLoginError('Please enter your email address.');
            return;
        }
        if (!password) {
            showLoginError('Please enter your password.');
            return;
        }

        const btn = document.getElementById('btn-athlete-login');
        btn.disabled = true;
        btn.querySelector('.btn-start-text').textContent = 'LOGGING IN...';
        hideLoginError();

        try {
            const resp = await fetch('/api/auth/athlete-login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });
            const data = await resp.json();

            if (resp.ok) {
                currentUser = data.user;
                authToken = data.token;
                localStorage.setItem('auth_token', data.token);
                localStorage.setItem('auth_user', JSON.stringify(data.user));

                // Check if athlete must set a new password
                if (data.user.password_must_change) {
                    document.getElementById('login-step-email').style.display = 'none';
                    document.getElementById('login-step-set-password').style.display = '';
                    document.getElementById('login-new-password').focus();
                } else {
                    onLoginSuccess();
                }
            } else {
                showLoginError(data.message || 'Login failed.');
            }
        } catch (err) {
            showLoginError('Network error. Please try again.');
        } finally {
            btn.disabled = false;
            btn.querySelector('.btn-start-text').textContent = 'LOG IN';
        }
    }

    async function setPassword() {
        const newPw = document.getElementById('login-new-password').value;
        const confirmPw = document.getElementById('login-confirm-password').value;

        if (newPw.length < 4) {
            showLoginError('Password must be at least 4 characters.');
            return;
        }
        if (newPw !== confirmPw) {
            showLoginError('Passwords do not match.');
            return;
        }

        const btn = document.getElementById('btn-set-password');
        btn.disabled = true;
        btn.querySelector('.btn-start-text').textContent = 'SAVING...';
        hideLoginError();

        try {
            const resp = await fetch('/api/auth/set-password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${authToken}`,
                },
                body: JSON.stringify({ new_password: newPw }),
            });
            const data = await resp.json();

            if (resp.ok) {
                currentUser = data.user;
                localStorage.setItem('auth_user', JSON.stringify(data.user));
                onLoginSuccess();
            } else {
                showLoginError(data.message || 'Could not set password.');
            }
        } catch (err) {
            showLoginError('Network error. Please try again.');
        } finally {
            btn.disabled = false;
            btn.querySelector('.btn-start-text').textContent = 'SET PASSWORD & CONTINUE';
        }
    }

    function onLoginSuccess() {
        // Pre-fill athlete name — editable for coach, read-only for athlete
        const nameInput = document.getElementById('athlete-name');
        nameInput.value = currentUser.name;
        nameInput.readOnly = (currentUser.role === 'athlete');

        // Pre-fill profile fields if available
        if (currentUser.threshold_power) {
            document.getElementById('threshold-power').value = currentUser.threshold_power;
        }
        if (currentUser.threshold_pace) {
            document.getElementById('threshold-pace').value = currentUser.threshold_pace;
        }
        // Use the correct HR max for the currently selected sport
        const hrmax = sport === 'run' ? currentUser.hrmax_run : currentUser.hrmax_bike;
        if (hrmax) {
            document.getElementById('hrmax').value = hrmax;
        }
        if (currentUser.weight_kg) {
            document.getElementById('weight').value = currentUser.weight_kg;
        }

        document.getElementById('btn-logout').style.display = '';
        document.getElementById('ble-status').style.display = '';

        // Show nav links for coach logins
        const navLinks = document.getElementById('nav-links');
        if (navLinks && currentUser && currentUser.role === 'coach') {
            navLinks.style.display = 'flex';
        }

        showScreen('setup-screen');
        updateProtocolPreview();
        updateStartButton();
    }

    function logout() {
        currentUser = null;
        authToken = null;
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');

        // Reset UI
        const navLinks = document.getElementById('nav-links');
        if (navLinks) navLinks.style.display = 'none';
        document.getElementById('btn-logout').style.display = 'none';
        document.getElementById('ble-status').style.display = 'none';
        document.getElementById('login-step-email').style.display = '';
        document.getElementById('login-step-set-password').style.display = 'none';
        document.getElementById('login-email').value = '';
        document.getElementById('login-password').value = '';
        hideLoginError();

        showScreen('login-screen');
    }

    function showLoginError(msg) {
        const el = document.getElementById('login-error');
        el.textContent = msg;
        el.style.display = '';
    }

    function hideLoginError() {
        document.getElementById('login-error').style.display = 'none';
    }

    // --- Event Binding ---
    function bindEvents() {
        document.getElementById('btn-bike').addEventListener('click', () => setSport('bike'));
        document.getElementById('btn-run').addEventListener('click', () => setSport('run'));

        document.getElementById('btn-connect-trainer').addEventListener('click', connectTrainer);
        document.getElementById('btn-connect-hrm-bike').addEventListener('click', connectHRM);
        document.getElementById('btn-connect-hrm-run').addEventListener('click', connectHRM);

        document.getElementById('threshold-power').addEventListener('input', () => {
            updateProtocolPreview();
            updateStartButton();
        });
        document.getElementById('threshold-pace').addEventListener('input', () => {
            updateProtocolPreview();
            updateStartButton();
        });

        document.getElementById('btn-tte-done').addEventListener('click', tteDone);
        document.getElementById('btn-no-pedaling-done').addEventListener('click', () => {
            document.getElementById('no-pedaling-overlay').style.display = 'none';
            tteDone();
        });
        document.getElementById('btn-no-pedaling-resume').addEventListener('click', () => {
            document.getElementById('no-pedaling-overlay').style.display = 'none';
            noPedalingSeconds = 0;
            mapPowerDropSec = 0;
        });
        document.getElementById('btn-start-cooldown').addEventListener('click', startCooldown);
        document.getElementById('btn-start').addEventListener('click', openPreflight);
        document.getElementById('btn-preflight-confirm').addEventListener('click', confirmAndStart);
        document.getElementById('btn-preflight-cancel').addEventListener('click', closePreflight);
        document.getElementById('btn-pause').addEventListener('click', togglePause);
        document.getElementById('btn-skip').addEventListener('click', skipStage);
        document.getElementById('btn-new-test').addEventListener('click', resetToSetup);

        document.getElementById('btn-export-fit').addEventListener('click', () => {
            Data.downloadFIT(document.getElementById('athlete-name').value, sport);
        });
        document.getElementById('btn-export-csv').addEventListener('click', () => {
            Data.downloadCSV(document.getElementById('athlete-name').value, sport);
        });
        document.getElementById('btn-export-json').addEventListener('click', () => {
            Data.downloadJSON(document.getElementById('athlete-name').value, sport);
        });
    }

    // --- Sport Selection ---
    function setSport(s) {
        sport = s;
        document.getElementById('btn-bike').classList.toggle('active', s === 'bike');
        document.getElementById('btn-run').classList.toggle('active', s === 'run');
        updateSportUI();
        updateProtocolPreview();
        updateStartButton();
    }

    function updateSportUI() {
        document.body.classList.toggle('sport-run', sport === 'run');

        document.getElementById('group-threshold-power').style.display = sport === 'bike' ? '' : 'none';
        document.getElementById('group-threshold-pace').style.display = sport === 'run' ? '' : 'none';
        document.getElementById('group-weight').style.display = sport === 'bike' ? '' : 'none';

        document.getElementById('bike-devices').style.display = sport === 'bike' ? '' : 'none';
        document.getElementById('run-devices').style.display = sport === 'run' ? '' : 'none';

        document.getElementById('power-display').style.display = sport === 'bike' ? '' : 'none';
        document.getElementById('cadence-display').style.display = sport === 'bike' ? '' : 'none';
        document.getElementById('pace-display').style.display = sport === 'run' ? '' : 'none';
        document.getElementById('target-power-row').style.display = sport === 'bike' ? '' : 'none';
    }

    // --- Protocol Preview ---
    function updateProtocolPreview() {
        const summary = Protocol.getProtocolSummary(sport);
        const bar = document.getElementById('protocol-bar');
        const stagesEl = document.getElementById('protocol-stages');

        bar.innerHTML = summary.segments.map(seg =>
            `<div class="protocol-segment ${seg.color}" style="flex:${seg.pct}" title="${seg.label}">${seg.label}</div>`
        ).join('');

        const threshold = getThresholdValue();
        if (threshold && threshold > 0) {
            const p = Protocol.buildPhases(sport, threshold);
            const rampPhases = p.filter(ph => ph.id === Protocol.PHASE.RAMP);
            stagesEl.innerHTML = rampPhases.map(ph =>
                `<div class="stage-chip"><span class="stage-num">${ph.label}</span><span class="stage-target">${ph.targetDisplay}</span></div>`
            ).join('');
        } else {
            stagesEl.innerHTML = '<p style="color:var(--text-dim);font-size:0.75rem;">Enter threshold to see stage targets</p>';
        }
    }

    function getThresholdValue() {
        if (sport === 'bike') {
            return parseInt(document.getElementById('threshold-power').value) || 0;
        } else {
            return Protocol.parsePace(document.getElementById('threshold-pace').value) || 0;
        }
    }

    // --- Device Connections ---
    async function connectTrainer() {
        const btn = document.getElementById('btn-connect-trainer');
        btn.textContent = 'CONNECTING...';
        btn.disabled = true;
        try {
            const name = await BLE.connectTrainer();
            document.getElementById('trainer-name').textContent = name;
            document.getElementById('trainer-name').classList.add('active');
            document.getElementById('slot-trainer').classList.add('connected');
            btn.textContent = 'CONNECTED';
            btn.classList.add('connected');

            // Safety net: explicitly set flag and update button
            trainerConnected = true;
            updateBLEStatus();
            updateStartButton();
        } catch (e) {
            btn.textContent = 'CONNECT';
            btn.disabled = false;
            console.error('Trainer connection failed:', e);
        }
    }

    async function connectHRM() {
        const isBike = sport === 'bike';
        const btn = document.getElementById(isBike ? 'btn-connect-hrm-bike' : 'btn-connect-hrm-run');
        const nameEl = document.getElementById(isBike ? 'hrm-bike-name' : 'hrm-run-name');
        const slot = document.getElementById(isBike ? 'slot-hrm-bike' : 'slot-hrm-run');

        btn.textContent = 'CONNECTING...';
        btn.disabled = true;
        try {
            const name = await BLE.connectHRM();
            nameEl.textContent = name;
            nameEl.classList.add('active');
            slot.classList.add('connected');
            btn.textContent = 'CONNECTED';
            btn.classList.add('connected');

            // Ensure both sport panels reflect the connection
            if (isBike) {
                document.getElementById('hrm-run-name').textContent = name;
                document.getElementById('hrm-run-name').classList.add('active');
            } else {
                document.getElementById('hrm-bike-name').textContent = name;
                document.getElementById('hrm-bike-name').classList.add('active');
            }

            // Safety net: explicitly set flag and update button
            // (handleConnectionChange should also do this via BLE callback,
            //  but ensure it happens even if callback timing varies)
            hrmConnected = true;
            updateBLEStatus();
            updateStartButton();
        } catch (e) {
            btn.textContent = 'CONNECT';
            btn.disabled = false;
            console.error('HRM connection failed:', e);
        }
    }

    function handleConnectionChange(device, connected, name) {
        if (device === 'trainer') {
            trainerConnected = connected;
            if (!connected) {
                document.getElementById('trainer-name').textContent = 'Disconnected';
                document.getElementById('trainer-name').classList.remove('active');
                document.getElementById('slot-trainer').classList.remove('connected');
                const btn = document.getElementById('btn-connect-trainer');
                btn.textContent = 'RECONNECT';
                btn.disabled = false;
                btn.classList.remove('connected');
            }
        } else if (device === 'hrm') {
            hrmConnected = connected;
            if (!connected) {
                ['hrm-bike-name', 'hrm-run-name'].forEach(id => {
                    document.getElementById(id).textContent = 'Disconnected';
                    document.getElementById(id).classList.remove('active');
                });
                ['slot-hrm-bike', 'slot-hrm-run'].forEach(id => {
                    document.getElementById(id).classList.remove('connected');
                });
                ['btn-connect-hrm-bike', 'btn-connect-hrm-run'].forEach(id => {
                    const btn = document.getElementById(id);
                    btn.textContent = 'RECONNECT';
                    btn.disabled = false;
                    btn.classList.remove('connected');
                });
            }
        }
        updateBLEStatus();
        updateStartButton();
    }

    function updateBLEStatus() {
        const el = document.getElementById('ble-status');
        if (sport === 'bike') {
            if (trainerConnected && hrmConnected) {
                el.textContent = 'ALL CONNECTED';
                el.className = 'status-chip connected';
            } else if (trainerConnected || hrmConnected) {
                el.textContent = 'PARTIAL';
                el.className = 'status-chip partial';
            } else {
                el.textContent = 'NO DEVICES';
                el.className = 'status-chip disconnected';
            }
        } else {
            if (hrmConnected) {
                el.textContent = 'HR CONNECTED';
                el.className = 'status-chip connected';
            } else {
                el.textContent = 'NO DEVICES';
                el.className = 'status-chip disconnected';
            }
        }
    }

    function updateStartButton() {
        const btn = document.getElementById('btn-start');
        const sub = document.getElementById('start-sub');
        const threshold = getThresholdValue();

        let canStart = false;
        if (sport === 'bike') {
            canStart = trainerConnected && hrmConnected && threshold > 0;
            if (!trainerConnected && !hrmConnected) sub.textContent = 'Connect trainer and HR monitor';
            else if (!trainerConnected) sub.textContent = 'Connect smart trainer';
            else if (!hrmConnected) sub.textContent = 'Connect HR monitor for HRV data';
            else if (!threshold) sub.textContent = 'Enter threshold power';
            else sub.textContent = `${Protocol.formatTime(Protocol.totalDuration(Protocol.buildPhases(sport, threshold)))} total`;
        } else {
            canStart = hrmConnected && threshold > 0;
            if (!hrmConnected) sub.textContent = 'Connect HR monitor';
            else if (!threshold) sub.textContent = 'Enter threshold pace';
            else sub.textContent = `${Protocol.formatTime(Protocol.totalDuration(Protocol.buildPhases(sport, threshold)))} total`;
        }

        btn.disabled = !canStart;
    }

    // --- Sensor Data Handlers ---
    function handleTrainerData(data) {
        if (data.power !== undefined) latestPower = data.power;
        if (data.cadence !== undefined) latestCadence = data.cadence;
        if (data.heartRate !== undefined && data.heartRate > 0 && !hrmConnected) {
            latestHR = data.heartRate;
        }
    }

    function handleHRMData(data) {
        // Only update HR if the sensor reports a valid (non-zero) value.
        // Polar H10 sends heartRate=0 intermittently when contact is marginal;
        // keep the last valid reading visible rather than flashing 0.
        if (data.heartRate && data.heartRate > 0) {
            latestHR = data.heartRate;
        }

        if (preflightActive && data.rrIntervals && data.rrIntervals.length > 0) {
            for (const rr of data.rrIntervals) {
                preflightRRData.push({ time: Date.now(), rr });
                preflightRRCount++;
            }
            updatePreflightUI(data);
        }

        if (data.rrIntervals && data.rrIntervals.length > 0 && isRunning && !isPaused) {
            Data.addRRIntervals(data.rrIntervals, Date.now());
            updateHRVDisplay();
        }
    }

    // --- RR Pre-flight ---
    function openPreflight() {
        preflightActive = true;
        preflightRRCount = 0;
        preflightRRData = [];

        document.getElementById('preflight-hr').textContent = '---';
        document.getElementById('preflight-rr-count').textContent = '0';
        document.getElementById('preflight-dfa').textContent = '---';
        document.getElementById('preflight-rr-tickers').innerHTML = '';
        document.getElementById('preflight-dot').className = 'preflight-dot';
        document.getElementById('preflight-status-text').textContent = 'Waiting for RR data from HR monitor...';
        document.getElementById('btn-preflight-confirm').disabled = true;
        document.getElementById('preflight-confirm-sub').textContent = 'Waiting for RR data...';

        document.getElementById('rr-preflight-overlay').classList.add('active');
    }

    function closePreflight() {
        preflightActive = false;
        preflightRRData = [];
        document.getElementById('rr-preflight-overlay').classList.remove('active');
    }

    function updatePreflightUI(data) {
        if (latestHR) {
            document.getElementById('preflight-hr').textContent = latestHR;
        }
        document.getElementById('preflight-rr-count').textContent = preflightRRCount;

        const tickers = document.getElementById('preflight-rr-tickers');
        if (data.rrIntervals) {
            for (const rr of data.rrIntervals) {
                const chip = document.createElement('span');
                chip.className = 'rr-tick';
                chip.textContent = Math.round(rr);
                tickers.prepend(chip);
                while (tickers.children.length > 30) {
                    tickers.removeChild(tickers.lastChild);
                }
            }
        }

        if (preflightRRData.length >= 64) {
            const alpha1 = Data.computeDFA(preflightRRData, 4, 16);
            if (alpha1 !== null) {
                document.getElementById('preflight-dfa').textContent = alpha1.toFixed(2);
            }
        }

        const dot = document.getElementById('preflight-dot');
        const text = document.getElementById('preflight-status-text');
        const btn = document.getElementById('btn-preflight-confirm');
        const sub = document.getElementById('preflight-confirm-sub');

        if (preflightRRCount >= 10) {
            dot.className = 'preflight-dot streaming';
            text.textContent = `RR data confirmed — ${preflightRRCount} intervals received`;
            btn.disabled = false;
            sub.textContent = 'RR data verified — ready to begin';
        } else {
            dot.className = 'preflight-dot waiting';
            text.textContent = `Receiving RR data... (${preflightRRCount}/10 minimum)`;
            sub.textContent = `Need ${10 - preflightRRCount} more RR intervals...`;
        }
    }

    function confirmAndStart() {
        closePreflight();
        startTest();
    }

    // --- Test Execution ---
    function startTest() {
        const threshold = getThresholdValue();
        phases = Protocol.buildPhases(sport, threshold);

        Data.init({
            sport,
            athleteName: document.getElementById('athlete-name').value,
            thresholdValue: threshold,
            hrMax: parseInt(document.getElementById('hrmax').value) || null,
            weight: parseFloat(document.getElementById('weight').value) || null,
        });

        isRunning = true;
        isPaused = false;
        startTimestamp = performance.now();
        pausedDuration = 0;
        currentPhaseIndex = -1;
        elapsedSec = 0;

        // Reset early termination state
        stageStartTimestamp = null;
        hrvt2StageCompleted = false;
        lastCompletedStageDFA = null;

        chartData = { labels: [], power: [], hr: [], target: [] };

        showScreen('test-screen');
        buildProgressSegments();
        initChart();

        tickInterval = setInterval(tick, 1000);

        if (sport === 'bike' && phases.length > 0) {
            BLE.setTargetPower(phases[0].target);
        }
    }

    function tick() {
        if (!isRunning || isPaused) return;

        const now = performance.now();
        elapsedSec = (now - startTimestamp - pausedDuration) / 1000;

        const state = Protocol.getPhaseAt(phases, elapsedSec);

        if (state.index !== currentPhaseIndex) {
            onPhaseChange(state);
            currentPhaseIndex = state.index;
        }

        if (state.phase.id === Protocol.PHASE.COMPLETE) {
            completeTest();
            return;
        }

        // --- Bike ERG mode control ---
        if (sport === 'bike') {
            if (state.phase.isMapRamp) {
                // MAP ramp: ERG on at target, auto-terminate on power drop
                BLE.setTargetPower(state.phase.target);
                if (latestPower != null && latestPower < state.phase.target * MAP_POWER_DROP_THRESHOLD) {
                    mapPowerDropSec++;
                    if (mapPowerDropSec >= MAP_POWER_DROP_DURATION) {
                        // Athlete has stopped — skip remaining MAP stages to cooldown
                        mapRampFailure(state);
                    }
                } else {
                    mapPowerDropSec = 0;
                }
            } else if (state.phase.isMaxEffort) {
                // Legacy max effort (non-ramp): ERG off
            } else if (state.phase.id === Protocol.PHASE.RECOVERY) {
                // Recovery: ERG on at recovery power for full duration
                BLE.setTargetPower(state.phase.target);
                showPowerWarning(false);
            } else if (state.phase.id === Protocol.PHASE.COOLDOWN) {
                // Cooldown: re-enable ERG
                BLE.setTargetPower(state.phase.target);
                showPowerWarning(false);
            } else {
                // Warmup & ramp stages: ERG on
                BLE.setTargetPower(state.phase.target);
                showPowerWarning(false);
            }
        }

        Data.addRecord({
            time: Date.now(),
            elapsed: elapsedSec,
            power: latestPower,
            heartRate: latestHR,
            cadence: latestCadence,
            targetPower: sport === 'bike' ? state.phase.target : null,
            phase: state.phase.id,
            stageNum: state.phase.stageNum || null,
        });

        // --- Effort overlay timer update (run TTE only) ---
        if (state.phase.id === Protocol.PHASE.TTE) {
            document.getElementById('effort-overlay-timer').textContent =
                Protocol.formatTime(Math.floor(state.phaseElapsed));
            document.getElementById('effort-overlay-target').textContent = '';
        }

        // --- No-pedaling detection (bike MAP ramp only) ---
        if (sport === 'bike' && state.phase.isMaxEffort) {
            if (latestPower != null && latestPower < state.phase.target * 0.30) {
                noPedalingSeconds++;
                if (noPedalingSeconds >= 8) {
                    document.getElementById('no-pedaling-overlay').style.display = 'flex';
                }
            } else {
                noPedalingSeconds = 0;
                document.getElementById('no-pedaling-overlay').style.display = 'none';
            }
        }

        updateTestUI(state);
        updateChart(state);
    }

    function onPhaseChange(state) {
        const prevPhaseIndex = currentPhaseIndex;
        const prevPhase = prevPhaseIndex >= 0 ? phases[prevPhaseIndex] : null;

        // --- Early ramp termination check ---
        // When transitioning OUT of a ramp stage, compute that stage's DFA.
        //
        // Logic: if stage N has DFA < 0.50 (HRVT2 reached), mark it.
        // Then the athlete completes one MORE full stage (N+1).
        // When stage N+1 completes → skip remaining ramp → recovery.
        //
        // Example: Stage 7 DFA = 0.43 → mark hrvt2StageCompleted.
        //          Stage 8 completes → early termination triggers.
        //          Test proceeds: recovery → max effort/TTE → cooldown.
        if (prevPhase && prevPhase.id === Protocol.PHASE.RAMP && stageStartTimestamp) {
            const stageEndTimestamp = Date.now();
            const stageDFA = Data.getStageDFA(stageStartTimestamp, stageEndTimestamp);

            if (stageDFA !== null) {
                lastCompletedStageDFA = stageDFA;
                console.log(`Stage ${prevPhase.stageNum} DFA: ${stageDFA.toFixed(3)}`);

                if (hrvt2StageCompleted && state.phase.id === Protocol.PHASE.RAMP) {
                    // hrvt2 was flagged on a PREVIOUS stage, and the athlete
                    // has now completed one full additional stage — trigger early end.
                    const completedStage = prevPhase.stageNum;
                    console.log(`Early ramp termination after stage ${completedStage}`);

                    Data.setEarlyRampEnd({
                        stage: completedStage,
                        reason: 'hrvt2_reached',
                        dfa: stageDFA,
                        timestamp: stageEndTimestamp,
                    });

                    // Remove remaining ramp stages from phases array
                    const recoveryIdx = phases.findIndex(p => p.id === Protocol.PHASE.RECOVERY);
                    if (recoveryIdx > state.index) {
                        const removed = phases.splice(state.index, recoveryIdx - state.index);
                        console.log(`Removed ${removed.length} remaining ramp stages`);

                        // Rebuild progress segments to reflect shorter test
                        buildProgressSegments();

                        // Play distinctive double-beep notification
                        playBeep(440, 0.3);
                        setTimeout(() => playBeep(880, 0.3), 400);

                        // Update state to point at the recovery phase
                        state.phase = phases[state.index];
                    }
                }

                // Mark HRVT2 AFTER the termination check, so it doesn't
                // trigger on the same stage it's detected — the athlete
                // must complete one more full stage first.
                if (stageDFA < HRVT2_THRESHOLD && !hrvt2StageCompleted) {
                    hrvt2StageCompleted = true;
                    console.log(`HRVT2 reached at stage ${prevPhase.stageNum} (DFA=${stageDFA.toFixed(3)}) — one more full stage required`);
                }
            }
        }

        // Track when a new ramp stage starts
        if (state.phase.id === Protocol.PHASE.RAMP) {
            stageStartTimestamp = Date.now();
        } else {
            stageStartTimestamp = null;
        }

        // --- Bike ERG transitions on phase change ---
        if (sport === 'bike') {
            if (state.phase.isMapRamp) {
                // Entering MAP ramp step — ERG on at target
                BLE.setTargetPower(state.phase.target);
                mapPowerDropSec = 0;
            } else if (state.phase.id === Protocol.PHASE.MAX_EFFORT) {
                // Legacy max effort — disable ERG
                BLE.setTargetPower(0);
            } else if (state.phase.id === Protocol.PHASE.COOLDOWN) {
                // Entering cooldown — re-enable ERG
                BLE.setTargetPower(state.phase.target);
                showPowerWarning(false);
            }
        }

        // Audio cue
        playBeep(state.phase.id === Protocol.PHASE.MAX_EFFORT || state.phase.id === Protocol.PHASE.TTE ? 880 : 660);

        // Show/hide effort overlay for RUN TTE only (bike uses normal UI with no-pedaling detection)
        const effortOverlay = document.getElementById('effort-overlay');
        if (state.phase.id === Protocol.PHASE.TTE) {
            effortOverlay.style.display = 'flex';
            document.getElementById('effort-overlay-label').textContent = 'TIME TO EXHAUSTION';
            document.getElementById('effort-overlay-total').textContent =
                'of ' + Protocol.formatTime(state.phase.duration);
        } else if (state.phase.id === Protocol.PHASE.MAX_EFFORT) {
            // Bike MAP ramp: no overlay, just reset no-pedaling counter
            noPedalingSeconds = 0;
        } else {
            effortOverlay.style.display = 'none';
            document.getElementById('no-pedaling-overlay').style.display = 'none';
            document.getElementById('cooldown-transition-overlay').style.display = 'none';
            noPedalingSeconds = 0;
        }

        // Update progress segments
        const segments = document.querySelectorAll('.progress-seg');
        segments.forEach((seg, i) => {
            seg.classList.toggle('active', i === state.index);
            seg.classList.toggle('done', i < state.index);
        });
    }

    /**
     * "I'M DONE" button handler — athlete ends the TTE/max effort early.
     * Skips to cooldown phase immediately.
     */
    function tteDone() {
        if (!isRunning) return;

        const state = Protocol.getPhaseAt(phases, elapsedSec);
        if (state.phase.id !== Protocol.PHASE.TTE && state.phase.id !== Protocol.PHASE.MAX_EFFORT) {
            return; // Safety — only works during TTE/max effort
        }

        // Record the actual effort duration within current phase
        const phasesBeforeCurrent = phases.slice(0, state.index);
        const phaseStartElapsed = phasesBeforeCurrent.reduce((sum, p) => sum + p.duration, 0);
        const actualDuration = elapsedSec - phaseStartElapsed;

        console.log(`TTE/Max effort ended by athlete after ${actualDuration.toFixed(1)}s`);

        // Shorten current phase to end now
        phases[state.index].duration = actualDuration;

        // Remove any remaining MAP ramp steps after this one (skip to cooldown)
        const cooldownIdx = phases.findIndex(p => p.id === Protocol.PHASE.COOLDOWN);
        if (cooldownIdx > state.index + 1) {
            phases.splice(state.index + 1, cooldownIdx - state.index - 1);
            buildProgressSegments();
        }

        // Jump elapsed time to the end of this phase
        const newElapsed = phaseStartElapsed + actualDuration;
        const adjustment = (newElapsed - elapsedSec) * 1000;
        startTimestamp -= adjustment;

        // Hide overlays
        document.getElementById('effort-overlay').style.display = 'none';
        document.getElementById('no-pedaling-overlay').style.display = 'none';

        if (sport === 'run') {
            // Run: show cooldown transition overlay with treadmill pace
            const cooldownPhase = phases.find(p => p.id === Protocol.PHASE.COOLDOWN);
            const paceHint = cooldownPhase
                ? `set treadmill to ${cooldownPhase.targetDisplay}`
                : '';
            document.getElementById('cooldown-pace-hint').textContent = paceHint;
            document.getElementById('cooldown-transition-overlay').style.display = 'flex';

            // Pause the test while athlete gets back on treadmill
            if (!isPaused) togglePause();
        } else {
            // Bike: proceed directly to cooldown
            playBeep(440, 0.3);
        }
    }

    /**
     * "START COOLDOWN" button handler — resumes test after run TTE.
     */
    function startCooldown() {
        document.getElementById('cooldown-transition-overlay').style.display = 'none';
        if (isPaused) togglePause();
        playBeep(440, 0.3);
    }

    /**
     * MAP ramp auto-termination — athlete has stopped pedalling.
     * Truncates remaining MAP stages and skips to cooldown.
     */
    function mapRampFailure(state) {
        if (!isRunning) return;
        console.log(`MAP ramp: power drop detected at stage ${state.phase.label}`);
        // Show the no-pedaling overlay — athlete decides to stop or continue
        document.getElementById('no-pedaling-overlay').style.display = 'flex';
    }

    function togglePause() {
        const btn = document.getElementById('btn-pause');
        if (isPaused) {
            isPaused = false;
            pausedDuration += performance.now() - pauseStart;
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>PAUSE`;
            btn.classList.remove('paused');
        } else {
            isPaused = true;
            pauseStart = performance.now();
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><polygon points="5,3 19,12 5,21"/></svg>RESUME`;
            btn.classList.add('paused');
        }
    }

    function skipStage() {
        if (!isRunning) return;
        const state = Protocol.getPhaseAt(phases, elapsedSec);
        const phasesBeforeCurrent = phases.slice(0, state.index + 1);
        const newElapsed = phasesBeforeCurrent.reduce((sum, p) => sum + p.duration, 0);
        const adjustment = (newElapsed - elapsedSec) * 1000;
        startTimestamp -= adjustment;
        playBeep(440);
    }

    async function completeTest() {
        isRunning = false;
        clearInterval(tickInterval);

        if (sport === 'bike') {
            BLE.setTargetPower(50);
        }

        playBeep(440, 0.5);
        showResults();

        // Auto-upload to server
        const uploadStatus = document.getElementById('upload-status');
        try {
            const result = await Data.uploadToServer();
            if (result.ok) {
                uploadStatus.innerHTML = `
                    <p style="color:#22c55e;">Data uploaded successfully.</p>
                    <p style="color:var(--text-dim);font-size:0.8rem;">Your coach can now analyse this test in the dashboard.</p>
                `;
            } else {
                uploadStatus.innerHTML = `
                    <p style="color:#f59e0b;">Upload issue: ${result.message}</p>
                    <p style="color:var(--text-dim);font-size:0.8rem;">Use the export buttons below to save your data locally.</p>
                `;
            }
        } catch (err) {
            uploadStatus.innerHTML = `
                <p style="color:#f59e0b;">Could not upload — please export data manually.</p>
            `;
        }
    }

    // --- UI Updates ---
    function updateTestUI(state) {
        document.getElementById('metric-elapsed').textContent = Protocol.formatTime(elapsedSec);
        document.getElementById('metric-stage').textContent = state.phase.label;
        document.getElementById('metric-remaining').textContent = Protocol.formatTime(state.phaseRemaining);

        if (sport === 'bike') {
            document.getElementById('metric-power').textContent = latestPower != null ? latestPower : '---';

            // Target display: show context-appropriate target
            if (state.phase.isMaxEffort) {
                document.getElementById('metric-target-power').textContent = state.phase.target;
            } else if (state.phase.id === Protocol.PHASE.RECOVERY && state.phaseElapsed >= state.phase.duration / 2) {
                // Second half of recovery — ERG off, athlete spinning up
                document.getElementById('metric-target-power').textContent = 'FREE';
            } else {
                document.getElementById('metric-target-power').textContent = state.phase.target;
            }

            document.getElementById('metric-cadence').textContent = latestCadence != null ? Math.round(latestCadence) : '---';
        } else {
            document.getElementById('metric-target-pace').textContent = state.phase.targetDisplay || '--:--';
        }

        document.getElementById('metric-hr').textContent = latestHR != null ? latestHR : '---';
        document.getElementById('metric-rr-count').textContent = Data.allRR.length;

        document.getElementById('progress-bar').style.width = `${state.totalProgress * 100}%`;
    }

    function showPowerWarning(show) {
        let el = document.getElementById('power-warning');
        if (show) {
            if (!el) {
                el = document.createElement('div');
                el.id = 'power-warning';
                el.className = 'power-warning';
                el.textContent = 'POWER TOO HIGH — ease off for recovery';
                const metrics = document.querySelector('.primary-metrics');
                if (metrics) metrics.parentNode.insertBefore(el, metrics.nextSibling);
            }
            el.style.display = '';
        } else if (el) {
            el.style.display = 'none';
        }
    }

    function updateHRVDisplay() {
        const hrv = Data.getRecentHRV(30);
        const dot = document.getElementById('hrv-dot');
        const text = document.getElementById('hrv-status-text');

        dot.classList.add('receiving');

        if (hrv.artifactPct > 10) {
            dot.classList.add('warning');
            dot.classList.remove('receiving');
            text.textContent = `High artifact rate (${hrv.artifactPct}%)`;
        } else {
            dot.classList.remove('warning');
            text.textContent = `RR data streaming (${Data.allRR.length} intervals)`;
        }

        document.getElementById('hrv-rmssd').textContent = `RMSSD: ${hrv.rmssd ?? '---'}`;
        document.getElementById('hrv-sdnn').textContent = `SDNN: ${hrv.sdnn ?? '---'}`;
        document.getElementById('metric-artifact').textContent = hrv.artifactPct.toFixed(1);

        const dfa = Data.getRecentDFA(120);
        const dfaEl = document.getElementById('metric-dfa');
        const dfaZone = document.getElementById('dfa-zone-label');

        if (dfa !== null) {
            latestDFA = dfa;
            dfaEl.textContent = dfa.toFixed(2);

            if (dfa >= 0.75) {
                dfaEl.className = 'small-value dfa-value dfa-zone1';
                dfaZone.textContent = 'ZONE 1 — aerobic';
            } else if (dfa >= 0.5) {
                dfaEl.className = 'small-value dfa-value dfa-transition';
                dfaZone.textContent = 'TRANSITION';
            } else {
                dfaEl.className = 'small-value dfa-value dfa-zone2';
                dfaZone.textContent = 'ZONE 2+ — threshold';
            }
        } else {
            dfaEl.textContent = '---';
            dfaEl.className = 'small-value dfa-value';
            dfaZone.textContent = Data.allRR.length > 0 ? `${Data.allRR.length}/120 beats` : 'waiting for data';
        }
    }

    function buildProgressSegments() {
        const container = document.getElementById('progress-segments');
        container.innerHTML = phases.map((p, i) => {
            const cssClass = p.id === Protocol.PHASE.TTE ? 'max-effort' : p.id;
            return `<div class="progress-seg ${cssClass}" title="${p.label}">${p.stageNum || p.label.charAt(0)}</div>`;
        }).join('');
    }

    // --- Chart ---
    function initChart() {
        const ctx = document.getElementById('live-chart').getContext('2d');
        if (chart) chart.destroy();

        const dfaDataset = {
            label: 'DFA \u03b11',
            data: [],
            borderColor: '#22c55e',
            backgroundColor: 'rgba(34, 197, 94, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            yAxisID: 'yDFA',
            fill: false,
            tension: 0.4,
            spanGaps: true,
        };

        const datasets = [
            {
                label: 'Heart Rate',
                data: [],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 1.5,
                pointRadius: 0,
                yAxisID: 'y1',
                fill: true,
                tension: 0.3,
            },
            dfaDataset,
        ];

        if (sport === 'bike') {
            datasets.unshift(
                {
                    label: 'Power',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    yAxisID: 'y',
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: 'Target',
                    data: [],
                    borderColor: 'rgba(139, 92, 246, 0.5)',
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    yAxisID: 'y',
                    fill: false,
                },
            );
        }

        // Custom plugin: draw horizontal DFA threshold reference lines
        const dfaThresholdLines = {
            id: 'dfaThresholdLines',
            afterDraw(chart) {
                const yAxis = chart.scales.yDFA;
                if (!yAxis) return;
                const ctx = chart.ctx;
                const left = chart.chartArea.left;
                const right = chart.chartArea.right;

                [{ val: 0.75, color: 'rgba(34,197,94,0.35)', label: 'LT1' },
                 { val: 0.50, color: 'rgba(239,68,68,0.35)', label: 'LT2' }].forEach(ref => {
                    const y = yAxis.getPixelForValue(ref.val);
                    if (y < chart.chartArea.top || y > chart.chartArea.bottom) return;
                    ctx.save();
                    ctx.strokeStyle = ref.color;
                    ctx.lineWidth = 1;
                    ctx.setLineDash([6, 4]);
                    ctx.beginPath();
                    ctx.moveTo(left, y);
                    ctx.lineTo(right, y);
                    ctx.stroke();
                    ctx.fillStyle = ref.color;
                    ctx.font = '9px Inter, sans-serif';
                    ctx.fillText(ref.label + ' (' + ref.val + ')', right - 52, y - 3);
                    ctx.restore();
                });
            },
        };

        chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets },
            plugins: [dfaThresholdLines],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'nearest', intersect: false },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#6b7280',
                            font: { family: 'Inter', size: 10 },
                            boxWidth: 12,
                            padding: 12,
                        },
                    },
                },
                scales: {
                    x: {
                        display: true,
                        ticks: { color: '#4b5563', font: { size: 9 }, maxTicksLimit: 8 },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                    },
                    y: {
                        display: sport === 'bike',
                        position: 'left',
                        title: { display: true, text: 'Power (W)', color: '#6b7280', font: { size: 10 } },
                        ticks: { color: '#4b5563', font: { size: 9 } },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        beginAtZero: true,
                    },
                    y1: {
                        display: true,
                        position: sport === 'bike' ? 'right' : 'left',
                        title: { display: true, text: 'HR (bpm)', color: '#6b7280', font: { size: 10 } },
                        ticks: { color: '#4b5563', font: { size: 9 } },
                        grid: { drawOnChartArea: sport !== 'bike', color: 'rgba(255,255,255,0.03)' },
                        beginAtZero: false,
                    },
                    yDFA: {
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'DFA \u03b11', color: '#22c55e', font: { size: 10 } },
                        ticks: { color: '#22c55e', font: { size: 9 } },
                        grid: { drawOnChartArea: false },
                        min: 0,
                        max: 1.5,
                        afterBuildTicks(axis) {
                            axis.ticks = [0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5].map(v => ({ value: v }));
                        },
                    },
                },
            },
        });
    }

    function updateChart(state) {
        const label = Protocol.formatTime(elapsedSec);
        chart.data.labels.push(label);

        // DFA dataset is always the last one
        const dfaIdx = chart.data.datasets.length - 1;

        if (sport === 'bike') {
            chart.data.datasets[0].data.push(latestPower);
            chart.data.datasets[1].data.push(state.phase.target);
            chart.data.datasets[2].data.push(latestHR);
        } else {
            chart.data.datasets[0].data.push(latestHR);
        }

        // Push DFA value (null if not yet available — spanGaps handles it)
        chart.data.datasets[dfaIdx].data.push(latestDFA);

        if (chart.data.labels.length > MAX_CHART_POINTS) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(ds => ds.data.shift());
        }

        chart.update('none');
    }

    // --- Results ---
    function showResults() {
        const summary = Data.getSummary();

        document.getElementById('result-duration').textContent = Protocol.formatTime(summary.duration);
        document.getElementById('result-stages').textContent = summary.stagesCompleted;
        document.getElementById('result-peak-power').textContent = summary.peakPower > 0 ? `${summary.peakPower} W` : 'N/A';
        document.getElementById('result-peak-hr').textContent = summary.peakHR > 0 ? `${summary.peakHR} bpm` : 'N/A';
        document.getElementById('result-rr-total').textContent = summary.totalRR.toLocaleString();
        document.getElementById('result-artifact').textContent = `${summary.artifactPct}%`;

        const qualityBar = document.getElementById('quality-bar');
        const qualityNote = document.getElementById('quality-note');
        const qualityClass = summary.artifactPct < 5 ? 'good' : summary.artifactPct < 15 ? 'moderate' : 'poor';
        const qualityWidth = Math.max(5, 100 - summary.artifactPct * 2);
        qualityBar.innerHTML = `<div class="quality-fill ${qualityClass}" style="width:${qualityWidth}%"></div>`;

        if (qualityClass === 'good') {
            qualityNote.textContent = `Excellent data quality. ${summary.totalRR} RR intervals collected with ${summary.artifactPct}% artifact rate. Ready for DFA analysis.`;
        } else if (qualityClass === 'moderate') {
            qualityNote.textContent = `Moderate data quality. Consider checking HR strap contact. ${summary.artifactPct}% artifact rate may affect threshold precision.`;
        } else {
            qualityNote.textContent = `High artifact rate (${summary.artifactPct}%). DFA results may be unreliable. Ensure HR strap is wet and fitted snugly.`;
        }

        showScreen('results-screen');
    }

    function resetToSetup() {
        latestPower = null;
        latestHR = null;
        latestCadence = null;
        currentPhaseIndex = -1;
        if (chart) chart.destroy();
        chart = null;
        // Reset upload status
        document.getElementById('upload-status').innerHTML = '<p style="color:var(--text-dim);">Uploading test data to server...</p>';
        showScreen('setup-screen');
    }

    // --- Screen Management ---
    function showScreen(id) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.getElementById(id).classList.add('active');
    }

    // --- Audio Cue ---
    function playBeep(freq = 660, duration = 0.15) {
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.frequency.value = freq;
            gain.gain.value = 0.3;
            osc.start();
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
            osc.stop(audioCtx.currentTime + duration + 0.05);
        } catch (e) {
            // Audio not available
        }
    }

    // --- Boot ---
    document.addEventListener('DOMContentLoaded', init);

    return { init };
})();
