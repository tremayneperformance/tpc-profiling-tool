/* ============================================
   Data Module — Collection, HRV Analysis, Export & Server Upload
   DFA Alpha1 pipeline matches dfa_core.py (Kubios HRV)
   ============================================ */

const Data = (() => {

    // --- Collected Data Arrays ---
    let records = [];       // Per-second records
    let allRR = [];         // All RR intervals with timestamps { time, rr }
    let metadata = {};      // Test metadata

    // =================================================================
    // MATH HELPERS
    // =================================================================

    const RR_MIN = 300;
    const RR_MAX = 2000;

    function _median(arr) {
        if (arr.length === 0) return 0;
        const s = arr.slice().sort((a, b) => a - b);
        const m = Math.floor(s.length / 2);
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    }

    function _percentile(arr, p) {
        if (arr.length === 0) return 0;
        const s = arr.slice().sort((a, b) => a - b);
        const idx = (p / 100) * (s.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.ceil(idx);
        if (lo === hi) return s[lo];
        return s[lo] + (idx - lo) * (s[hi] - s[lo]);
    }

    function _localMedianRR(rrValues, idx, halfWindow) {
        const lo = Math.max(0, idx - halfWindow);
        const hi = Math.min(rrValues.length, idx + halfWindow + 1);
        return _median(rrValues.slice(lo, hi));
    }

    // =================================================================
    // ARTIFACT DETECTION — Adaptive Lipponen & Tarvainen (2019)
    // Matches dfa_core.py clean_rr_intervals() exactly
    // =================================================================

    /**
     * Adaptive threshold from local dRR distribution.
     * 3.32 * QD + Q1, floor 10 ms.
     * Matches dfa_core._compute_local_threshold().
     */
    function _adaptiveThreshold(drr, idx, halfWindow = 45) {
        const lo = Math.max(0, idx - halfWindow);
        const hi = Math.min(drr.length, idx + halfWindow + 1);
        const local = drr.slice(lo, hi).map(Math.abs);
        if (local.length < 5) return 100.0;
        const q1 = _percentile(local, 25);
        const q3 = _percentile(local, 75);
        return Math.max(3.32 * (q3 - q1) + q1, 10.0);
    }

    /**
     * Clean RR intervals with adaptive Lipponen & Tarvainen detection
     * and interpolation correction.
     * Matches dfa_core.clean_rr_intervals().
     *
     * @param {number[]} rrValues - raw RR intervals in ms
     * @returns {{ clean: number[], artifactPct: number }}
     */
    function cleanRRIntervals(rrValues) {
        const N = rrValues.length;
        if (N === 0) return { clean: [], artifactPct: 0 };

        const artifact = new Array(N).fill(false);

        // Step 1: Physiological bounds
        for (let i = 0; i < N; i++) {
            if (rrValues[i] < RR_MIN || rrValues[i] > RR_MAX) artifact[i] = true;
        }

        // Step 2: Adaptive dRR threshold (Threshold 1)
        const drr = [];
        for (let i = 1; i < N; i++) drr.push(rrValues[i] - rrValues[i - 1]);

        for (let i = 1; i < N; i++) {
            if (artifact[i]) continue;
            const thresh = _adaptiveThreshold(drr, Math.min(i - 1, drr.length - 1));
            if (Math.abs(drr[i - 1]) > thresh) artifact[i] = true;
        }

        // Step 3: Adaptive local median threshold (Threshold 2)
        for (let i = 0; i < N; i++) {
            if (artifact[i]) continue;
            const localMed = _localMedianRR(rrValues, i, 10);
            if (Math.abs(rrValues[i] - localMed) / localMed > 0.30) artifact[i] = true;
        }

        const artifactCount = artifact.filter(Boolean).length;
        const artifactPct = (artifactCount / N) * 100;

        // Step 4: Correct artifacts via interpolation from valid neighbours
        const clean = rrValues.slice();
        for (let idx = 0; idx < N; idx++) {
            if (!artifact[idx]) continue;
            let left = idx - 1;
            while (left >= 0 && artifact[left]) left--;
            let right = idx + 1;
            while (right < N && artifact[right]) right++;

            if (left >= 0 && right < N) {
                const span = right - left;
                const frac = span > 0 ? (idx - left) / span : 0.5;
                clean[idx] = clean[left] + frac * (rrValues[right] - clean[left]);
            } else if (left >= 0) {
                clean[idx] = clean[left];
            } else if (right < N) {
                clean[idx] = rrValues[right];
            }
        }

        return { clean, artifactPct };
    }

    // --- Simple artifact check (used only for the HRV display quick-filter) ---
    function _isSimpleArtifact(rr, localMedian) {
        if (rr < RR_MIN || rr > RR_MAX) return true;
        if (localMedian > 0 && Math.abs(rr - localMedian) / localMedian > 0.20) return true;
        return false;
    }

    function _getLocalMedianObj(arr, idx, windowSize = 5) {
        const start = Math.max(0, idx - windowSize);
        const end = Math.min(arr.length, idx + windowSize + 1);
        const win = arr.slice(start, end).map(r => r.rr).sort((a, b) => a - b);
        const mid = Math.floor(win.length / 2);
        return win.length % 2 ? win[mid] : (win[mid - 1] + win[mid]) / 2;
    }

    // =================================================================
    // HRV METRICS (rolling window)
    // =================================================================

    function computeHRVMetrics(rrArray) {
        if (rrArray.length < 4) return { rmssd: null, sdnn: null, artifactPct: 0 };

        const rrValues = rrArray.map(r => r.rr);
        const { clean, artifactPct } = cleanRRIntervals(rrValues);

        if (clean.length < 4) return { rmssd: null, sdnn: null, artifactPct };

        let sumSqDiff = 0;
        for (let i = 1; i < clean.length; i++) {
            const diff = clean[i] - clean[i - 1];
            sumSqDiff += diff * diff;
        }
        const rmssd = Math.sqrt(sumSqDiff / (clean.length - 1));

        const mean = clean.reduce((s, v) => s + v, 0) / clean.length;
        let sumSqDev = 0;
        for (const rr of clean) sumSqDev += (rr - mean) * (rr - mean);
        const sdnn = Math.sqrt(sumSqDev / (clean.length - 1));

        return {
            rmssd: Math.round(rmssd * 10) / 10,
            sdnn: Math.round(sdnn * 10) / 10,
            artifactPct: Math.round(artifactPct * 10) / 10,
        };
    }

    function getRecentHRV(windowSec = 30) {
        if (allRR.length === 0) return { rmssd: null, sdnn: null, artifactPct: 0 };
        const now = allRR[allRR.length - 1].time;
        const cutoff = now - windowSec * 1000;
        const recent = allRR.filter(r => r.time >= cutoff);
        return computeHRVMetrics(recent);
    }

    // =================================================================
    // CUBIC SPLINE INTERPOLATION (natural boundary)
    // Matches scipy.interpolate.CubicSpline
    // =================================================================

    function _cubicSplineEval(xs, ys, ts) {
        const n = xs.length;
        if (n < 2) return ts.map(() => ys[0] || 0);
        if (n === 2) {
            // Linear fallback
            const slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
            return ts.map(t => ys[0] + slope * (t - xs[0]));
        }

        const h = new Array(n - 1);
        const delta = new Array(n - 1);
        for (let i = 0; i < n - 1; i++) {
            h[i] = xs[i + 1] - xs[i];
            delta[i] = (ys[i + 1] - ys[i]) / (h[i] || 1e-12);
        }

        // Tridiagonal system for second derivatives (natural: c[0]=c[n-1]=0)
        const c = new Float64Array(n);
        if (n > 2) {
            const sz = n - 2;
            const mu = new Float64Array(sz);
            const z = new Float64Array(sz);

            let l = 2 * (h[0] + h[1]);
            mu[0] = h[1] / l;
            z[0] = 6 * (delta[1] - delta[0]) / l;

            for (let i = 1; i < sz; i++) {
                l = 2 * (h[i] + h[i + 1]) - h[i] * mu[i - 1];
                if (Math.abs(l) < 1e-15) l = 1e-15;
                mu[i] = h[i + 1] / l;
                z[i] = (6 * (delta[i + 1] - delta[i]) - h[i] * z[i - 1]) / l;
            }

            for (let i = sz - 2; i >= 0; i--) {
                c[i + 1] = z[i] - mu[i] * c[i + 2];
            }
            if (sz >= 1) c[sz] = z[sz - 1] - (sz >= 2 ? mu[sz - 1] * c[sz + 1] : 0);
        }

        const result = new Array(ts.length);
        for (let k = 0; k < ts.length; k++) {
            const t = ts[k];
            let lo = 0, hi = n - 2;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (xs[mid + 1] < t) lo = mid + 1; else hi = mid;
            }
            const i = lo;
            const dx = t - xs[i];
            const hh = h[i] || 1e-12;
            const a = ys[i];
            const b = delta[i] - hh * (2 * c[i] + c[i + 1]) / 6;
            const cc = c[i] / 2;
            const d = (c[i + 1] - c[i]) / (6 * hh);
            result[k] = a + dx * (b + dx * (cc + dx * d));
        }
        return result;
    }

    // =================================================================
    // 4 Hz RESAMPLING
    // Matches dfa_core._resample_rr_to_4hz()
    // =================================================================

    function _resampleRRto4Hz(rrMs) {
        if (rrMs.length < 4) return rrMs;

        // Cumulative time axis (seconds)
        const cumTimes = [0];
        for (let i = 0; i < rrMs.length; i++) {
            cumTimes.push(cumTimes[i] + rrMs[i] / 1000.0);
        }

        // Midpoint times for each interval
        const midTimes = new Array(rrMs.length);
        for (let i = 0; i < rrMs.length; i++) {
            midTimes[i] = (cumTimes[i] + cumTimes[i + 1]) / 2.0;
        }

        // Uniform 4Hz grid (250 ms steps)
        const tStart = midTimes[0];
        const tEnd = midTimes[midTimes.length - 1];
        const tUniform = [];
        for (let t = tStart; t < tEnd; t += 0.250) tUniform.push(t);

        if (tUniform.length < 4) return rrMs;

        const resampled = _cubicSplineEval(midTimes, rrMs, tUniform);
        return resampled.filter(v => isFinite(v));
    }

    // =================================================================
    // SMOOTHNESS PRIORS DETRENDING (Tarvainen et al. 2002)
    // Matches dfa_core.smoothness_priors_detrend(rr, lam=500)
    //
    // Solves H * trend = rr  where H = I + λ² D₂ᵀD₂
    // H is pentadiagonal — solved with banded Gaussian elimination.
    // =================================================================

    function _smoothnessPriorsDetrend(rr, lam = 500) {
        const N = rr.length;
        if (N < 4) {
            const m = rr.reduce((s, v) => s + v, 0) / N;
            return rr.map(v => v - m);
        }

        const L2 = lam * lam;

        // D₂ᵀD₂ produces a pentadiagonal pattern.
        // Main diagonal entries of D₂ᵀD₂:
        //   row 0: 1,  row 1: 5,  rows 2..N-3: 6,  row N-2: 5,  row N-1: 1
        // Off-diagonal ±1:
        //   row 0→1: -2,  rows 1..N-3→next: -4,  row N-2→N-1: -2
        // Off-diagonal ±2: all 1

        // Build pentadiagonal H = I + L² * D₂ᵀD₂
        // We store: a[i] = main, b[i] = upper-1 (H[i][i+1]), c[i] = upper-2 (H[i][i+2])
        // Lower diags are symmetric: lb[i] = H[i][i-1] = b[i-1], lc[i] = H[i][i-2] = c[i-2]
        const a = new Float64Array(N);
        const b = new Float64Array(N);  // b[i] = H[i][i+1]
        const c = new Float64Array(N);  // c[i] = H[i][i+2]

        // Main diagonal
        a[0]     = 1 + L2 * 1;
        a[1]     = 1 + L2 * 5;
        for (let i = 2; i < N - 2; i++) a[i] = 1 + L2 * 6;
        if (N > 3) a[N - 2] = 1 + L2 * 5;
        a[N - 1] = 1 + L2 * 1;

        // Upper off-1
        b[0] = L2 * (-2);
        for (let i = 1; i < N - 2; i++) b[i] = L2 * (-4);
        if (N > 2) b[N - 2] = L2 * (-2);

        // Upper off-2
        for (let i = 0; i < N - 2; i++) c[i] = L2 * 1;

        // Solve with forward elimination (pentadiagonal Gaussian)
        // Working copies
        const aa = Float64Array.from(a);
        const bb = Float64Array.from(b);
        const cc = Float64Array.from(c);
        const rhs = Float64Array.from(rr);

        // Lower diags (symmetric copy)
        const lb = Float64Array.from(b);  // lb[i] = H[i+1][i] = b[i]
        const lc = Float64Array.from(c);  // lc[i] = H[i+2][i] = c[i]

        // Forward sweep
        for (let i = 0; i < N; i++) {
            // Eliminate from row i-2
            if (i >= 2) {
                const m = lc[i - 2] / aa[i - 2];
                // lc[i-2] is the (i, i-2) entry
                // Row i -= m * Row (i-2)
                // affects columns i-2, i-1, i, i+1, i+2
                // (i, i-1) -= m * (i-2, i-1) = m * bb[i-2]   -- but (i-2,i-1) is bb[i-2]
                lb[i - 1] -= m * bb[i - 2];
                aa[i]     -= m * cc[i - 2];
                rhs[i]    -= m * rhs[i - 2];
                lc[i - 2]  = 0;
            }
            // Eliminate from row i-1
            if (i >= 1) {
                const m = lb[i - 1] / aa[i - 1];
                aa[i]    -= m * bb[i - 1];
                if (i < N - 1) bb[i] -= m * cc[i - 1];
                rhs[i]   -= m * rhs[i - 1];
                lb[i - 1] = 0;
            }
        }

        // Back substitution
        const trend = new Float64Array(N);
        for (let i = N - 1; i >= 0; i--) {
            let val = rhs[i];
            if (i + 1 < N) val -= bb[i] * trend[i + 1];
            if (i + 2 < N) val -= cc[i] * trend[i + 2];
            trend[i] = val / aa[i];
        }

        const detrended = new Array(N);
        for (let i = 0; i < N; i++) detrended[i] = rr[i] - trend[i];
        return detrended;
    }

    // =================================================================
    // DFA ALPHA1 — Full Kubios HRV pipeline
    // Matches dfa_core.dfa_alpha1() step-for-step:
    //   1. Artifact correction (adaptive Lipponen & Tarvainen)
    //   2. Cubic spline interpolation + 4 Hz resampling
    //   3. Smoothness priors detrending (λ=500)
    //   4. Integration (cumsum of mean-centred detrended series)
    //   5. DFA window analysis (scales n=4..16, per-segment RMS → mean)
    //   6. Alpha1 = slope of log(F(n)) vs log(n)
    // =================================================================

    function computeDFA(rrArray, minBox = 4, maxBox = 16) {
        if (rrArray.length < maxBox * 4) return null;

        // Step 0: Extract raw RR values and clean with adaptive detection
        const rawRR = rrArray.map(r => r.rr);
        const { clean: cleanedRR } = cleanRRIntervals(rawRR);

        if (cleanedRR.length < maxBox * 4) return null;

        // Step 1: Resample to uniform 4 Hz grid
        const resampled = _resampleRRto4Hz(cleanedRR);
        const Nrs = resampled.length;

        if (Nrs < maxBox * 4) return null;

        // Step 2: Smoothness priors detrending
        const detrended = _smoothnessPriorsDetrend(resampled, 500);

        // Step 3: Integration — cumulative sum of mean-centred detrended series
        const mean = detrended.reduce((s, v) => s + v, 0) / Nrs;
        const y = new Array(Nrs);
        y[0] = detrended[0] - mean;
        for (let i = 1; i < Nrs; i++) {
            y[i] = y[i - 1] + (detrended[i] - mean);
        }

        // Steps 4-5: DFA window analysis — per-segment RMS then average
        // Matches server: np.mean(rms_list)
        const logN = [];
        const logF = [];

        for (let n = minBox; n <= maxBox; n++) {
            const nSegments = Math.floor(Nrs / n);
            if (nSegments < 4) continue;

            let rmsSum = 0;
            for (let seg = 0; seg < nSegments; seg++) {
                const start = seg * n;
                // Linear detrend within segment (least-squares)
                let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
                for (let i = 0; i < n; i++) {
                    sumX += i;
                    sumY += y[start + i];
                    sumXY += i * y[start + i];
                    sumXX += i * i;
                }
                const denom = n * sumXX - sumX * sumX;
                if (denom === 0) continue;
                const slope = (n * sumXY - sumX * sumY) / denom;
                const intercept = (sumY - slope * sumX) / n;

                let residualSq = 0;
                for (let i = 0; i < n; i++) {
                    const trend = intercept + slope * i;
                    const diff = y[start + i] - trend;
                    residualSq += diff * diff;
                }
                rmsSum += Math.sqrt(residualSq / n);
            }

            const F = rmsSum / nSegments;
            if (F > 0) {
                logN.push(Math.log(n));
                logF.push(Math.log(F));
            }
        }

        // Step 6: Linear regression → alpha1
        // Server requires valid.sum() >= 4
        if (logN.length < 4) return null;

        let sX = 0, sY = 0, sXY = 0, sXX = 0;
        const len = logN.length;
        for (let i = 0; i < len; i++) {
            sX += logN[i];
            sY += logF[i];
            sXY += logN[i] * logF[i];
            sXX += logN[i] * logN[i];
        }
        const d = len * sXX - sX * sX;
        if (d === 0) return null;

        const alpha1 = (len * sXY - sX * sY) / d;
        return Math.round(alpha1 * 100) / 100;
    }

    function getRecentDFA(windowBeats = 120) {
        if (allRR.length < windowBeats) return null;
        const recent = allRR.slice(-windowBeats);
        return computeDFA(recent);
    }

    function getOverallArtifactRate() {
        if (allRR.length === 0) return 0;
        const rrValues = allRR.map(r => r.rr);
        const { artifactPct } = cleanRRIntervals(rrValues);
        return Math.round(artifactPct * 10) / 10;
    }

    // =================================================================
    // RECORDING
    // =================================================================

    function init(meta) {
        records = [];
        allRR = [];
        metadata = { ...meta, startTime: Date.now() };
    }

    function addRecord(record) {
        records.push({
            time: record.time,
            elapsed: record.elapsed,
            power: record.power ?? null,
            heartRate: record.heartRate ?? null,
            cadence: record.cadence ?? null,
            targetPower: record.targetPower ?? null,
            phase: record.phase ?? null,
            stageNum: record.stageNum ?? null,
        });
    }

    function addRRIntervals(intervals, timestamp) {
        for (const rr of intervals) {
            allRR.push({ time: timestamp, rr });
        }
    }

    // =================================================================
    // SERVER UPLOAD
    // =================================================================

    async function uploadToServer() {
        const token = localStorage.getItem('auth_token');
        if (!token) {
            console.warn('No auth token — skipping server upload');
            return { ok: false, message: 'Not authenticated' };
        }

        const summary = getSummary();
        const payload = { metadata, records, rrIntervals: allRR, summary };

        try {
            const resp = await fetch('/api/test/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify(payload),
            });
            const result = await resp.json();
            return { ok: resp.ok, ...result };
        } catch (err) {
            console.error('Upload failed:', err);
            return { ok: false, message: 'Network error — data saved locally via export.' };
        }
    }

    // =================================================================
    // EXPORT: CSV
    // =================================================================

    function exportCSV() {
        const lines = ['timestamp_ms,elapsed_s,power_w,heart_rate_bpm,cadence_rpm,target_power_w,phase,stage'];
        for (const r of records) {
            lines.push([
                r.time, r.elapsed.toFixed(1), r.power ?? '', r.heartRate ?? '',
                r.cadence ?? '', r.targetPower ?? '', r.phase ?? '', r.stageNum ?? '',
            ].join(','));
        }
        lines.push('');
        lines.push('--- RR INTERVALS ---');
        lines.push('timestamp_ms,rr_interval_ms');
        for (const rr of allRR) lines.push(`${rr.time},${rr.rr.toFixed(2)}`);
        return lines.join('\n');
    }

    // =================================================================
    // EXPORT: JSON
    // =================================================================

    function exportJSON() {
        return JSON.stringify({
            metadata: {
                ...metadata,
                endTime: Date.now(),
                totalRecords: records.length,
                totalRRIntervals: allRR.length,
                overallArtifactPct: getOverallArtifactRate(),
            },
            records,
            rrIntervals: allRR,
        }, null, 2);
    }

    // =================================================================
    // EXPORT: FIT
    // =================================================================

    function exportFIT() {
        const encoder = new FITEncoder();
        encoder.writeFileHeader();
        encoder.writeFileId(metadata.sport === 'bike' ? 2 : 1);
        encoder.writeSession(metadata.startTime, records.length > 0 ? records[records.length - 1].elapsed : 0);
        for (const r of records) encoder.writeRecord(r);
        const rrChunks = chunkArray(allRR.map(r => r.rr), 5);
        for (const chunk of rrChunks) encoder.writeHRV(chunk);
        encoder.writeCRC();
        return encoder.getBlob();
    }

    function chunkArray(arr, size) {
        const chunks = [];
        for (let i = 0; i < arr.length; i += size) chunks.push(arr.slice(i, i + size));
        return chunks;
    }

    class FITEncoder {
        constructor() { this.buffer = []; this.dataSize = 0; }
        writeByte(v) { this.buffer.push(v & 0xFF); this.dataSize++; }
        writeUint16(v) { this.writeByte(v); this.writeByte(v >> 8); }
        writeUint32(v) { this.writeByte(v); this.writeByte(v >> 8); this.writeByte(v >> 16); this.writeByte(v >> 24); }
        writeInt16(v) { this.writeUint16(v < 0 ? v + 65536 : v); }

        writeFileHeader() {
            this.writeByte(14); this.writeByte(0x20); this.writeUint16(0x08D0);
            this.writeUint32(0);
            this.writeByte(0x2E); this.writeByte(0x46); this.writeByte(0x49); this.writeByte(0x54);
            this.writeUint16(0x0000); this.dataSize = 0;
        }
        writeDefinition(localMsgType, globalMsgNum, fields) {
            this.writeByte(0x40 | localMsgType); this.writeByte(0); this.writeByte(0);
            this.writeUint16(globalMsgNum); this.writeByte(fields.length);
            for (const f of fields) { this.writeByte(f.num); this.writeByte(f.size); this.writeByte(f.type); }
        }
        writeFileId(sport) {
            this.writeDefinition(0, 0, [{ num: 0, size: 1, type: 0 },{ num: 1, size: 2, type: 132 },{ num: 4, size: 4, type: 134 }]);
            this.writeByte(0x00); this.writeByte(4); this.writeUint16(255); this.writeUint32(fitTimestamp(metadata.startTime));
        }
        writeSession(startTime, elapsedSec) {
            this.writeDefinition(1, 18, [{ num: 253, size: 4, type: 134 },{ num: 5, size: 1, type: 0 },{ num: 7, size: 4, type: 134 }]);
            this.writeByte(0x01); this.writeUint32(fitTimestamp(startTime));
            this.writeByte(metadata.sport === 'bike' ? 2 : 1); this.writeUint32(Math.round(elapsedSec * 1000));
        }
        writeRecord(r) {
            if (!this._recordDefWritten) {
                this.writeDefinition(2, 20, [{ num: 253, size: 4, type: 134 },{ num: 7, size: 2, type: 132 },{ num: 3, size: 1, type: 2 },{ num: 4, size: 1, type: 2 }]);
                this._recordDefWritten = true;
            }
            this.writeByte(0x02); this.writeUint32(fitTimestamp(r.time));
            this.writeUint16(r.power != null ? r.power : 0xFFFF);
            this.writeByte(r.heartRate != null ? r.heartRate : 0xFF);
            this.writeByte(r.cadence != null ? r.cadence : 0xFF);
        }
        writeHRV(rrValues) {
            if (!this._hrvDefWritten) {
                this.writeDefinition(3, 78, [{ num: 0, size: 10, type: 136 }]);
                this._hrvDefWritten = true;
            }
            this.writeByte(0x03);
            for (let i = 0; i < 5; i++) this.writeUint16(i < rrValues.length ? Math.round(rrValues[i]) : 0xFFFF);
        }
        writeCRC() {
            const d = this.dataSize;
            this.buffer[4] = d & 0xFF; this.buffer[5] = (d >> 8) & 0xFF;
            this.buffer[6] = (d >> 16) & 0xFF; this.buffer[7] = (d >> 24) & 0xFF;
            const crc = crc16(this.buffer);
            this.buffer.push(crc & 0xFF); this.buffer.push((crc >> 8) & 0xFF);
        }
        getBlob() { return new Blob([new Uint8Array(this.buffer)], { type: 'application/octet-stream' }); }
    }

    function fitTimestamp(jsTimestamp) {
        return Math.round((jsTimestamp - Date.UTC(1989, 11, 31, 0, 0, 0)) / 1000);
    }

    function crc16(bytes) {
        const t = [0x0000,0xCC01,0xD801,0x1400,0xF001,0x3C00,0x2800,0xE401,0xA001,0x6C00,0x7800,0xB401,0x5000,0x9C01,0x8801,0x4400];
        let crc = 0;
        for (const b of bytes) {
            let tmp = t[crc & 0xF]; crc = (crc >> 4) & 0x0FFF; crc = crc ^ tmp ^ t[b & 0xF];
            tmp = t[crc & 0xF]; crc = (crc >> 4) & 0x0FFF; crc = crc ^ tmp ^ t[(b >> 4) & 0xF];
        }
        return crc;
    }

    // =================================================================
    // DOWNLOAD & SUMMARY
    // =================================================================

    function download(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
        URL.revokeObjectURL(url);
    }

    function downloadCSV(athleteName, sport) {
        download(new Blob([exportCSV()], { type: 'text/csv' }), `${athleteName || 'athlete'}_${sport}_ramp_${_fmtDate()}.csv`);
    }
    function downloadJSON(athleteName, sport) {
        download(new Blob([exportJSON()], { type: 'application/json' }), `${athleteName || 'athlete'}_${sport}_ramp_${_fmtDate()}.json`);
    }
    function downloadFIT(athleteName, sport) {
        download(exportFIT(), `${athleteName || 'athlete'}_${sport}_ramp_${_fmtDate()}.fit`);
    }

    function _fmtDate() {
        const d = new Date();
        return `${d.getFullYear()}${(d.getMonth()+1).toString().padStart(2,'0')}${d.getDate().toString().padStart(2,'0')}_${d.getHours().toString().padStart(2,'0')}${d.getMinutes().toString().padStart(2,'0')}`;
    }

    function getSummary() {
        const peakPower = records.reduce((max, r) => Math.max(max, r.power ?? 0), 0);
        const peakHR = records.reduce((max, r) => Math.max(max, r.heartRate ?? 0), 0);
        const duration = records.length > 0 ? records[records.length - 1].elapsed : 0;
        const stagesCompleted = new Set(records.filter(r => r.phase === 'ramp' && r.stageNum).map(r => r.stageNum)).size;
        return { duration, peakPower, peakHR, totalRR: allRR.length, artifactPct: getOverallArtifactRate(), stagesCompleted, totalRecords: records.length };
    }

    // =================================================================
    // STAGE-LEVEL DFA — for early ramp termination
    // =================================================================

    /**
     * Compute DFA Alpha1 for RR intervals collected during a specific
     * time window (stage start to stage end).
     * Discards the first discardSec (default 60s) per Rogers protocol.
     *
     * @param {number} stageStartMs - stage start timestamp (ms since epoch)
     * @param {number} stageEndMs   - stage end timestamp (ms since epoch)
     * @param {number} discardSec   - seconds to discard from start (default 60)
     * @returns {number|null} DFA Alpha1 value or null if insufficient data
     */
    function getStageDFA(stageStartMs, stageEndMs, discardSec = 60) {
        const discardMs = discardSec * 1000;
        const analysisStart = stageStartMs + discardMs;

        // Collect RR intervals from the analysis window
        const stageRR = allRR.filter(r => r.time >= analysisStart && r.time <= stageEndMs);

        if (stageRR.length < 64) return null; // Need enough beats for DFA

        return computeDFA(stageRR);
    }

    /**
     * Record early ramp termination metadata.
     */
    let earlyRampEnd = null;

    function setEarlyRampEnd(info) {
        earlyRampEnd = info;
        // Store in metadata for server upload
        metadata.earlyRampEnd = info;
    }

    function getEarlyRampEnd() {
        return earlyRampEnd;
    }

    // =================================================================
    // PUBLIC API
    // =================================================================

    return {
        init, addRecord, addRRIntervals,
        getRecentHRV, getRecentDFA, computeDFA,
        getOverallArtifactRate, getSummary, uploadToServer,
        downloadCSV, downloadJSON, downloadFIT,
        getStageDFA, setEarlyRampEnd, getEarlyRampEnd,
        get records() { return records; },
        get allRR() { return allRR; },
    };
})();
