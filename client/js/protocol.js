/* ============================================
   Protocol Engine — DFA A1 Ramp Test
   Manages phases, timing, and power targets
   ============================================ */

const Protocol = (() => {

    // --- Default Protocol Definitions ---
    // Based on Tremayne Performance DFA A1 Ramp Protocol
    const BIKE_PROTOCOL = {
        warmup: { duration: 20 * 60, powerPct: 0.45 },      // 20 min @ 45% FTP
        ramp: {
            stages: 10,
            stageDuration: 3 * 60,                            // 3 min per stage
            // Exact intensities matching analysis engine protocol
            intensities: [0.60, 0.66, 0.71, 0.77, 0.82, 0.88, 0.93, 0.99, 1.04, 1.10],
        },
        recovery: { duration: 8 * 60, powerPct: 0.45 },      // 8 min @ 45% FTP
        mapRamp: {                                             // MAP ramp to failure
            stageDuration: 60,                                 // 1 min per step
            intensities: [1.10, 1.15, 1.20, 1.25, 1.30, 1.35],
        },
        cooldown: { duration: 10 * 60, powerPct: 0.45 },     // 10 min @ 45% FTP
    };

    const RUN_PROTOCOL = {
        warmup: { duration: 15 * 60, pacePct: 0.60 },        // 15 min @ 60% threshold pace
        ramp: {
            stages: 10,
            stageDuration: 3 * 60,
            // Exact intensities from ramp_analysis.py RUN_RAMP_INTENSITIES
            // Non-linear steps matching the analysis engine
            intensities: [0.70, 0.74, 0.78, 0.82, 0.86, 0.91, 0.95, 0.99, 1.03, 1.07],
        },
        recovery: { duration: 8 * 60, pacePct: 0.30 },       // 8 min @ 30%
        tte: { duration: 6 * 60, pacePct: 1.20 },            // TTE up to 6 min @ 120%
        cooldown: { duration: 10 * 60, pacePct: 0.60 },      // 10 min @ 60%
    };

    // --- Phase Enum ---
    const PHASE = {
        IDLE: 'idle',
        WARMUP: 'warmup',
        RAMP: 'ramp',
        RECOVERY: 'recovery',
        MAX_EFFORT: 'max-effort',
        TTE: 'tte',
        COOLDOWN: 'cooldown',
        COMPLETE: 'complete',
    };

    // --- Build Phase List ---
    function buildPhases(sport, thresholdValue) {
        const proto = sport === 'bike' ? BIKE_PROTOCOL : RUN_PROTOCOL;
        const phases = [];
        const unit = sport === 'bike' ? 'W' : 'min/km';

        // Warmup
        const warmupTarget = sport === 'bike'
            ? Math.round(thresholdValue * proto.warmup.powerPct)
            : thresholdValue / proto.warmup.pacePct;
        phases.push({
            id: PHASE.WARMUP,
            label: 'Warmup',
            duration: proto.warmup.duration,
            target: warmupTarget,
            targetDisplay: sport === 'bike' ? `${warmupTarget} W` : formatPaceWithKmh(warmupTarget),
            kmh: sport === 'run' ? paceToKmh(warmupTarget) : null,
        });

        // Ramp stages — both sports use exact intensities arrays
        const stageCount = proto.ramp.intensities
            ? proto.ramp.intensities.length
            : proto.ramp.stages;

        for (let i = 0; i < stageCount; i++) {
            const pct = proto.ramp.intensities
                ? proto.ramp.intensities[i]
                : proto.ramp.startPct + (proto.ramp.incrementPct * i);

            const target = sport === 'bike'
                ? Math.round(thresholdValue * pct)
                : thresholdValue / pct;
            phases.push({
                id: PHASE.RAMP,
                label: `Stage ${i + 1}`,
                stageNum: i + 1,
                duration: proto.ramp.stageDuration,
                target: target,
                targetDisplay: sport === 'bike' ? `${target} W` : formatPaceWithKmh(target),
                kmh: sport === 'run' ? paceToKmh(target) : null,
                pct: pct,
            });
        }

        // Recovery
        const recoveryTarget = sport === 'bike'
            ? Math.round(thresholdValue * proto.recovery.powerPct)
            : thresholdValue / proto.recovery.pacePct;
        phases.push({
            id: PHASE.RECOVERY,
            label: 'Recovery',
            duration: proto.recovery.duration,
            target: recoveryTarget,
            targetDisplay: sport === 'bike' ? `${recoveryTarget} W` : formatPaceWithKmh(recoveryTarget),
            kmh: sport === 'run' ? paceToKmh(recoveryTarget) : null,
        });

        // MAP Ramp (bike) or TTE (run)
        if (sport === 'bike') {
            const mapRamp = proto.mapRamp;
            for (let i = 0; i < mapRamp.intensities.length; i++) {
                const pct = mapRamp.intensities[i];
                const target = Math.round(thresholdValue * pct);
                phases.push({
                    id: PHASE.MAX_EFFORT,
                    label: `MAP ${Math.round(pct * 100)}%`,
                    stageNum: i + 1,
                    duration: mapRamp.stageDuration,
                    target: target,
                    targetDisplay: `${target} W`,
                    isMaxEffort: true,
                    isMapRamp: true,
                    pct: pct,
                });
            }
        } else {
            const tteTarget = thresholdValue / proto.tte.pacePct;
            phases.push({
                id: PHASE.TTE,
                label: 'TTE',
                duration: proto.tte.duration,
                target: tteTarget,
                targetDisplay: formatPaceWithKmh(tteTarget),
                kmh: paceToKmh(tteTarget),
                isTTE: true,
            });
        }

        // Cooldown
        const cooldownTarget = sport === 'bike'
            ? Math.round(thresholdValue * proto.cooldown.powerPct)
            : thresholdValue / proto.cooldown.pacePct;
        phases.push({
            id: PHASE.COOLDOWN,
            label: 'Cooldown',
            duration: proto.cooldown.duration,
            target: cooldownTarget,
            targetDisplay: sport === 'bike' ? `${cooldownTarget} W` : formatPaceWithKmh(cooldownTarget),
            kmh: sport === 'run' ? paceToKmh(cooldownTarget) : null,
        });

        return phases;
    }

    // --- Calculate total duration ---
    function totalDuration(phases) {
        return phases.reduce((sum, p) => sum + p.duration, 0);
    }

    // --- Get phase at elapsed time ---
    function getPhaseAt(phases, elapsedSec) {
        let cumulative = 0;
        for (let i = 0; i < phases.length; i++) {
            if (elapsedSec < cumulative + phases[i].duration) {
                return {
                    index: i,
                    phase: phases[i],
                    phaseElapsed: elapsedSec - cumulative,
                    phaseRemaining: phases[i].duration - (elapsedSec - cumulative),
                    totalProgress: elapsedSec / totalDuration(phases),
                };
            }
            cumulative += phases[i].duration;
        }
        // Past all phases
        return {
            index: phases.length - 1,
            phase: { id: PHASE.COMPLETE, label: 'Complete', duration: 0, target: 0 },
            phaseElapsed: 0,
            phaseRemaining: 0,
            totalProgress: 1,
        };
    }

    // --- Get protocol summary for preview ---
    function getProtocolSummary(sport) {
        const proto = sport === 'bike' ? BIKE_PROTOCOL : RUN_PROTOCOL;
        const segments = [];

        segments.push({
            id: 'warmup',
            label: 'Warmup',
            duration: proto.warmup.duration,
            color: 'warmup',
        });

        const rampStageCount = proto.ramp.intensities ? proto.ramp.intensities.length : proto.ramp.stages;
        const rampTotal = rampStageCount * proto.ramp.stageDuration;
        segments.push({
            id: 'ramp',
            label: `Ramp (${rampStageCount}×${proto.ramp.stageDuration / 60}min)`,
            duration: rampTotal,
            color: 'ramp',
        });

        segments.push({
            id: 'recovery',
            label: 'Recovery',
            duration: proto.recovery.duration,
            color: 'recovery',
        });

        if (sport === 'bike') {
            const mapTotal = proto.mapRamp.intensities.length * proto.mapRamp.stageDuration;
            segments.push({
                id: 'max-effort',
                label: 'MAP Ramp',
                duration: mapTotal,
                color: 'max-effort',
            });
        } else {
            segments.push({
                id: 'tte',
                label: 'TTE',
                duration: proto.tte.duration,
                color: 'max-effort',
            });
        }

        segments.push({
            id: 'cooldown',
            label: 'Cooldown',
            duration: proto.cooldown.duration,
            color: 'cooldown',
        });

        const total = segments.reduce((s, seg) => s + seg.duration, 0);
        segments.forEach(seg => seg.pct = seg.duration / total * 100);

        return { segments, totalDuration: total };
    }

    // --- Pace formatting helpers ---
    function parsePace(paceStr) {
        // "4:30" → 4.5 (minutes per km)
        const parts = paceStr.split(':');
        if (parts.length !== 2) return NaN;
        return parseInt(parts[0]) + parseInt(parts[1]) / 60;
    }

    function formatPace(minPerKm) {
        if (!isFinite(minPerKm) || minPerKm <= 0) return '--:--';
        let mins = Math.floor(minPerKm);
        let secs = Math.round((minPerKm - mins) * 60);
        if (secs === 60) { mins++; secs = 0; }
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function paceToKmh(minPerKm) {
        if (!isFinite(minPerKm) || minPerKm <= 0) return 0;
        return Math.round((60 / minPerKm) * 10) / 10; // rounded to 0.1
    }

    function formatPaceWithKmh(minPerKm) {
        const pace = formatPace(minPerKm);
        const kmh = paceToKmh(minPerKm);
        if (pace === '--:--') return '--:--';
        return `${pace} / ${kmh} km/h`;
    }

    function formatTime(totalSec) {
        const m = Math.floor(totalSec / 60);
        const s = Math.floor(totalSec % 60);
        return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    // --- Public API ---
    return {
        PHASE,
        BIKE_PROTOCOL,
        RUN_PROTOCOL,
        buildPhases,
        totalDuration,
        getPhaseAt,
        getProtocolSummary,
        parsePace,
        formatPace,
        paceToKmh,
        formatPaceWithKmh,
        formatTime,
    };
})();
