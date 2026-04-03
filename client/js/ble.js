/* ============================================
   BLE Module — Auuki-pattern Bluetooth Manager
   Handles: Smart Trainer (FTMS), HR Monitor (HRS)
   ============================================ */

const BLE = (() => {
    // --- GATT Service & Characteristic UUIDs ---
    const UUID = {
        // Fitness Machine Service (FTMS)
        ftms: {
            service: 0x1826,
            indoorBikeData: 0x2AD2,
            controlPoint: 0x2AD9,
            machineFeature: 0x2ACC,
            supportedPowerRange: 0x2AD8,
            machineStatus: 0x2ADA,
        },
        // Heart Rate Service (HRS)
        hrs: {
            service: 0x180D,
            measurement: 0x2A37,
        },
        // Cycling Power Service (CPS) — fallback
        cps: {
            service: 0x1818,
            measurement: 0x2A63,
            controlPoint: 0x2A66,
        },
        // Tacx FE-C over BLE
        fec: {
            service: '6e40fec1-b5a3-f393-e0a9-e50e24dcca9e',
            tx: '6e40fec2-b5a3-f393-e0a9-e50e24dcca9e',
            rx: '6e40fec3-b5a3-f393-e0a9-e50e24dcca9e',
        },
    };

    // --- State ---
    let trainerDevice = null;
    let trainerServer = null;
    let trainerControlPoint = null;
    let trainerProtocol = null; // 'ftms' | 'fec'

    let hrmDevice = null;
    let hrmServer = null;

    let onTrainerData = null;
    let onHRMData = null;
    let onConnectionChange = null;

    // --- Helpers ---
    function isAvailable() {
        return !!navigator.bluetooth;
    }

    function nthBit(value, n) {
        return (value >> n) & 1;
    }

    // --- FTMS Indoor Bike Data Parser ---
    function parseIndoorBikeData(dataView) {
        const flags = dataView.getUint16(0, true);
        let offset = 2;
        const result = {};

        // Bit 0: More Data (0 = speed present)
        if (!nthBit(flags, 0)) {
            result.speed = dataView.getUint16(offset, true) * 0.01; // km/h
            offset += 2;
        }
        // Bit 1: Average Speed
        if (nthBit(flags, 1)) {
            result.avgSpeed = dataView.getUint16(offset, true) * 0.01;
            offset += 2;
        }
        // Bit 2: Instantaneous Cadence
        if (nthBit(flags, 2)) {
            result.cadence = dataView.getUint16(offset, true) * 0.5; // rpm
            offset += 2;
        }
        // Bit 3: Average Cadence
        if (nthBit(flags, 3)) {
            result.avgCadence = dataView.getUint16(offset, true) * 0.5;
            offset += 2;
        }
        // Bit 4: Total Distance (3 bytes)
        if (nthBit(flags, 4)) {
            result.distance = dataView.getUint16(offset, true) + (dataView.getUint8(offset + 2) << 16);
            offset += 3;
        }
        // Bit 5: Resistance Level
        if (nthBit(flags, 5)) {
            result.resistance = dataView.getInt16(offset, true);
            offset += 2;
        }
        // Bit 6: Instantaneous Power
        if (nthBit(flags, 6)) {
            result.power = dataView.getInt16(offset, true); // watts
            offset += 2;
        }
        // Bit 7: Average Power
        if (nthBit(flags, 7)) {
            result.avgPower = dataView.getInt16(offset, true);
            offset += 2;
        }
        // Bit 8-9: Expended Energy
        if (nthBit(flags, 8)) {
            result.totalEnergy = dataView.getUint16(offset, true);
            result.energyPerHour = dataView.getUint16(offset + 2, true);
            result.energyPerMinute = dataView.getUint8(offset + 4);
            offset += 5;
        }
        // Bit 9: Heart Rate
        if (nthBit(flags, 9)) {
            result.heartRate = dataView.getUint8(offset);
            offset += 1;
        }

        return result;
    }

    // --- Heart Rate Measurement Parser ---
    function parseHeartRateMeasurement(dataView) {
        const flags = dataView.getUint8(0);
        let offset = 1;
        const result = {};

        // Bit 0: HR format (0 = uint8, 1 = uint16)
        if (nthBit(flags, 0)) {
            result.heartRate = dataView.getUint16(offset, true);
            offset += 2;
        } else {
            result.heartRate = dataView.getUint8(offset);
            offset += 1;
        }

        // Bit 1: Sensor contact status supported
        result.sensorContactSupported = !!nthBit(flags, 2);
        result.sensorContactStatus = !!nthBit(flags, 1);

        // Bit 3: Energy Expended
        if (nthBit(flags, 3)) {
            result.energyExpended = dataView.getUint16(offset, true);
            offset += 2;
        }

        // Bit 4: RR-Interval present
        result.rrIntervals = [];
        if (nthBit(flags, 4)) {
            while (offset + 1 < dataView.byteLength) {
                const rawRR = dataView.getUint16(offset, true);
                // RR intervals in 1/1024 second resolution → convert to ms
                result.rrIntervals.push(rawRR / 1024 * 1000);
                offset += 2;
            }
        }

        return result;
    }

    // --- FTMS Control Point ---
    async function ftmsRequestControl() {
        if (!trainerControlPoint) return;
        const buf = new ArrayBuffer(1);
        new DataView(buf).setUint8(0, 0x00); // Request Control
        await trainerControlPoint.writeValue(buf);
    }

    async function ftmsReset() {
        if (!trainerControlPoint) return;
        const buf = new ArrayBuffer(1);
        new DataView(buf).setUint8(0, 0x01); // Reset
        await trainerControlPoint.writeValue(buf);
    }

    async function ftmsSetTargetPower(watts) {
        if (!trainerControlPoint) return;
        watts = Math.max(0, Math.min(65534, Math.round(watts)));
        const buf = new ArrayBuffer(3);
        const dv = new DataView(buf);
        dv.setUint8(0, 0x05); // Set Target Power opcode
        dv.setInt16(1, watts, true);
        await trainerControlPoint.writeValue(buf);
    }

    async function ftmsStartOrResume() {
        if (!trainerControlPoint) return;
        const buf = new ArrayBuffer(1);
        new DataView(buf).setUint8(0, 0x07); // Start or Resume
        await trainerControlPoint.writeValue(buf);
    }

    async function ftmsStop() {
        if (!trainerControlPoint) return;
        const buf = new ArrayBuffer(2);
        const dv = new DataView(buf);
        dv.setUint8(0, 0x08); // Stop or Pause
        dv.setUint8(1, 0x01); // Stop
        await trainerControlPoint.writeValue(buf);
    }

    // --- FE-C Control ---
    let fecTxCharacteristic = null;

    function fecChecksum(bytes) {
        let xor = 0;
        for (let i = 0; i < bytes.length; i++) xor ^= bytes[i];
        return xor;
    }

    async function fecSetTargetPower(watts) {
        if (!fecTxCharacteristic) return;
        watts = Math.max(0, Math.min(4093.75, watts));
        const raw = Math.round(watts * 4); // 0.25W resolution
        const page = [
            0xA4, 0x09, 0x4F, 0x05, // sync, length, msg id, channel
            0x31, // data page 49 (Target Power)
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // reserved
            raw & 0xFF, (raw >> 8) & 0xFF,
        ];
        page.push(fecChecksum(page));
        await fecTxCharacteristic.writeValue(new Uint8Array(page));
    }

    // --- Connect Trainer ---
    async function connectTrainer() {
        if (!isAvailable()) throw new Error('Web Bluetooth not available');

        // Clean up any stale connection before starting a new one
        if (trainerDevice && trainerDevice.gatt.connected) {
            try { trainerDevice.gatt.disconnect(); } catch (_) {}
        }
        trainerDevice = null;
        trainerServer = null;
        trainerControlPoint = null;
        fecTxCharacteristic = null;

        // Use acceptAllDevices for maximum compatibility with iOS BLE browsers.
        // Bluefy can reject specific service UUID filters (especially custom UUIDs).
        // After connecting, we detect the protocol (FTMS or FE-C) automatically.
        const device = await navigator.bluetooth.requestDevice({
            acceptAllDevices: true,
            optionalServices: [UUID.ftms.service, UUID.fec.service, UUID.cps.service],
        });

        device.addEventListener('gattserverdisconnected', () => {
            trainerDevice = null;
            trainerServer = null;
            trainerControlPoint = null;
            fecTxCharacteristic = null;
            if (onConnectionChange) onConnectionChange('trainer', false);
        });

        trainerDevice = device;
        trainerServer = await device.gatt.connect();

        // Try FTMS first
        try {
            const ftmsService = await trainerServer.getPrimaryService(UUID.ftms.service);

            // Subscribe to Indoor Bike Data
            const bikeData = await ftmsService.getCharacteristic(UUID.ftms.indoorBikeData);
            bikeData.addEventListener('characteristicvaluechanged', (e) => {
                const parsed = parseIndoorBikeData(e.target.value);
                if (onTrainerData) onTrainerData(parsed);
            });
            await bikeData.startNotifications();

            // Get control point
            trainerControlPoint = await ftmsService.getCharacteristic(UUID.ftms.controlPoint);

            // Enable indications on control point
            trainerControlPoint.addEventListener('characteristicvaluechanged', () => {});
            await trainerControlPoint.startNotifications();

            // Request control & start
            await ftmsRequestControl();
            await ftmsStartOrResume();

            trainerProtocol = 'ftms';
        } catch {
            // Fall back to FE-C
            try {
                const fecService = await trainerServer.getPrimaryService(UUID.fec.service);
                const rxChar = await fecService.getCharacteristic(UUID.fec.rx);
                fecTxCharacteristic = await fecService.getCharacteristic(UUID.fec.tx);

                rxChar.addEventListener('characteristicvaluechanged', (e) => {
                    const dv = e.target.value;
                    // Parse FE-C general data page 25
                    if (dv.byteLength >= 12 && dv.getUint8(4) === 0x19) {
                        const power = dv.getUint16(10, true) & 0x0FFF;
                        const cadence = dv.getUint8(6);
                        if (onTrainerData) onTrainerData({ power, cadence });
                    }
                });
                await rxChar.startNotifications();

                trainerProtocol = 'fec';
            } catch (e2) {
                throw new Error('Trainer does not support FTMS or FE-C: ' + e2.message);
            }
        }

        if (onConnectionChange) onConnectionChange('trainer', true, device.name);
        return device.name || 'Smart Trainer';
    }

    // --- Connect HR Monitor ---
    async function connectHRM() {
        if (!isAvailable()) throw new Error('Web Bluetooth not available');

        const device = await navigator.bluetooth.requestDevice({
            filters: [{ services: [UUID.hrs.service] }],
        });

        device.addEventListener('gattserverdisconnected', () => {
            hrmDevice = null;
            hrmServer = null;
            if (onConnectionChange) onConnectionChange('hrm', false);
        });

        hrmDevice = device;
        hrmServer = await device.gatt.connect();

        const hrsService = await hrmServer.getPrimaryService(UUID.hrs.service);
        const hrMeasurement = await hrsService.getCharacteristic(UUID.hrs.measurement);

        hrMeasurement.addEventListener('characteristicvaluechanged', (e) => {
            const parsed = parseHeartRateMeasurement(e.target.value);
            if (onHRMData) onHRMData(parsed);
        });
        await hrMeasurement.startNotifications();

        if (onConnectionChange) onConnectionChange('hrm', true, device.name);
        return device.name || 'HR Monitor';
    }

    // --- Set ERG Power Target ---
    async function setTargetPower(watts) {
        if (trainerProtocol === 'ftms') {
            await ftmsSetTargetPower(watts);
        } else if (trainerProtocol === 'fec') {
            await fecSetTargetPower(watts);
        }
    }

    // --- Disconnect ---
    function disconnectTrainer() {
        if (trainerDevice && trainerDevice.gatt.connected) {
            trainerDevice.gatt.disconnect();
        }
        trainerDevice = null;
        trainerServer = null;
        trainerControlPoint = null;
        fecTxCharacteristic = null;
    }

    function disconnectHRM() {
        if (hrmDevice && hrmDevice.gatt.connected) {
            hrmDevice.gatt.disconnect();
        }
        hrmDevice = null;
        hrmServer = null;
    }

    function disconnectAll() {
        disconnectTrainer();
        disconnectHRM();
    }

    // --- Status ---
    function isTrainerConnected() {
        return trainerDevice?.gatt?.connected ?? false;
    }

    function isHRMConnected() {
        return hrmDevice?.gatt?.connected ?? false;
    }

    // --- Public API ---
    return {
        isAvailable,
        connectTrainer,
        connectHRM,
        setTargetPower,
        disconnectTrainer,
        disconnectHRM,
        disconnectAll,
        isTrainerConnected,
        isHRMConnected,

        set onTrainerData(fn) { onTrainerData = fn; },
        set onHRMData(fn) { onHRMData = fn; },
        set onConnectionChange(fn) { onConnectionChange = fn; },
    };
})();
