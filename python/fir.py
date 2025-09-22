#!/usr/bin/env python3
"""
fir.py — patched to use NTAPS = 181 (single-shot)
Design & test bandpass 90-110 kHz @ fs=1 MHz, quantize Q1.23, simulate fixed-point.
"""
import sys, math
from pathlib import Path
import numpy as np
try:
    from scipy import signal
except Exception:
    print("ERROR: scipy is required. Install with: pip install scipy")
    raise SystemExit(1)

FS = 1_000_000.0
PB_LOW = 90_000.0
PB_HIGH = 110_000.0
PB_CENTER = (PB_LOW + PB_HIGH) / 2.0

NTAPS = 385  
QT = 23
COEFF_SCALE = 1 << QT
TRANSITION = 8_000.0
IN_SAMPLES_FN = "in_samples.txt"
OUT_COEFF_INT = "fir_61_coeffs.txt"
OUT_COEFF_HEX = "fir_61_coeffs_hex.txt"
OUT_REF = "fir_61_pyref_file.txt"
OUT_DBG = "fir_61_pydbg.txt"
OUT_REPORT = "fir_61_report.txt"
OUT_MIN = -(1 << 31)
OUT_MAX = (1 << 31) - 1
SYNTH_LEN = 200_000
TONE_AMP = 10000
NOISE_AMP = 200.0

def read_in_samples(fn=IN_SAMPLES_FN):
    p = Path(fn)
    if not p.exists():
        return None
    data=[]
    for ln in p.read_text().splitlines():
        s=ln.strip()
        if not s or s.startswith('#'): continue
        try:
            data.append(int(float(s.split()[0])))
        except:
            try: data.append(int(float(s)))
            except: data.append(0)
    return np.array(data, dtype=np.int64)

def synth_signal(fs=FS, length=SYNTH_LEN, tone_freq=PB_CENTER, tone_amp=TONE_AMP, noise_amp=NOISE_AMP):
    t = np.arange(length)/fs
    tone = tone_amp * np.sin(2*math.pi*tone_freq*t)
    rng = np.random.default_rng(123456)
    noise = noise_amp * rng.standard_normal(length)
    x = tone + noise
    return np.round(x).astype(np.int64)

def design_remez_bandpass(ntaps, fs, f1, f2):
    nyq = fs/2.0
    margin = max(2000.0, (f2-f1)*0.25)
    bands = [0.0, (f1 - margin)/nyq, f1/nyq, f2/nyq, (f2 + margin)/nyq, 1.0]
    bands = [max(0.0, min(1.0, b)) for b in bands]
    desired=[0,1,0]
    weights=[20.0, 1.0, 20.0]
    return signal.remez(numtaps=ntaps, bands=bands, desired=desired, weight=weights, fs=2)

def normalize_center(h, fs=FS, center=PB_CENTER):
    w = 2.0 * math.pi * center / fs
    z = complex(math.cos(w), math.sin(w))
    exps = z ** -np.arange(len(h))
    Hc = np.dot(h, exps)
    mag = abs(Hc)
    if mag == 0: return h, 0.0
    return h * (1.0/mag), 20*math.log10(mag)

def quantize_coeffs(h, qbits=QT):
    scale = 1 << qbits
    return np.round(h * scale).astype(np.int64)

def coeffs_to_hex(q):
    return ["{:08X}".format(int(v) & 0xFFFFFFFF) for v in q]

def fixed_point_filter(x, qcoeff, qbits):
    conv = np.convolve(x.astype(np.int64), qcoeff.astype(np.int64))
    M = len(qcoeff); L = len(x); offset = M - 1
    conv_trim = conv[offset: offset+L]
    rnd = 1 << (qbits - 1) if qbits>0 else 0
    y = (conv_trim + rnd) >> qbits
    return np.clip(y, OUT_MIN, OUT_MAX).astype(np.int64)

def measure_response(x, y, fs=FS, pb_low=PB_LOW, pb_high=PB_HIGH, center=PB_CENTER):
    L = len(x)
    if L < 1024: return None
    nfft = max(65536, 1 << int(np.ceil(np.log2(L))))
    win = np.hanning(L); coh = np.mean(win)
    X = np.fft.rfft(np.pad(x*win, (0, nfft-L))) / (coh + 1e-30)
    Y = np.fft.rfft(np.pad(y*win, (0, nfft-L))) / (coh + 1e-30)
    freqs = np.fft.rfftfreq(nfft, 1.0/fs)
    eps = 1e-16
    H = np.zeros_like(Y, dtype=complex)
    nz = np.abs(X) > eps
    H[nz] = Y[nz] / X[nz]
    Hdb = 20*np.log10(np.abs(H) + 1e-20)
    pb_mask = (freqs >= pb_low) & (freqs <= pb_high) & nz
    if not np.any(pb_mask): return None
    pb_vals = Hdb[pb_mask]
    pb_min = float(np.min(pb_vals)); pb_max = float(np.max(pb_vals)); pb_ripple = pb_max - pb_min
    center_gain = float(np.interp(center, freqs, Hdb))
    guard_low = max(0.0, pb_low - TRANSITION); guard_high = min(fs/2.0, pb_high + TRANSITION)
    sb_mask = ((freqs <= guard_low) | (freqs >= guard_high)) & nz
    sb_vals = Hdb[sb_mask] if np.any(sb_mask) else np.array([])
    worst_stop_db = float(np.max(sb_vals)) if sb_vals.size>0 else None
    return {'center_db':center_gain, 'pb_min':pb_min, 'pb_max':pb_max, 'pb_ripple':pb_ripple, 'worst_stop_db':worst_stop_db}

def main():
    print(f"fir.py — NTAPS={NTAPS} test")
    xin = read_in_samples()
    if xin is None:
        print("in_samples.txt not found -> synthesizing test vector.")
        xin = synth_signal()
        Path("in_samples_synth.txt").write_text("\n".join(str(int(v)) for v in xin))
    else:
        print(f"Loaded in_samples.txt length {len(xin)}")
    print("Designing remez...")
    taps = design_remez_bandpass(NTAPS, FS, PB_LOW, PB_HIGH)
    taps_s, orig_center_db = normalize_center(taps, FS, PB_CENTER)
    print(f"Floating pre-scale center db = {orig_center_db:.4f}")
    qcoeff = quantize_coeffs(taps_s, QT)
    hexs = coeffs_to_hex(qcoeff)
    Path(OUT_COEFF_INT).write_text("\n".join(str(int(v)) for v in qcoeff))
    Path(OUT_COEFF_HEX).write_text("\n".join(hexs))
    y = fixed_point_filter(xin, qcoeff, QT)
    Path(OUT_REF).write_text("\n".join(str(int(v)) for v in y))
    dbg = ["# coeff_int_first32"] + [str(int(v)) for v in qcoeff[:32]] + ["# coeff_hex_first32"] + hexs[:32] + ["# out_first256"] + [str(int(v)) for v in y[:256]]
    Path(OUT_DBG).write_text("\n".join(dbg))
    meas = measure_response(xin.astype(float), y.astype(float))
    if meas is None:
        print("Measurement failed.")
        return
    center = meas['center_db']; ripple = meas['pb_ripple']; worst_stop = meas['worst_stop_db']
    print("\nMeasurement:")
    print(f" center = {center:.4f} dB")
    print(f" passband min = {meas['pb_min']:.4f} dB, max = {meas['pb_max']:.4f} dB, ripple = {ripple:.4f} dB")
    if worst_stop is not None:
        print(f" worst stopband = {worst_stop:.4f} dB (atten = {-worst_stop:.4f} dB)")
    rpt = [
        f"NTAPS={NTAPS}",
        f"QT={QT}",
        f"Floating_pre_center_db={orig_center_db:.6f}",
        f"Measured_center_db={center:.6f}",
        f"pass_min={meas['pb_min']:.6f}",
        f"pass_max={meas['pb_max']:.6f}",
        f"ripple={ripple:.6f}",
        f"worst_stop={worst_stop:.6f}",
    ]
    Path(OUT_REPORT).write_text("\n".join(rpt))
    print(f"\nWrote report {OUT_REPORT}")

if __name__=="__main__":
    main()
