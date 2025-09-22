#!/usr/bin/env python3
"""
plot_compare.py

Produce frequency-domain and time-domain verification plots comparing:
 - Python golden output (fir_61_pyref_file.txt)
 - Verilog DUT output (verilog_ref_file.txt)
 - Input (in_samples.txt)
 - Ideal DTFT of quantized coefficients (fir_61_coeffs.txt)

Outputs (PNG):
 - time_overlay.png         : time-domain overlay (full)
 - time_zoom_overlay.png    : time-domain zoom (first N_zoom samples)
 - fft_overlay.png          : FFT overlay (python vs verilog)
 - ideal_vs_measured.png    : ideal DTFT (coeffs) vs measured H(f)=Y/X
 - py_fft.png               : python-only FFT (for inspection)
 - ver_fft.png              : verilog-only FFT

Adjust filenames, QT and FS below if needed.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import sys

# ---------- user config ----------
IN_SAMPLES = "in_samples.txt"
PY_REF = "fir_61_pyref_file.txt"   # python golden outputs
VER_REF = "verilog_ref_file.txt"   # verilog outputs
COEFFS_INT = "fir_61_coeffs.txt"   # integer coefficients (one per line)
QT = 23                            # fractional bits used to quantize coefficients
FS = 1_000_000.0                   # sampling frequency
PB_LOW = 90e3
PB_HIGH = 110e3

# plotting params
NFFT_DISPLAY = 200000  # dense freq vector size for DTFT plots
N_ZOOM = 2000          # zoom samples for time-domain zoom plot
# ---------------------------------

def load_int_file(fn):
    p = Path(fn)
    if not p.exists():
        raise FileNotFoundError(fn)
    vals = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith('#'): continue
        try:
            # accept decimal or hex (0x..)
            if s.lower().startswith("0x"):
                vals.append(int(s,16))
            else:
                vals.append(int(s.split()[0],0))
        except:
            continue
    return np.array(vals, dtype=np.int64)

def load_float_file(fn):
    p = Path(fn)
    if not p.exists():
        raise FileNotFoundError(fn)
    vals = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith('#'): continue
        try:
            vals.append(float(s.split()[0]))
        except:
            continue
    return np.array(vals, dtype=float)

def load_int_or_decimal(fn):
    # some files contain small ints that are intended signed 32-bit, handle them
    arr = load_int_file(fn)
    return arr.astype(np.int64)

def align_series(x, y, max_lag=2000):
    """
    Align y to x by computing cross-correlation on a window and finding best lag.
    Returns (x_trimmed, y_trimmed). Keeps same length after trimming edges.
    This is robust to small delays; for large fixed offsets (like NTAPS-1) it's still fine.
    """
    L = min(len(x), len(y))
    if L == 0:
        return x, y
    # take a representative window near start (but not all-zeros)
    w = min(16384, L)
    sx = x[:w] - np.mean(x[:w])
    sy = y[:w] - np.mean(y[:w])
    # compute cross-correlation via FFT for speed
    n = 1 << int(np.ceil(np.log2(len(sx) + len(sy) - 1)))
    X = np.fft.rfft(sx, n=n)
    Y = np.fft.rfft(sy, n=n)
    cc = np.fft.irfft(X * np.conj(Y), n=n)
    # cc[k] corresponds to lag k (0..n-1) with circular; convert to linear lags
    # center at zero lag mapping:
    cc = np.concatenate((cc[-(w-1):], cc[:w]))  # lags -(w-1) .. (w-1)
    lags = np.arange(-(w-1),(w))
    # restrict search to reasonable lags
    idx = np.argmax(np.abs(cc[(lags >= -max_lag) & (lags <= max_lag)]))
    valid_lags = lags[(lags >= -max_lag) & (lags <= max_lag)]
    best_lag = int(valid_lags[idx])
    # if best_lag > 0 => x leads y by best_lag => y should be shifted right -> trim first best_lag from x
    if best_lag > 0:
        x2 = x[best_lag:]
        y2 = y[:len(x2)]
    elif best_lag < 0:
        lag = -best_lag
        y2 = y[lag:]
        x2 = x[:len(y2)]
    else:
        Lmin = min(len(x), len(y))
        x2 = x[:Lmin]
        y2 = y[:Lmin]
    # final trim to equal length
    Lf = min(len(x2), len(y2))
    return x2[:Lf], y2[:Lf]

def compute_measured_H(x, y, fs=FS, nfft=None):
    # windowed DFT method with Hann window for amplitude correctness
    L = min(len(x), len(y))
    if L < 4: return None
    if nfft is None:
        nfft = max(65536, 1 << int(np.ceil(np.log2(L))))
    win = np.hanning(L)
    coh = np.mean(win)
    X = np.fft.rfft(x * win, n=nfft) / (coh + 1e-30)
    Y = np.fft.rfft(y * win, n=nfft) / (coh + 1e-30)
    freqs = np.fft.rfftfreq(nfft, 1.0/fs)
    eps = 1e-16
    H = np.zeros_like(Y, dtype=complex)
    nz = np.abs(X) > eps
    H[nz] = Y[nz] / X[nz]
    Hdb = 20*np.log10(np.abs(H) + 1e-20)
    return freqs, H, Hdb

def dtft_from_coeffs(h_float, fs=FS, npoints=200000):
    freqs = np.linspace(0, fs/2.0, npoints)
    k = np.arange(len(h_float))
    # z = exp(-j*2*pi * f * k / fs)  => shape (npoints, len(h))
    # compute vectorized dot
    Z = np.exp(-2j * math.pi * np.outer(freqs, k) / fs)
    H = Z.dot(h_float)
    Hdb = 20*np.log10(np.abs(H) + 1e-20)
    return freqs, H, Hdb

def measure_passband_stats(freqs, Hdb, pb_low=PB_LOW, pb_high=PB_HIGH):
    mask = (freqs >= pb_low) & (freqs <= pb_high)
    if not np.any(mask):
        return None
    pb_vals = Hdb[mask]
    return {
        'center_db': float(np.interp((pb_low+pb_high)/2.0, freqs, Hdb)),
        'pb_min': float(pb_vals.min()),
        'pb_max': float(pb_vals.max()),
        'ripple': float(pb_vals.max() - pb_vals.min())
    }

def main():
    # load data
    print("Loading files...")
    x_in = load_float_file(IN_SAMPLES) if Path(IN_SAMPLES).exists() else None
    py = load_float_file(PY_REF)
    ver = load_float_file(VER_REF)
    coeffs_int = load_int_file(COEFFS_INT)

    # If input missing, try to infer from python ref (if python ref used synthesized input)
    if x_in is None:
        print("Input file not found; trying to infer input from python debug (not implemented). Exiting.")
        raise SystemExit(1)

    print(f"Lengths: input={len(x_in)} python={len(py)} verilog={len(ver)} coeffs={len(coeffs_int)}")

    # convert coefficients to float (Q format)
    h_float = coeffs_int.astype(float) / float(1<<QT)

    # Align python & verilog outputs with input:
    # Typical situation: python produced full-length conv output (len = len(x))
    # Verilog TB produced outputs delayed/trimmed by NTAPS-1; find best alignment
    py2, ver2 = align_series(py, ver, max_lag=2000)
    print(f"After simple alignment: py2_len={len(py2)} ver2_len={len(ver2)}")
    # Now align py2/ver2 to input x_in
    x_trim1, py3 = align_series(x_in, py2, max_lag=2000)
    x_trim2, ver3 = align_series(x_in, ver2, max_lag=2000)
    # choose length common
    L = min(len(x_trim1), len(py3), len(ver3))
    x = x_trim1[:L]
    py = py3[:L]
    ver = ver3[:L]
    print(f"Aligned lengths -> L = {L}")

    # Time-domain overlay (full)
    t = np.arange(L) / FS
    plt.figure(figsize=(10,4))
    plt.plot(t, x, label='Input (in_samples)', linewidth=0.8)
    plt.plot(t, py, label='Python ref', linewidth=0.6, alpha=0.9)
    plt.plot(t, ver, label='Verilog ref', linewidth=0.6, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time-domain: Input vs Python vs Verilog (overlay)")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig("time_overlay.png", dpi=200)
    print("Wrote time_overlay.png")

    # Zoomed time view (first N_ZOOM samples)
    Z = min(N_ZOOM, L)
    plt.figure(figsize=(10,4))
    tz = np.arange(Z) / FS
    plt.plot(tz, x[:Z], label='Input', linewidth=0.8)
    plt.plot(tz, py[:Z], label='Python', linewidth=0.8)
    plt.plot(tz, ver[:Z], label='Verilog', linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Time-domain zoom (first {Z} samples)")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig("time_zoom_overlay.png", dpi=200)
    print("Wrote time_zoom_overlay.png")

    # Frequency domain: measured H = Y/X for python and verilog
    freqs_py, H_py_complex, H_py_db = compute_measured_H(x, py, fs=FS, nfft=NFFT_DISPLAY)
    freqs_ver, H_ver_complex, H_ver_db = compute_measured_H(x, ver, fs=FS, nfft=NFFT_DISPLAY)

    # Save separate FFTs
    plt.figure(figsize=(9,4))
    plt.plot(freqs_py/1e3, H_py_db, linewidth=0.6)
    plt.xlabel("Freq (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Python measured H(f) = Y/X (FFT)")
    plt.xlim(0, FS/2/1e3)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout(); plt.savefig("py_fft.png", dpi=200)
    print("Wrote py_fft.png")

    plt.figure(figsize=(9,4))
    plt.plot(freqs_ver/1e3, H_ver_db, linewidth=0.6)
    plt.xlabel("Freq (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Verilog measured H(f) = Y/X (FFT)")
    plt.xlim(0, FS/2/1e3)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout(); plt.savefig("ver_fft.png", dpi=200)
    print("Wrote ver_fft.png")

    # Overlay python vs verilog measured
    plt.figure(figsize=(9,5))
    plt.plot(freqs_py/1e3, H_py_db, label='Python measured H', linewidth=0.8)
    plt.plot(freqs_ver/1e3, H_ver_db, label='Verilog measured H', linewidth=0.6, alpha=0.9)
    plt.xlabel("Freq (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Measured H(f) overlay: Python vs Verilog")
    plt.xlim(0, FS/2/1e3)
    plt.legend(fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout(); plt.savefig("fft_overlay.png", dpi=200)
    print("Wrote fft_overlay.png")

    # Ideal DTFT (from quantized int coeffs scaled by Q)
    freqs_dtft, H_dtft_complex, H_dtft_db = dtft_from_coeffs(h_float, fs=FS, npoints=NFFT_DISPLAY)

    # Compare ideal DTFT vs measured (use python measured for measured)
    plt.figure(figsize=(9,5))
    plt.plot(freqs_dtft/1e3, H_dtft_db, label='Ideal DTFT (coeffs)', linewidth=0.8)
    plt.plot(freqs_py/1e3, H_py_db, label='Measured H (python Y/X)', linewidth=0.7, alpha=0.9)
    plt.xlabel("Freq (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Ideal DTFT (from coeffs) vs Measured H (python)")
    plt.xlim(0, FS/2/1e3)
    plt.legend(fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout(); plt.savefig("ideal_vs_measured.png", dpi=200)
    print("Wrote ideal_vs_measured.png")

    # Print passband/stopband stats numeric (python-measured and ideal)
    py_stats = measure_passband_stats(freqs_py, H_py_db, PB_LOW, PB_HIGH)
    ver_stats = measure_passband_stats(freqs_ver, H_ver_db, PB_LOW, PB_HIGH)
    ideal_stats = measure_passband_stats(freqs_dtft, H_dtft_db, PB_LOW, PB_HIGH)
    print("\nPassband/Stopband summary (center @ 100 kHz):")
    if py_stats:
        print("Python-measured: center={center_db:.3f} dB, min={pb_min:.3f} dB, max={pb_max:.3f} dB, ripple={ripple:.3f} dB".format(**py_stats))
    if ver_stats:
        print("Verilog-measured: center={center_db:.3f} dB, min={pb_min:.3f} dB, max={pb_max:.3f} dB, ripple={ripple:.3f} dB".format(**ver_stats))
    if ideal_stats:
        print("Ideal DTFT: center={center_db:.3f} dB, min={pb_min:.3f} dB, max={pb_max:.3f} dB, ripple={ripple:.3f} dB".format(**ideal_stats))

    # Stopband numeric: worst stopband (outside guard)
    guard_low = PB_LOW - (PB_LOW * 0.1)
    guard_high = PB_HIGH + (PB_HIGH * 0.1)
    sb_mask_py = ((freqs_py <= guard_low) | (freqs_py >= guard_high))
    worst_stop_py = float(np.max(H_py_db[sb_mask_py])) if np.any(sb_mask_py) else None
    print("Python stopband H(f) = {:.3f} dB (attenuation = {:.3f} dB)".format(worst_stop_py, -worst_stop_py))

    # done
    print("\nAll plots written. Inspect PNGs: time_overlay.png, time_zoom_overlay.png, fft_overlay.png, ideal_vs_measured.png")

if __name__ == "__main__":
    main()
