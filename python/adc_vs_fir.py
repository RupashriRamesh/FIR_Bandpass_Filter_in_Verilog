#!/usr/bin/env python3
"""
adc_vs_fir_time.py

Plot two stacked time-domain subplots:
 - Top: ADC digital output (in_samples.txt)   -- normalized view
 - Bottom: FIR filtered output (verilog_ref_file.txt) -- normalized view
Optional: overlay scaled FIR (dashed) to compare absolute amplitude.

Outputs: adc_vs_fir_time.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --------- User config ----------
IN_FILE = "in_samples.txt"
VER_FILE = "verilog_ref_file.txt"
FS = 1_000_000.0   # sampling frequency (Hz) -- used only for time axis units
OUT_PNG = "adc_vs_fir_time.png"
TIME_ZOOM_SAMPLES = None    # e.g. 800 to enable inset zoom; set None to disable
OVERLAY_SCALED_FIR = False  # set True to overlay scaled FIR dashed line on bottom plot
# --------------------------------

def load_int_file(fn):
    p = Path(fn)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {fn}")
    vals = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        try:
            vals.append(int(float(s.split()[0])))
        except:
            continue
    return np.array(vals, dtype=np.int64)

def norm(x):
    if x.size == 0:
        return x.astype(float)
    m = np.max(np.abs(x))
    if m == 0:
        return x.astype(float)
    return x.astype(float) / float(m)

def main():
    x = load_int_file(IN_FILE)
    y = load_int_file(VER_FILE)

    if len(x) == 0 or len(y) == 0:
        raise SystemExit("Input or Verilog output file is empty or missing numeric rows.")
    max_samples=int(Fs*0.1)
    L = min(L,max_samples)
    x = x[:L]
    y = y[:L]

    t = np.arange(L) / FS

    x_n = norm(x)
    y_n = norm(y)

    # compute scale factor (optional) to overlay FIR with ADC amplitude
    scale_factor = 1.0
    if OVERLAY_SCALED_FIR:
        # scale FIR to match ADC peak for visual overlay
        max_x = np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else 1.0
        max_y = np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else 1.0
        scale_factor = max_x / max_y

    # Plot
    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1]})

    # Top: ADC digital output
    ax1.plot(t, x_n, linewidth=0.8, label="ADC digital output (normalized)")
    ax1.set_title("ADC Digital Output (normalized)")
    ax1.set_ylabel("Normalized code")
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Bottom: FIR filtered output
    ax2.plot(t, y_n, linewidth=0.8, color='tab:orange', label="FIR output (normalized)")
    if OVERLAY_SCALED_FIR:
        ax2.plot(t, (y * scale_factor) / (np.max(np.abs(x)) if np.max(np.abs(x))!=0 else 1.0),
                 linestyle='--', linewidth=0.8, color='tab:red',
                 label=f"FIR scaled ×{scale_factor:.2f}")
    ax2.set_title("FIR Filtered Output (normalized)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Normalized value")
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, linestyle=':', alpha=0.4)

    # Optional zoom inset in bottom plot
    if TIME_ZOOM_SAMPLES and TIME_ZOOM_SAMPLES > 8 and L >= TIME_ZOOM_SAMPLES:
        axins = ax2.inset_axes([0.02, 0.55, 0.36, 0.4])  # relative coords
        tz = np.arange(TIME_ZOOM_SAMPLES) / FS
        axins.plot(tz, x_n[:TIME_ZOOM_SAMPLES], linewidth=0.7)
        axins.plot(tz, y_n[:TIME_ZOOM_SAMPLES], linewidth=0.7, alpha=0.9)
        axins.set_title("Zoom (first {} samples)".format(TIME_ZOOM_SAMPLES), fontsize=8)
        axins.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PNG}")

if __name__ == "__main__":
    main()
