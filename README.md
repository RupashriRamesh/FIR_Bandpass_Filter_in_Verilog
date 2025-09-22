# ADC System with RF Frontend and FIR Bandpass Filter

## 📌 Project Overview
This repository contains an end-to-end implementation and verification flow for a **Finite Impulse Response (FIR) bandpass filter**.  
The filter was designed using the **Remez algorithm in Python**, quantized to **Q1.23 fixed-point format**, implemented in **Verilog (direct convolution)**, and verified by comparing Verilog outputs against a Python golden model.  
Additional circuit-level validation was performed using **Multisim**.

---

## 🎯 Target Specifications
- **Passband:** 90–110 kHz  
- **Sampling Frequency (fs):** 1 MHz  
- **Default Taps (NTAPS):** 385  
- **Coefficient Quantization:** Q1.23  

---

## ✨ Features
- Parameterized FIR design (adjustable NTAPS and QBITS)  
- Python golden model for filter design and fixed-point simulation  
- Verilog RTL implementation: `FIR_FILTER.v` (direct convolution)  
- Testbench: `FIR_TB.v` for ModelSim simulation and file I/O  
- Automated Python–Verilog comparison: `ver_py_check_fixed.py`  
- Plotting utilities: `plot_compare.py`, `time_and_freq_compare.py`, `adc_vs_fir_time.py`  
- Multisim input generation and validation (normalized to ±1 V, exported as `in_samples.txt`)  


## 📂 Repository Structure

```
├── docs/ # Documentation & reports (PDFs, diagrams)
├── src/ # Verilog source
│ ├── FIR_FILTER.v
│ └── FIR_TB.v
├── python/ # Python scripts for design & verification
│ ├── fir.py
│ ├── ver_py_check_fixed.py
│ ├── plot_compare.py
│ └── adc_vs_fir_time.py
├── data/ # Coeffs, inputs and outputs
│ ├── fir_61_coeffs.txt
│ ├── fir_61_coeffs_hex.txt
│ ├── in_samples.txt
│ ├── fir_61_pyref_file.txt
│ ├── fir_61_report.txt
│ └── verilog_ref_file.txt
├── results/ # Generated PNGs and reports
│ ├── time_overlay.png
│ ├── time_zoom_overlay.png
│ ├── fft_overlay.png
│ ├── ideal_vs_measured.png
│ └── adc_vs_fir_time.png
├── .gitignore
├── README.md
└── LICENSE

```

## ⚙️ Prerequisites
- Python **3.8+**  
- Python packages: `numpy`, `scipy`, `matplotlib`  
- ModelSim (or another Verilog simulator with command-line support)  
- *(Optional)* Multisim for circuit-level input generation  

Install Python packages:
```bash
pip install numpy scipy matplotlib
```
## 🚀 How to Run
1. Design the FIR filter (Python)

Generates quantized coefficients, golden reference output, debug files, and a report
```
vlog ../src/FIR_FILTER.v ../src/FIR_TB.v
vsim -c work.FIR_TB -do "run -all; quit"
```
2. Outputs

verilog_ref_file.txt – Verilog output
verilog_dbg.txt – debug trace

3. Compare Python vs Verilog outputs
```
python ver_py_check_fixed.py --py fir_61_pyref_file.txt --ver verilog_ref_file.txt
```

If matched, the script prints:
No mismatches found. Files match for all compared samples.
python plot_compare.py
python time_and_freq_compare.py
python adc_vs_fir_time.py
All plots are saved in the results/ folder.

## 🔎 Notes on Input Preparation / ADC Scaling

Multisim outputs were within ±1 V.
They were normalized under an ideal ADC assumption (Vref = ±1 V) and quantized to integers for in_samples.txt, ensuring consistency with fixed-point Python/Verilog workflows.

## ✅ Example Results

From fir_61_report.txt:

NTAPS = 385

QT = 23

Floating-point pre-center ≈ −0.4985 dB

Measured center ≈ −0.0685 dB

Passband ripple ≈ 0.99 dB

Worst stopband attenuation ≈ 41 dB

## ⚡ Challenges & Future Improvements

### Challenges

Coefficient quantization increased ripple versus floating-point design
Verilog shift register required blocking assignments for correct MAC operations
Output alignment required pipeline flushing and robust comparison scripts

### Future Work

Optimize implementation for FPGA (distributed arithmetic / FFT-based methods)
Add pipeline stages for higher throughput
Support runtime coefficient reload (adaptive filtering)
Extend framework for other FIR types (low-pass, high-pass, multiband)

## 📜 License

This project is licensed under the MIT License. See LICENSE for details.

## 👩‍💻 Author

Rupashri R