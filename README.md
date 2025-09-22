
# FIR_Bandpass_Filter_in_Verilog
A bandpass FIR filter (90–110 kHz at 1 MHz sampling) designed in Python, implemented in Verilog RTL, verified in ModelSim, and validated with Multisim.
=======
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
- Automated Python–Verilog comparison script: `ver_py_check_fixed.py`  
- Plotting utilities for time/frequency domain figures  
- Multisim input generation and validation (normalized to ±1 V, exported as `in_samples.txt`)  

---

## 📂 Repository Structure
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
| ├── fir_61_report.txt
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

yaml
Copy code

---

## ⚙️ Prerequisites
- Python **3.8+**  
- Python packages: `numpy`, `scipy`, `matplotlib`  
- ModelSim (or another Verilog simulator with command-line batch support)  
- *(Optional)* Multisim for circuit-level input generation  

Install Python packages:  
```bash
pip install numpy scipy matplotlib
🚀 How to Run
1) Design the FIR Filter (Python)
Generates quantized coefficients, golden reference output, debug files, and a report.

bash
Copy code
cd python
python fir.py
Outputs:

fir_61_coeffs.txt – integer coefficients

fir_61_coeffs_hex.txt – hex coefficients for Verilog

fir_61_pyref_file.txt – Python golden output

fir_61_report.txt – summary

2) Simulate Verilog (ModelSim)
Compile and run the Verilog testbench. Example:

tcl
Copy code
vlog ../src/FIR_FILTER.v ../src/FIR_TB.v
vsim -c work.FIR_TB -do "run -all; quit"
Outputs:

verilog_ref_file.txt – DUT outputs

verilog_dbg.txt – debug trace

3) Compare Python vs Verilog Outputs
bash
Copy code
python ver_py_check_fixed.py --py fir_61_pyref_file.txt --ver verilog_ref_file.txt
✅ If all matched:
No mismatches found. Files match for all compared samples.

4) Generate Plots
bash
Copy code
python plot_compare.py
python time_and_freq_compare.py
python adc_vs_fir_time.py
Plots are saved in results/ folder.

📊 Figures
Fig 1: FIR Filter Architecture Flow Chart

Fig 2: Multisim RF frontend circuit (ADC input generation)

Fig 3: Time-domain overlay (Python vs Verilog outputs)

Fig 4: Zoomed alignment view

Fig 5: Frequency-domain comparison (FFT)

Fig 6: Ideal DTFT vs Measured FIR response

Fig 7: Combined time + frequency comparison

Fig 8: ADC digital output vs FIR filtered output (stacked plot)

🔎 Notes on Input Preparation / ADC Scaling
Multisim outputs were within ±1 V

Normalized assuming ideal ADC (Vref = ±1 V)

Quantized to integers for in_samples.txt → consistent with fixed-point Verilog/Python

✅ Example Results
From fir_61_report.txt:

NTAPS = 385

QT = 23

Floating pre-center ≈ -0.4985 dB

Measured center ≈ -0.0685 dB

Passband ripple ≈ 0.99 dB

Worst stopband ≈ 41 dB

⚡ Challenges & Future Improvements
Challenges

Coefficient quantization increased ripple vs floating-point design

Verilog shift register required blocking assignments for correct MAC

Output alignment required flushing pipeline and robust comparison

Future Work

Optimize for FPGA (distributed arithmetic / FFT-based FIR)

Add pipelining for higher throughput

Add runtime coefficient reload (adaptive filtering)

Extend framework to other FIR types (LPF, HPF, multiband)

📜 License
This project is licensed under the MIT License. See LICENSE for details.

👩‍💻 Author
Rupashri R


yaml
---


