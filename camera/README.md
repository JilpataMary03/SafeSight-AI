# Camera Tamper Detection System (Real-Time)

## 📌 Overview
This project implements a **real-time camera tamper detection system** using OpenCV.
It detects common camera tampering scenarios such as:

- Camera shaking
- Camera blacked out / covered
- Excessive blur
- Fog or dust obstruction

The system is designed for **CCTV and surveillance applications**, running efficiently on CPU.

---

## 🎯 Features
- 📷 **Camera Shake Detection**
  - Optical flow–based motion analysis
- ⚫ **Black / Covered Camera Detection**
  - Brightness, variance, and dark pixel ratio analysis
- 🌫️ **Fog / Dust Detection**
  - Contrast and saturation degradation detection
- 🔍 **Blur Detection**
  - Laplacian variance method
- ⏱️ **Temporal Validation**
  - Tamper is confirmed only after consecutive frames
- 📸 **Automatic Snapshot Capture**
  - Snapshot saved during tamper events with cooldown
- ⚡ **Real-Time Performance**
  - Optimized frame resizing for fast analysis

---

## 🧠 Tamper Detection Logic

| Tamper Type | Technique Used |
|------------|---------------|
| Camera Shake | Optical Flow (Lucas-Kanade) |
| Black Cover | Mean brightness + variance + dark ratio |
| Blur | Laplacian variance |
| Fog / Dust | Low contrast + low saturation |
| False Positive Control | Consecutive-frame validation |

---

## 🛠️ Tech Stack
- Python
- OpenCV
- NumPy

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python main.py
