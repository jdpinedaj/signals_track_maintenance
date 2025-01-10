<!-- pandoc README.md -s -o README.docx -->

# Signal processing for railway track maintenance prediction.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

<hr>

The aim of this project is to predict the maintenance needs of railway tracks using a novel dual-methodology that combines clustering-based and distance-based approaches for enhanced anomaly detection. This project utilizes accelerometer data from in-service locomotives provided by Metro Valencia, a Spanish railway company. The vibration signals, collected through sensors placed on axle boxes, are analyzed to monitor the condition of railway tracks. The integrated methodology improves predictive maintenance by ensuring continuous and scalable monitoring of track conditions, offering actionable insights for optimizing maintenance schedules.

---

## Highlights

- Novel dual-methodology for railway track anomaly detection using accelerometers.
- Combines clustering-based detection (PCA and MiniBatch KMeans) and distance-from-mean spectrum methods.
- Utilizes Short-Term Fourier Transform (STFT) for spectral analysis of vertical and lateral accelerations.
- Effective in identifying isolated and subtle irregularities in track conditions.
- Adaptable to various railway systems without requiring labeled datasets.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Analysis and Methodologies](#analysis-and-methodologies)
4. [Files](#files)
5. [Parameters](#parameters)
6. [Results and Insights](#results-and-insights)
7. [Future Work](#future-work)
8. [TODOs](#todos)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview

Ensuring the safety and reliability of railway systems necessitates continuous monitoring of track conditions to detect irregularities that could compromise performance. This project proposes a dual-methodology approach combining clustering-based and distance-based anomaly detection techniques to analyze vibration data from railway tracks. Key methodologies include:

- Clustering-based detection using Principal Component Analysis (PCA) and MiniBatch KMeans for unsupervised clustering.
- Distance-from-mean spectrum analysis to identify deviations from a computed mean spectrum.
- Integration of both methods to flag overlapping anomalies, prioritizing critical track segments for maintenance.

This methodology is designed for adaptability and scalability, allowing real-time, continuous monitoring without the need for labeled datasets, making it suitable for various railway networks.

---

## Dataset

The datasets consist of vibration signals captured from sensors on locomotives operating in the Metro Valencia network. The data include vertical and lateral accelerations measured at axle boxes and processed through the Short-Term Fourier Transform (STFT) to produce normalized spectrograms.

---

## Analysis and Methodologies

### Signal Processing

- **STFT**: Used to generate spectrograms, revealing the frequency components of vertical and lateral accelerations.
- **Normalization**: Ensures consistency across different signal intensities.

### Clustering-Based Detection

- **Principal Component Analysis (PCA)**: Reduces the dimensionality of spectrogram data for efficient clustering.
- **MiniBatch KMeans**: Groups data into clusters, identifying anomalies as deviations beyond a user-defined threshold (98th percentile).

### Distance-Based Detection

- **Distance-from-Mean Spectrum**: Flags anomalies by measuring deviations from a computed mean spectrum, effectively capturing isolated irregularities.

### Integrated Methodology

- Anomalies detected through both methods are combined to identify high-priority track segments for maintenance.

---

## Files

The main files of the project are:

1. `src/utils.py`: Utility functions.
2. `src/logs.py`: Logging functions.
3. `src/load_config.py` and `configs/app_config.yml`: Configuration files and parameters.

---

## Parameters

- **window_length**: 0.25 seconds (25 samples per window with a sampling frequency of 100 Hz).
- **overlap**: 95%, ensuring substantial overlap between windows with only 5% unique data per window.
- **nfft**: 128 points for FFT, determining the frequency resolution of the spectrogram.
- **sampling_frequency_stft_prepared**: 100 Hz, with data points spaced at 0.01-second intervals.

---

## Results and Insights

- **Effectiveness of Combined Methods**: The clustering-based approach captures common patterns, while the distance-based method detects rare anomalies, providing a comprehensive view of track conditions.
- **Route-Specific Analysis**: Variations in anomaly distributions across routes highlight track sections requiring targeted maintenance.
- **Predictive Maintenance**: Enables proactive identification of high-priority anomalies, optimizing maintenance efforts and enhancing track safety.

---

## Future Work

- **Field Validation**: Test methodology in real-world scenarios to fine-tune anomaly thresholds.
- **Extended Features**: Investigate additional accelerometer features and integrate complementary sensor data (e.g., GPS or visual data).
- **Generalization**: Apply the methodology to other railway networks to validate adaptability and scalability.

---

## TODOs

- Include different versions of `TO_EM_v2` and `EM_TO_v2`.
- Add combined anomalies in the dashboard.

---

## Acknowledgments

Special thanks to contributors and collaborators for their insights and support in this project.
