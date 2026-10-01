# SpasmSense

SpasmSense is an experimental wearable biosignal-processing system designed to explore whether surface electromyography (sEMG) and machine learning can identify unusual or sustained patterns of muscle activity.

The long-term goal is to investigate whether a wearable system could recognize abnormal muscle activity in real time and eventually support personalized monitoring or closed-loop intervention research.

## Current Prototype

The current version is a software proof of concept using synthetic EMG data.

It includes:

- EMG signal preprocessing
- band-pass and notch filtering
- signal segmentation
- time-domain and frequency-domain feature extraction
- personalized anomaly detection
- sustained-event detection
- visualization of detected abnormal activity

## System Architecture

```text
Surface EMG
    ↓
Signal filtering
    ↓
Short-time segmentation
    ↓
Feature extraction
    ↓
Personalized anomaly detection
    ↓
Temporal event detection
    ↓
Possible sustained abnormal muscle activity