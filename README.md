# ASL Translator Glove

EE 446 TinyML Final Project — Spring 2026
University of Washington

**Team:** Jayson, Carter, Ananya, Praxides

A wearable that recognizes American Sign Language gestures using IMU data
and a quantized 1D-CNN running entirely on Arduino Nano 33 BLE Sense.
Recognized signs are streamed over BLE for real-time display and speech output.

## Hardware
- Arduino Nano 33 BLE Sense (IMU: LSM9DS1)

## Pipeline
Data Collection → 1D-CNN Training → Quantization → Arduino Deployment → BLE Output

## Repo Structure
- `/data` — IMU recordings and labels
- `/model` — Training notebooks, architectures, saved models
- `/compression` — Quantization scripts, before/after metrics
- `/deployment` — Arduino sketch, BLE protocol, profiling results
- `/webapp` — Web Bluetooth demo app
- `/report` — Final report and figures

## Track Ownership
- **Data:** TBD
- **Model:** Praxides
- **Compression:** Carter
- **Deployment:** Jayson
