# ASL Translator Glove

A wearable that recognizes American Sign Language gestures and speaks them out loud — running entirely on a $30 microcontroller.

EE 446 TinyML Final Project · University of Washington · Spring 2026

---

## What it does

You wear the device on your hand, sign one of 8 ASL words, and the system identifies the gesture in under 50ms and streams the result over Bluetooth to a paired display that speaks it aloud through TTS. Everything inference-side runs on-device — no cloud, no phone-side ML, no internet.

**Vocabulary (v1):** HELLO, THANK YOU, HELP, GOODBYE, STOP, MORE, EAT, WATER

Chosen for motion separability — IMU alone can't disambiguate hand-shape-dominant signs, so we picked signs with distinct motion signatures.

## Why this matters

Roughly 70 million people worldwide use sign language as their primary form of communication. The US alone has about 11.8 million deaf or hard-of-hearing individuals but only 2,300 certified ASL interpreters. Real-time interpretation hardware exists but it's either expensive, cloud-dependent, or both. We wanted to see how far you can push that with a $30 board and a 200KB model.

## How it works

IMU (6-axis) → Window buffer → 1D-CNN (int8 quantized) → Confidence threshold → BLE → Web display + TTS

| Stage | What happens |
|---|---|
| Data collection | 4 signers, ~25 reps × 8 signs, IMU streamed via Edge Impulse |
| Model | 1D-CNN, custom architecture, trained in TensorFlow |
| Compression | INT8 post-training quantization, target <200KB |
| Deployment | Arduino sketch with BLE characteristic + confidence-threshold rejection |
| Display | Web Bluetooth app with live gesture, sentence builder, and TTS |

## Hardware

Arduino Nano 33 BLE Sense (LSM9DS1 IMU, 256KB RAM, 1MB flash, BLE 5.0)

## Edge Impulse Link

https://studio.edgeimpulse.com/public/1022005/live

## Repo layout

- data/ — IMU recordings, labeling protocol, sample CSVs
- model/ — Training notebooks, saved checkpoints
- compression/ — Quantization scripts, accuracy/size/latency comparison tables
- deployment/ — Arduino sketch, BLE protocol, on-device profiling
- webapp/ — Web Bluetooth demo
- report/ — Final report, figures
- INTERFACE.md — Model ↔ deployment contract (input shape, output format, BLE schema)

## Team

| Track | Owner |
|---|---|
| Data collection + labeling | Carter |
| Data collection + labeling | Praxides |
| Data collection + labeling | Ananya |
| Deployment + Model architecture + training + quantization + BLE + web app + system integration | Jayson |

## Status

Active development. Pipeline validated end-to-end on Lab 9 baseline. Working on real data + production model.
