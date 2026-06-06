# Model ↔ Deployment Interface Contract

Owners: Praxides (model), Jayson (deployment)
Status: Draft — will refine as we go

## Word List (8 ASL signs)
HELLO, THANK_YOU, HELP, GOODBYE, STOP, MORE, EAT, WATER

(Switched from original list — IMU can't pick up hand-shape-dominant signs 
like NO/YES/PLEASE/SORRY well, so going with motion-heavy signs.)

## Model Input
- Shape: (window_size, 6)
  - window_size: TBD (depends on sample length / sampling rate)
  - 6 channels: ax, ay, az, gx, gy, gz
- Sampling rate: TBD (100Hz target)
- Type after int8 quantization: int8

## Model Output
- Format: softmax probabilities
- Shape: (8,) — one per class
- Type: int8 (post-quantization)
- Class order MUST match deployment:
  [HELLO, THANK_YOU, HELP, GOODBYE, STOP, MORE, EAT, WATER]

## Export Format
- int8-quantized TFLite → C header (model.h)
- Same flow as lab 9

## Deployment Constraints (Arduino Nano 33 BLE Sense)
- Model size (post int8): < 200 KB
- Tensor arena RAM: < 250 KB
- Inference latency target: < 50 ms

## BLE Output (Deployment → Display)
JSON over BLE characteristic on each prediction:
{
  "gesture": "HELLO",
  "confidence": 0.92,
  "latency_us": 12345
}

## Confidence Threshold
- If max softmax < 0.70 → output "UNKNOWN"
- Implemented on Arduino side (deployment)
