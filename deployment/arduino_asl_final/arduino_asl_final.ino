/*
  ASL Translator Glove — Final Sketch
  EE 446 TinyML Final Project, Spring 2026
  
  Reads IMU, runs quantized model, streams predictions over BLE
  to a paired web app for display + TTS.
  
  Based on Lab 9 IMU Classifier (Harvard TinyMLx).
  
  BLE Service UUID: 19b10000-e8f2-537e-4f6c-d104768a1214
  BLE Char UUID:    19b10001-e8f2-537e-4f6c-d104768a1214
  Payload (JSON):
    {"gesture":"HELLO","confidence":0.92,"latency_us":12345}
*/

#define USE_NANO_33_BLE_REV2_IMU 1

#if USE_NANO_33_BLE_REV2_IMU
  #include <Arduino_BMI270_BMM150.h>
#else
  #include <Arduino_LSM9DS1.h>
#endif

#include <TensorFlowLite.h>
#include <ArduinoBLE.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "model.h"

// === Config ===
const float accelerationThreshold = 2.5;  // G's, motion trigger
const int   numSamples            = 119;
const float CONFIDENCE_THRESHOLD  = 0.70; // below this → "UNKNOWN"

int samplesRead = numSamples;

// === TFLite Micro globals ===
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 16 * 1024;
byte tensorArena[tensorArenaSize];

// === Gesture vocabulary (8 ASL signs) ===
// NOTE: Currently using Lab 9's 2-class model.h as placeholder.
// When Praxides' 8-class model.h is ready, replace model.h and uncomment full list.
const char* GESTURES[] = {
  "hi",   // placeholder — will become HELLO
  "sup"   // placeholder — will become THANK_YOU / HELP / GOODBYE / STOP / MORE / EAT / WATER
};

// Final 8-class list (uncomment when model.h is updated):
// const char* GESTURES[] = {
//   "HELLO", "THANK_YOU", "HELP", "GOODBYE",
//   "STOP", "MORE", "EAT", "WATER"
// };

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// === BLE setup ===
BLEService gestureService("19b10000-e8f2-537e-4f6c-d104768a1214");
BLEStringCharacteristic gestureChar(
  "19b10001-e8f2-537e-4f6c-d104768a1214",
  BLERead | BLENotify,
  120  // max payload length
);

void setup() {
  Serial.begin(9600);
  // Don't block forever waiting for Serial — we want BLE to work without a USB host
  unsigned long serialStart = millis();
  while (!Serial && (millis() - serialStart) < 3000);

  // --- IMU init ---
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.print("Accel rate = "); Serial.print(IMU.accelerationSampleRate()); Serial.println(" Hz");
  Serial.print("Gyro rate  = "); Serial.print(IMU.gyroscopeSampleRate());    Serial.println(" Hz");

  // --- TFLite Micro init ---
  tflModel = tflite::GetModel(model);
  tflInterpreter = new tflite::MicroInterpreter(
    tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter
  );

  TfLiteStatus allocateStatus = tflInterpreter->AllocateTensors();
  if (allocateStatus != kTfLiteOk) {
    Serial.println("AllocateTensors() failed. Try increasing tensorArenaSize.");
    while (1);
  }
  tflInputTensor  = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // --- BLE init ---
  if (!BLE.begin()) {
    Serial.println("Failed to start BLE!");
    while (1);
  }
  BLE.setLocalName("ASL_Glove");
  BLE.setAdvertisedService(gestureService);
  gestureService.addCharacteristic(gestureChar);
  BLE.addService(gestureService);
  gestureChar.writeValue("{\"gesture\":\"READY\",\"confidence\":0.00,\"latency_us\":0}");
  BLE.advertise();
  Serial.println("BLE advertising as 'ASL_Glove'");
}

void loop() {
  // Poll BLE (handles connect/disconnect events)
  BLE.poll();

  float aX, aY, aZ, gX, gY, gZ;

  // === Wait for significant motion ===
  while (samplesRead == numSamples) {
    BLE.poll();
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
      if (aSum >= accelerationThreshold) {
        samplesRead = 0;
        break;
      }
    }
  }

  // === Collect numSamples worth of IMU data ===
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // Normalize to [0, 1]
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // === Inference + latency measurement ===
        unsigned long t0 = micros();
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        unsigned long latency_us = micros() - t0;

        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          return;
        }

        // === Find argmax + confidence ===
        float maxProb = 0.0;
        int   maxIdx  = 0;
        for (int i = 0; i < NUM_GESTURES; i++) {
          float p = tflOutputTensor->data.f[i];
          if (p > maxProb) {
            maxProb = p;
            maxIdx  = i;
          }
        }

        // === Confidence threshold ===
        const char* predicted;
        if (maxProb < CONFIDENCE_THRESHOLD) {
          predicted = "UNKNOWN";
        } else {
          predicted = GESTURES[maxIdx];
        }

        // === Serial debug ===
        Serial.print("Gesture: "); Serial.print(predicted);
        Serial.print("  Confidence: "); Serial.print(maxProb, 3);
        Serial.print("  Latency: "); Serial.print(latency_us); Serial.println(" us");

        // === BLE output (JSON) ===
        char payload[120];
        snprintf(payload, sizeof(payload),
          "{\"gesture\":\"%s\",\"confidence\":%.2f,\"latency_us\":%lu}",
          predicted, maxProb, latency_us);
        gestureChar.writeValue(payload);
      }
    }
  }
}
