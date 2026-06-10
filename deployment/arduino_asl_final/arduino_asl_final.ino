/*
 * ASL Translator Glove — Final Sketch
 * EE 446 TinyML Final Project · Spring 2026
 *
 * Reads 6-axis IMU (BMI270 on Nano 33 BLE Sense Rev2),
 * runs Edge Impulse INT8 quantized classifier (8-class ASL),
 * applies confidence threshold, broadcasts JSON over BLE
 * to a paired web app for display + TTS.
 *
 * Trained model: 92.7% test accuracy (INT8 quantized)
 * Vocabulary: HELLO, THANK_YOU, HELP, GOODBYE, STOP, MORE, EAT, WATER
 *
 * BLE Service: 19b10000-e8f2-537e-4f6c-d104768a1214
 * BLE Char:    19b10001-e8f2-537e-4f6c-d104768a1214
 * Payload (JSON):
 *   {"gesture":"HELLO","confidence":0.92,"latency_us":63000}
 */

#include <asl-translator-glove_inferencing.h>
#include <Arduino_BMI270_BMM150.h>   // Rev2 IMU
#include <ArduinoBLE.h>

/* === Config === */
#define CONVERT_G_TO_MS2       9.80665f
#define MAX_ACCEPTED_RANGE     2.0f
#define CONFIDENCE_THRESHOLD   0.70f   // below this → "UNKNOWN"
#define MOTION_GYRO_THRESHOLD  12.0f   // avg |gyro| deg/s; below = idle, skip inference
#define GESTURE_COOLDOWN_MS    1200    // after a confident detection, pause before next

/* === BLE === */
BLEService gestureService("19b10000-e8f2-537e-4f6c-d104768a1214");
BLEStringCharacteristic gestureChar(
  "19b10001-e8f2-537e-4f6c-d104768a1214",
  BLERead | BLENotify,
  120
);

/* === Inference buffer === */
static bool debug_nn = false;
static float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };
static float inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

/* === Track last output (avoid spamming BLE / Serial) === */
static String last_sent_gesture = "";
static String last_printed_gesture = "";
static unsigned long last_detection_ms = 0;

/* Forward decls */
void run_inference_background();
void emit_gesture(const String &predicted, float confidence, unsigned long latency_us);
float ei_get_sign(float number) { return (number >= 0.0) ? 1.0 : -1.0; }

// True when the current window has enough gyro activity to be a real gesture.
static bool window_has_motion(const float *buf) {
  float gyro_energy = 0.0f;
  for (int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
    int b = i * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME + 3;
    gyro_energy += fabs(buf[b]) + fabs(buf[b + 1]) + fabs(buf[b + 2]);
  }
  gyro_energy /= EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  return gyro_energy >= MOTION_GYRO_THRESHOLD;
}

// Edge Impulse label → web app label (e.g. "THANK YOU" → "THANK_YOU").
static String normalize_gesture_label(const char *label) {
  String normalized = String(label);
  normalized.replace(' ', '_');
  return normalized;
}

// Send to Serial + BLE only when the prediction changes.
void emit_gesture(const String &predicted, float confidence, unsigned long latency_us) {
  if (predicted == last_printed_gesture) {
    return;
  }

  Serial.print("Gesture: "); Serial.print(predicted);
  Serial.print("  Conf: ");  Serial.print(confidence, 3);
  Serial.print("  Latency: "); Serial.print(latency_us); Serial.println(" us");
  last_printed_gesture = predicted;

  if (predicted != last_sent_gesture) {
    char payload[120];
    snprintf(payload, sizeof(payload),
      "{\"gesture\":\"%s\",\"confidence\":%.2f,\"latency_us\":%lu}",
      predicted.c_str(), confidence, latency_us);
    gestureChar.writeValue(payload);
    last_sent_gesture = predicted;
  }
}

/* ============================================================
 *  setup
 * ============================================================ */
void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 3000);
  Serial.println("ASL Translator Glove — starting up");

  // --- IMU init ---
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.print("Accel rate = "); Serial.print(IMU.accelerationSampleRate()); Serial.println(" Hz");
  Serial.print("Gyro rate  = "); Serial.print(IMU.gyroscopeSampleRate());    Serial.println(" Hz");

  // --- sanity check: model expects 6 axes ---
  if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 6) {
    Serial.print("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME = ");
    Serial.print(EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
    Serial.println(" (expected 6 for accX/Y/Z + gyrX/Y/Z)");
    while (1);
  }

  // --- BLE init ---
  if (!BLE.begin()) {
    Serial.println("Failed to start BLE!");
    while (1);
  }
  BLE.setLocalName("ASL_Glove");
  BLE.setDeviceName("ASL_Glove");
  BLE.setAdvertisedService(gestureService);
  gestureService.addCharacteristic(gestureChar);
  BLE.addService(gestureService);
  gestureChar.writeValue("{\"gesture\":\"READY\",\"confidence\":0.00,\"latency_us\":0}");
  BLE.advertise();
  Serial.println("BLE advertising as 'ASL_Glove'");

}

/* ============================================================
 *  Background inference thread
 * ============================================================ */
void run_inference_background() {
  unsigned long now = millis();

  // After a confident detection, skip further inference until cooldown ends.
  if (last_detection_ms > 0 && (now - last_detection_ms) < GESTURE_COOLDOWN_MS) {
    return;
  }

  // Skip inference when the glove is still — avoids idle false positives (HELLO / THANK YOU spam).
  if (!window_has_motion(buffer)) {
    if (last_printed_gesture != "IDLE") {
      Serial.println("Idle — waiting for gesture motion");
      last_printed_gesture = "IDLE";
    }
    return;
  }

  memcpy(inference_buffer, buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE * sizeof(float));

  signal_t signal;
  int err = numpy::signal_from_buffer(inference_buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) {
    Serial.print("Failed to create signal ("); Serial.print(err); Serial.println(")");
    return;
  }

  ei_impulse_result_t result = { 0 };
  unsigned long inf_start = micros();
  err = run_classifier(&signal, &result, debug_nn);
  unsigned long latency_us = micros() - inf_start;

  if (err != EI_IMPULSE_OK) {
    Serial.print("ERR: classifier failed ("); Serial.print(err); Serial.println(")");
    return;
  }

  float max_prob = 0.0;
  int   max_idx  = 0;
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (result.classification[i].value > max_prob) {
      max_prob = result.classification[i].value;
      max_idx  = i;
    }
  }

  String predicted;
  if (max_prob < CONFIDENCE_THRESHOLD) {
    predicted = "UNKNOWN";
  } else {
    predicted = normalize_gesture_label(result.classification[max_idx].label);
  }

  emit_gesture(predicted, max_prob, latency_us);

  // Start cooldown only after a confident gesture — lets the user reset between signs.
  if (predicted != "UNKNOWN") {
    last_detection_ms = now;
  }
}

/* ============================================================
 *  Main loop: fill buffer with 6-axis IMU data + poll BLE
 * ============================================================ */
void loop() {
  while (1) {
    BLE.poll();   // handle BLE events

    uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // roll buffer back 6 slots (1 sample = 6 axes)
      numpy::roll(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, -6);

      float ax, ay, az, gx, gy, gz;
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);

      // clip accel to ±2G
      if (fabs(ax) > MAX_ACCEPTED_RANGE) ax = ei_get_sign(ax) * MAX_ACCEPTED_RANGE;
      if (fabs(ay) > MAX_ACCEPTED_RANGE) ay = ei_get_sign(ay) * MAX_ACCEPTED_RANGE;
      if (fabs(az) > MAX_ACCEPTED_RANGE) az = ei_get_sign(az) * MAX_ACCEPTED_RANGE;

      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 6] = ax * CONVERT_G_TO_MS2;
      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 5] = ay * CONVERT_G_TO_MS2;
      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 4] = az * CONVERT_G_TO_MS2;
      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3] = gx;  // gyro: deg/s, no conversion
      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 2] = gy;
      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 1] = gz;

      // Warm up 2 s buffer, then infer every slice (~0.5 s) for faster response
      static int total_samples = 0;
      static int slice_count = 0;
      if (++total_samples >= EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
        if (++slice_count >= EI_CLASSIFIER_SLICE_SIZE) {
          slice_count = 0;
          run_inference_background();
        }
      }
    }

    // wait for next tick
    uint64_t now_us = micros();
    if (next_tick > now_us) {
      uint64_t time_to_wait = next_tick - now_us;
      delay((int)(time_to_wait / 1000));
      delayMicroseconds(time_to_wait % 1000);
    }
  }
}