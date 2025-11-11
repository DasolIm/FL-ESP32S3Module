#ifndef CONFIG_H
#define CONFIG_H
#pragma once

#include <Arduino.h>

// ----------------------------- WiFi Configuration -----------------------------
// #define WIFI_SSID         "Yonsei-IoT-2G"
// #define WIFI_PASSWORD     "yonseiiot209"
#define WIFI_SSID         "hm-ph401_2Ghz"
#define WIFI_PASSWORD     "pink2lion92"
#define WIFI_TIMEOUT_MS   10000
#define WIFI_RETRY_DELAY  500
#define MQTT_INITIAL_CONNECT_TIMEOUT_MS 100000
#define WIFI_RECONNECT_INTERVAL_MS 5000

// ----------------------------- MQTT Configuration -----------------------------
#define MQTT_SERVER       "broker.emqx.io"
#define MQTT_PORT         1883
#define MQTT_BUFFER_SIZE  32768
#define MQTT_KEEPALIVE    20
#define MQTT_TIMEOUT_MS   5000
#define MQTT_RECONNECT_INTERVAL_MS 5000
#define MQTT_PUBLISH_QUEUE_SIZE 10 // MQTT 발행 요청 큐 크기

// MQTT Topics
#define TOPIC_SOT         "yonseiiot/ysk/fl/command/sot"
#define TOPIC_WEIGHT_FMT  "yonseiiot/ysk/fl/%s/weight"
#define TOPIC_STATUS_FMT  "yonseiiot/ysk/fl/%s/status"
#define TOPIC_GLOBAL      "yonseiiot/ysk/fl/global/weight"

// ----------------------------- Model Configuration -----------------------------
#define N_HIDDEN          3
#define BATCH_SIZE        8
#define TRAIN_EPOCHS      5
#define LEARNING_RATE     0.0005f
#define ADAM_BETA1        0.9f
#define ADAM_BETA2        0.999f
#define ADAM_EPSILON      1e-8f
#define SELECTED_DATASET_ID 0
#define RANDOM_SEED       0
// Hidden layer neurons
static const uint8_t HIDDEN_NEURONS[N_HIDDEN] = {32, 32, 32};

// ----------------------------- Memory Configuration -----------------------------
#define PSRAM_MIN_SIZE    1024 * 1024  // 1MB minimum
#define MEMORY_ALIGNMENT  4

// ----------------------------- Debug Configuration -----------------------------
#define DEBUG_LEVEL       1  // 0: No debug, 1: Info, 2: Verbose
#define SERIAL_BAUD       115200

// Debug macros
#if DEBUG_LEVEL >= 1
  #define DEBUG_PRINT(x) Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
  #define DEBUG_PRINTF(fmt, ...) Serial.printf(fmt, ##__VA_ARGS__)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
  #define DEBUG_PRINTF(fmt, ...)
#endif

#if DEBUG_LEVEL >= 2
  #define DEBUG_VERBOSE_PRINT(x) Serial.print(x)
  #define DEBUG_VERBOSE_PRINTLN(x) Serial.println(x)
  #define DEBUG_VERBOSE_PRINTF(fmt, ...) Serial.printf(fmt, ##__VA_ARGS__)
#else
  #define DEBUG_VERBOSE_PRINT(x)
  #define DEBUG_VERBOSE_PRINTLN(x)
  #define DEBUG_VERBOSE_PRINTF(fmt, ...)
#endif

// ----------------------------- Error Codes -----------------------------
enum ErrorCode {
    SUCCESS = 0,
    ERROR_WIFI_CONNECT = -1,
    ERROR_MQTT_CONNECT = -2,
    ERROR_TASK_CREATION_FAILED = -3,
    ERROR_PSRAM_INIT = -4,
    ERROR_MODEL_INIT = -5,
    ERROR_MEMORY_ALLOC = -6,
    ERROR_TRAINING = -7,
    ERROR_ENCODING = -8,
    ERROR_DECODING = -9
};

// ----------------------------- Status Codes -----------------------------
enum ClientStatus {
    STATUS_INITIALIZING,
    STATUS_READY,
    STATUS_TRAINING,
    STATUS_SENDING,
    STATUS_APPLYING,
    STATUS_ERROR,
};

#endif // CONFIG_H