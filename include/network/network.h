// network/network.h
#ifndef NETWORK_H
#define NETWORK_H
#pragma once

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "../common/config.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h> // FreeRTOS Queue 사용을 위해 추가

typedef void (*MqttCallback)(char* topic, byte* payload, unsigned int length);

// MQTT 발행 요청을 위한 메시지 구조체 정의
// 동적 할당된 페이로드를 가리킬 수 있도록 char* 사용.
// 중요: 이 페이로드는 큐에 들어가기 전에 heap_caps_malloc 등으로 할당되어야 하며,
// 태스크에서 발행 후 heap_caps_free로 해제되어야 합니다.
typedef struct {
    char topic[96]; // 토픽 길이에 따라 충분히 크게 설정
    char* payload;  // 메시지 페이로드 (동적 할당 포인터)
    size_t length;  // 페이로드 길이
    // bool retain;    // retain 플래그 (선택 사항, 필요하면 사용)
    // int qos;        // QoS 레벨 (선택 사항, 필요하면 사용)
} MqttPublishMessage_t;


class NetworkManager {
public:
    NetworkManager();
    ErrorCode begin(MqttCallback callback);
    bool isMqttConnected();
    PubSubClient& getMqttClient(); // (선택 사항: 외부에서 직접 접근하는 경우에만 유지)

    // 이 함수들은 이제 직접 발행하지 않고 큐에 요청을 넣습니다.
    void publishStatus(ClientStatus status);
    bool publishWeight(const char* encodedWeights, size_t length);
    void publishTrainingResult(float loss);

private:
    WiFiClient    _wifiClient;
    PubSubClient  _mqttClient;

    char _mqttTopicWeight[96];
    char _mqttTopicStatus[96];

    unsigned long _lastWifiReconnectAttempt = 0;
    unsigned long _lastMqttReconnectAttempt = 0;

    TaskHandle_t _mqttTaskHandle;
    QueueHandle_t _mqttPublishQueue; // MQTT 발행 요청 큐 핸들 추가

    ErrorCode connectWifi();
    ErrorCode connectMqttInternal();

    static void staticMqttLoopTask(void* pvParameters);
};

#endif // NETWORK_H