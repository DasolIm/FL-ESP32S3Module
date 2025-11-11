#include "network/network.h"
#include "common/config.h" // DEBUG_PRINTLN 등 디버그 매크로 사용을 위해 추가

NetworkManager::NetworkManager()
    : _mqttClient(_wifiClient), _mqttTaskHandle(nullptr), _mqttPublishQueue(nullptr) { // _mqttPublishQueue 초기화 추가
    _mqttClient.setBufferSize(MQTT_BUFFER_SIZE);
    _mqttClient.setKeepAlive(MQTT_KEEPALIVE);

    // MQTT 발행 큐 생성
    // 큐의 크기와 각 항목의 크기를 정의.
    _mqttPublishQueue = xQueueCreate(MQTT_PUBLISH_QUEUE_SIZE, sizeof(MqttPublishMessage_t));
    if (_mqttPublishQueue == nullptr) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to create MQTT Publish Queue!");
        // 여기서 에러 처리를 어떻게 할지 결정해야 함 (fatal error or continue with warnings)
    }
}

ErrorCode NetworkManager::begin(MqttCallback callback) {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) delay(10); // 시리얼 포트 대기
    delay(500);

    DEBUG_PRINTLN("NetworkManager: Initializing...");

    // 1. Wi-Fi 설정 및 연결
    ErrorCode wifiStatus = connectWifi();
    if (wifiStatus != SUCCESS) {
        DEBUG_PRINTF("NetworkManager: Wi-Fi connection failed with error code %d.\n", wifiStatus);
        return wifiStatus;
    }

    // Wi-Fi 연결 성공 후 MAC 주소 기반 토픽 초기화
    snprintf(_mqttTopicWeight, sizeof(_mqttTopicWeight), TOPIC_WEIGHT_FMT, WiFi.macAddress().c_str());
    snprintf(_mqttTopicStatus, sizeof(_mqttTopicStatus), TOPIC_STATUS_FMT, WiFi.macAddress().c_str());
    DEBUG_PRINTF("NetworkManager: Weight Topic: %s\n", _mqttTopicWeight);
    DEBUG_PRINTF("NetworkManager: Status Topic: %s\n", _mqttTopicStatus);

    // 2. MQTT 설정 (콜백 함수 지정)
    _mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
    _mqttClient.setCallback(callback);

    // 3. 초기 MQTT 연결 시도 (반복 시도)
    DEBUG_PRINTLN("NetworkManager: Attempting initial MQTT connection (repeatedly if needed)...");
    unsigned long mqttConnectStartTime = millis();
    ErrorCode mqttStatus = ERROR_MQTT_CONNECT; // 초기 상태를 실패로 설정

    while (mqttStatus != SUCCESS) {
        mqttStatus = connectMqttInternal(); // 연결 시도
        if (mqttStatus == SUCCESS) {
            DEBUG_PRINTLN("NetworkManager: Initial MQTT connection successful.");
            break; // 성공하면 루프 종료
        }

        // 실패 시 재시도 대기
        delay(MQTT_RECONNECT_INTERVAL_MS); // config.h에 정의된 재연결 간격 사용
        DEBUG_PRINT("."); // 연결 시도 중임을 나타내는 점 출력

        // 타임아웃 검사 (선택 사항: 무한 루프 방지)
        if (millis() - mqttConnectStartTime > MQTT_INITIAL_CONNECT_TIMEOUT_MS) {
            DEBUG_PRINTLN("\nNetworkManager: Initial MQTT connection timed out.");
            return ERROR_MQTT_CONNECT;
        }
    }
    DEBUG_PRINTLN(""); // 점 출력 후 줄바꿈

    // DEBUG: 태스크 생성 전 힙 메모리 상태 확인
    DEBUG_PRINTF("NetworkManager: Before task creation, Free Heap: %u bytes, Largest Block: %u bytes\n",
                 heap_caps_get_free_size(MALLOC_CAP_8BIT),
                 heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    DEBUG_PRINTF("NetworkManager: PSRAM Free Heap: %u bytes, PSRAM Largest Block: %u bytes\n",
                 heap_caps_get_free_size(MALLOC_CAP_SPIRAM), // PSRAM (SPIRAM)
                 heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)); // PSRAM (SPIRAM)


    // 4. MQTT 루프를 위한 FreeRTOS 태스크 생성 (Core 1에 고정)
    // 'this' 포인터를 태스크 파라미터로 전달하여 static 함수에서 멤버 접근 가능하게 함
    xTaskCreatePinnedToCore(
        NetworkManager::staticMqttLoopTask, // 태스크 함수
        "MqttLoopTask",                     // 태스크 이름 (디버깅용)
        1024 * 8,                           // 스택 크기 (바이트). PubSubClient가 충분히 작동하도록 넉넉히 설정.
        this,                               // 태스크에 전달할 파라미터 (NetworkManager 인스턴스)
        1,                                  // 태스크 우선순위 (0은 가장 낮음, configMAX_PRIORITIES-1이 가장 높음)
        &_mqttTaskHandle,                   // 생성된 태스크의 핸들을 저장할 변수
        1                                   // 태스크가 실행될 코어 (Core 1)
    );

    if (_mqttTaskHandle == nullptr) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to create MQTT Loop Task!");
        return ERROR_TASK_CREATION_FAILED;
    }
    DEBUG_PRINTLN("NetworkManager: MQTT Loop Task created successfully on Core 1.");
    DEBUG_PRINTLN("NetworkManager: Initialization complete.");
    return SUCCESS;
}

ErrorCode NetworkManager::connectWifi() {
    DEBUG_PRINTF("NetworkManager: Connecting to WiFi SSID: %s\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    unsigned long startTime = millis();
    while (WiFi.status() != WL_CONNECTED) {
        delay(WIFI_RETRY_DELAY);
        DEBUG_PRINT(".");
        if (millis() - startTime > WIFI_TIMEOUT_MS) {
            DEBUG_PRINTLN("\nNetworkManager: WiFi connection timed out.");
            return ERROR_WIFI_CONNECT;
        }
    }
    DEBUG_PRINTF("\nNetworkManager: WiFi connected. IP: %s\n", WiFi.localIP().toString().c_str());
    return SUCCESS;
}

ErrorCode NetworkManager::connectMqttInternal() {
    DEBUG_PRINT("NetworkManager: Attempting MQTT connection...");
    String clientId = "ESP32Client-" + WiFi.macAddress();
    if (_mqttClient.connect(clientId.c_str())) {
        DEBUG_PRINTLN("connected");
        // 구독
        _mqttClient.subscribe(TOPIC_SOT);
        _mqttClient.subscribe(TOPIC_GLOBAL); // QoS 2로 글로벌 가중치 토픽 구독
        return SUCCESS;
    } else {
        DEBUG_PRINTF("failed, rc=%d\n", _mqttClient.state());
        return ERROR_MQTT_CONNECT;
    }
}

// NetworkManager::loop() 함수는 이제 제거되었습니다.

// Core 1에서 실행될 MQTT 루프 태스크의 실제 구현
void NetworkManager::staticMqttLoopTask(void* pvParameters) {
    // 전달받은 NetworkManager 인스턴스 포인터를 원래 타입으로 캐스팅
    NetworkManager* self = static_cast<NetworkManager*>(pvParameters);
    MqttPublishMessage_t msg; // 큐에서 메시지를 받을 구조체

    for (;;) { // 무한 루프
        // Wi-Fi 연결 상태 확인 및 재연결 로직
        if (WiFi.status() != WL_CONNECTED) {
            if (millis() - self->_lastWifiReconnectAttempt >= WIFI_RECONNECT_INTERVAL_MS) {
                self->_lastWifiReconnectAttempt = millis();
                DEBUG_PRINTLN("NetworkManager [Task]: WiFi disconnected. Attempting to reconnect...");
                if (self->connectWifi() == SUCCESS) {
                    DEBUG_PRINTLN("NetworkManager [Task]: WiFi reconnected, attempting MQTT reconnect.");
                    // Wi-Fi 재연결 성공 시 MQTT도 재연결 시도
                    self->connectMqttInternal();
                }
            }
        } else { // Wi-Fi가 연결되어 있을 때만 MQTT 처리
            if (!self->_mqttClient.connected()) {
                if (millis() - self->_lastMqttReconnectAttempt >= MQTT_RECONNECT_INTERVAL_MS) {
                    self->_lastMqttReconnectAttempt = millis();
                    DEBUG_PRINTLN("NetworkManager [Task]: MQTT disconnected. Attempting to reconnect...");
                    self->connectMqttInternal();
                }
            }
            // MQTT 클라이언트의 루프 함수는 연결 상태와 상관없이 계속 호출되어야 함
            // PubSubClient의 loop()는 내부적으로 메시지를 처리하고 Keep-alive 핑을 보냅니다.
            self->_mqttClient.loop();
        }

        // 큐에서 발행 요청 확인 및 처리
        // xQueueReceive는 블로킹될 수 있으므로, vTaskDelay 없이 바로 호출하거나,
        // 일정 시간만 대기하도록 (portMAX_DELAY 대신 0 또는 작은 틱 수) 설정하여 다른 작업도 수행하게 함.
        if (xQueueReceive(self->_mqttPublishQueue, &msg, 0) == pdPASS) { // 0ms 대기 (논블로킹)
            if (self->_mqttClient.connected()) {
                // 실제 MQTT 발행
                bool pub_state = self->_mqttClient.publish(msg.topic, msg.payload, msg.length); // msg.retain, msg.qos 등 추가 가능
                DEBUG_PRINTF("NetworkManager [Task]: Published topic '%s' (len %u, state %d)\n", msg.topic, msg.length, pub_state);
            } else {
                DEBUG_PRINTF("NetworkManager [Task]: MQTT disconnected, failed to publish topic '%s'\n", msg.topic);
            }
            // 동적으로 할당된 페이로드 메모리 해제
            if (msg.payload != nullptr) {
                heap_caps_free(msg.payload);
                msg.payload = nullptr; // Dangling pointer 방지
            }
        }

        vTaskDelay(10 / portTICK_PERIOD_MS); // 10ms 대기 후 다시 실행 (다른 태스크에 CPU 양보)
    }
}

bool NetworkManager::isMqttConnected() {
    return _mqttClient.connected();
}

PubSubClient& NetworkManager::getMqttClient() {
    return _mqttClient;
}

void NetworkManager::publishStatus(ClientStatus status) {
    const char* statusStr;
    switch(status) {
        case STATUS_INITIALIZING: statusStr = "INITIALIZING"; break;
        case STATUS_READY:        statusStr = "READY";        break;
        case STATUS_TRAINING:     statusStr = "TRAINING";     break;
        case STATUS_SENDING:      statusStr = "SENDING";      break;
        case STATUS_APPLYING:
            statusStr = "APPLING GLOBAL WEIGHT";              break;
        case STATUS_ERROR:        statusStr = "ERROR";        break;
        default:                  statusStr = "UNKNOWN";      break;
    }

    // 임시 버퍼에 포맷된 상태 메시지 생성
    char formatted_msg[64]; // 충분한 크기의 버퍼
    snprintf(formatted_msg, sizeof(formatted_msg), "{\"Status\": \"%s\", \"DataSet ID\": %d}", statusStr, SELECTED_DATASET_ID);

    MqttPublishMessage_t msg;
    strncpy(msg.topic, _mqttTopicStatus, sizeof(msg.topic) - 1);
    msg.topic[sizeof(msg.topic) - 1] = '\0'; // Null-terminate

    msg.length = strlen(formatted_msg);
    // 포맷된 메시지 길이만큼 메모리 할당
    msg.payload = (char*)heap_caps_malloc(msg.length + 1, MALLOC_CAP_8BIT);
    if (msg.payload == nullptr) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to allocate payload for status message.");
        return;
    }

    // 포맷된 메시지 복사
    memcpy(msg.payload, formatted_msg, msg.length + 1); // +1 for null terminator

    // 큐에 메시지 전송
    if (xQueueSend(_mqttPublishQueue, &msg, portMAX_DELAY) != pdPASS) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to send status message to queue!");
        heap_caps_free(msg.payload);
    } else {
        DEBUG_PRINTF("NetworkManager: Status message enqueued: '%s'\n", formatted_msg);
    }
}

bool NetworkManager::publishWeight(const char* encodedWeights, size_t length) {
    MqttPublishMessage_t msg;
    strncpy(msg.topic, _mqttTopicWeight, sizeof(msg.topic) - 1);
    msg.topic[sizeof(msg.topic) - 1] = '\0'; // Null-terminate
    msg.length = length;
    // heap_caps_malloc을 사용하여 DRAM에 할당
    msg.payload = (char*)heap_caps_malloc(length + 1, MALLOC_CAP_8BIT); // 페이로드 메모리 할당 (널 종료 포함)
    if (msg.payload == nullptr) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to allocate payload for weight message.");
        return false;
    }
    memcpy(msg.payload, encodedWeights, length);
    msg.payload[length] = '\0'; // Ensure null-terminated

    // 큐에 메시지 전송
    if (xQueueSend(_mqttPublishQueue, &msg, portMAX_DELAY) != pdPASS) { // 큐가 찰 때까지 무한정 대기
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to send weight message to queue!");
        heap_caps_free(msg.payload); // 큐 전송 실패 시 할당된 메모리 해제
        return false;
    } else {
        DEBUG_PRINTF("NetworkManager: Weight message enqueued (length %u).\n", length);
        return true; // 큐에 성공적으로 넣었음을 알림
    }
}

void NetworkManager::publishTrainingResult(float loss) {
    char formatted_msg[96];
    int written = snprintf(formatted_msg, sizeof(formatted_msg),
                           "{\"Status\": \"TRAINING_RESULT\", \"DataSet ID\": %d, \"Loss\": %.6f}",
                           SELECTED_DATASET_ID, loss);
    if (written < 0 || written >= static_cast<int>(sizeof(formatted_msg))) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Training result message truncated.");
        return;
    }

    MqttPublishMessage_t msg;
    strncpy(msg.topic, _mqttTopicStatus, sizeof(msg.topic) - 1);
    msg.topic[sizeof(msg.topic) - 1] = '\0';
    msg.length = static_cast<size_t>(written);
    msg.payload = (char*)heap_caps_malloc(msg.length + 1, MALLOC_CAP_8BIT);
    if (msg.payload == nullptr) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to allocate payload for training result message.");
        return;
    }
    memcpy(msg.payload, formatted_msg, msg.length + 1);

    if (xQueueSend(_mqttPublishQueue, &msg, portMAX_DELAY) != pdPASS) {
        DEBUG_PRINTLN("NetworkManager: ERROR: Failed to enqueue training result message.");
        heap_caps_free(msg.payload);
    } else {
        DEBUG_PRINTF("NetworkManager: Training result enqueued: '%s'\n", formatted_msg);
    }
}