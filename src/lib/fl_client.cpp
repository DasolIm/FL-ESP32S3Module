#include "core/fl_client.h"
#include <cmath>

// 정적 멤버 변수 초기화
FLClient* FLClient::_instance = nullptr;

FLClient::FLClient()
    : _currentStatus(STATUS_INITIALIZING),
      _startLocalTraining(false) {
    _instance = this; // 현재 인스턴스 포인터를 정적 변수에 저장
}

ErrorCode FLClient::begin() {
    DEBUG_PRINTLN("FLClient: Initializing...");
    updateStatus(STATUS_INITIALIZING);

    // 1. 네트워크 매니저 초기화
    // staticMqttCallback 함수 포인터를 NetworkManager에 전달
    ErrorCode netStatus = _networkManager.begin(FLClient::staticMqttCallback);
    if (netStatus != SUCCESS) {
        DEBUG_PRINTF("FLClient: Network Manager initialization failed (Error: %d).\n", netStatus);
        updateStatus(STATUS_ERROR);
        return netStatus;
    }

    // 2. 모델 매니저 초기화
    ErrorCode modelStatus = _modelManager.setupModel();
    if (modelStatus != SUCCESS) {
        DEBUG_PRINTF("FLClient: Model Manager setup failed (Error: %d).\n", modelStatus);
        updateStatus(STATUS_ERROR);
        return modelStatus;
    }

    DEBUG_PRINTLN("FLClient: Initialization complete. Ready to operate.");
    updateStatus(STATUS_READY); // 모든 초기화 성공, READY 상태로 전환
    _networkManager.publishStatus(STATUS_READY); // MQTT로 상태 알림
    return SUCCESS;
}

void FLClient::loop() {
    // _networkManager.loop(); // 이 줄을 제거합니다. 이제 MQTT 루프는 별도 태스크에서 관리됩니다.

    if (_startLocalTraining) {
        _startLocalTraining = false;
        DEBUG_PRINTLN("FLClient: Start Local Training command detected.");

        updateStatus(STATUS_TRAINING);
        _networkManager.publishStatus(STATUS_TRAINING); // 큐에 메시지 넣음

        float finalLoss = _modelManager.trainModel();

        if (std::isfinite(finalLoss)) {
            DEBUG_PRINTF("FLClient: Final epoch loss %.6f\n", finalLoss);
            _networkManager.publishTrainingResult(finalLoss);
        } else {
            DEBUG_PRINTLN("FLClient: WARNING: Final loss is non-finite; training result will not be published.");
        }

        updateStatus(STATUS_SENDING);
        // 가중치 전송 전에 MQTT 연결 상태를 다시 확인할 필요가 없음.
        // publishWeight가 내부적으로 큐에 넣기 때문에, 연결 여부와 관계없이 큐잉 시도.
        // 실제 발행은 태스크에서 연결 확인 후 진행.
        // 다만, 큐잉 시도 자체가 실패할 경우를 대비하여 반환값 확인.

        String encodedWeights = _modelManager.getEncodedWeights();
        if (encodedWeights.length() > 0) {
            bool publish_success = _networkManager.publishWeight(encodedWeights.c_str(), encodedWeights.length());
            if (publish_success) { // publishWeight가 큐에 성공적으로 넣었는지 여부
                DEBUG_PRINTLN("FLClient: Weights enqueued successfully.");
            } else {
                DEBUG_PRINTLN("FLClient: ERROR: Failed to enqueue weights to MQTT queue!");
                // 큐잉 실패 시 바로 ERROR 상태로 전환
                updateStatus(STATUS_ERROR);
                _networkManager.publishStatus(STATUS_ERROR);
            }
        } else {
            DEBUG_PRINTLN("FLClient: ERROR: Failed to get encoded weights for publishing.");
            updateStatus(STATUS_ERROR);
            _networkManager.publishStatus(STATUS_ERROR);
        }

        updateStatus(STATUS_READY);
        _networkManager.publishStatus(STATUS_READY); // 큐에 메시지 넣음
    }

    if (_hasPendingGlobal) {
        // 플래그 먼저 내리고 시작 (중복 방지)
        _hasPendingGlobal = false;

        updateStatus(STATUS_APPLYING);
        _networkManager.publishStatus(STATUS_APPLYING);

        bool ok = _modelManager.applyGlobalWeights(_pendingGlobalB64);
        _pendingGlobalB64 = ""; // 메모리 회수

        if (ok) {
            DEBUG_PRINTLN("FLClient: Global weights applied successfully.");
            updateStatus(STATUS_READY);
            _networkManager.publishStatus(STATUS_READY);
        } else {
            DEBUG_PRINTLN("FLClient: ERROR: Failed to apply global weights.");
            updateStatus(STATUS_ERROR);
            _networkManager.publishStatus(STATUS_ERROR);
        }
    }
}

// PubSubClient 콜백을 위한 정적 래퍼 함수
void FLClient::staticMqttCallback(char* topic, byte* payload, unsigned int length) {
    if (_instance != nullptr) {
        _instance->handleMqttMessage(topic, payload, length);
    } else {
        DEBUG_PRINTLN("FLClient: ERROR: FLClient instance not available for MQTT callback.");
    }
}

// 실제 MQTT 메시지 처리 멤버 함수
void FLClient::handleMqttMessage(char* topic, byte* payload, unsigned int length) {
    payload[length] = '\0'; // 문자열로 처리하기 위해 널 종료 문자 추가
    DEBUG_PRINTF("FLClient: MQTT Message arrived [%s] %s\n", topic, (char*)payload);

    if (String(topic) == TOPIC_SOT) {
        _startLocalTraining = true; // SOT 명령 수신 플래그 설정
    } else if (String(topic) == TOPIC_GLOBAL) {
        // 콜백에서는 복잡한 작업 금지 → 그냥 저장만
        _pendingGlobalB64.reserve(length + 1);
        _pendingGlobalB64 = String((const char*)payload, length);
        _hasPendingGlobal = true;   // loop에서 처리하도록 플래그
    }
    // 다른 MQTT 토픽에 대한 처리 로직 추가
}

void FLClient::updateStatus(ClientStatus newStatus) {
    _currentStatus = newStatus;
    const char* statusStr;
    switch(newStatus) {
        case STATUS_INITIALIZING: statusStr = "INITIALIZING"; break;
        case STATUS_READY:        statusStr = "READY";        break;
        case STATUS_TRAINING:     statusStr = "TRAINING";     break;
        case STATUS_SENDING:      statusStr = "SENDING";      break;
        case STATUS_APPLYING:
            statusStr = "APPLING GLOBAL WEIGHT";              break;
        case STATUS_ERROR:        statusStr = "ERROR";        break;
        default:                  statusStr = "UNKNOWN";      break;
    }
    DEBUG_PRINTF("FLClient: Status updated to %s\n", statusStr);

    if (newStatus == STATUS_READY) {
        _modelManager.logParameterPreview("READY");
    }
}