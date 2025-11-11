#ifndef FL_CLIENT_H
#define FL_CLIENT_H
#pragma once

#include <Arduino.h>
#include "../common/config.h"
#include "network/network.h" // NetworkManager 포함
#include "model.h"   // ModelManager 포함

// FLClient는 NetworkManager의 MQTT 콜백으로 사용될 정적(static) 콜백 래퍼 함수를 필요로 할 수 있습니다.
// 또는 FLClient를 캡슐화하는 방식으로 콜백을 구현할 수도 있습니다.
// 여기서는 간단히 FLClient 인스턴스를 멤버로 갖는 함수 포인터와 람다 캡처를 고려합니다.
// PubSubClient 라이브러리의 콜백은 static 또는 전역 함수여야 하므로,
// 클래스 메서드를 콜백으로 사용하기 위한 트릭이 필요합니다.

class FLClient {
public:
    FLClient(); // 생성자

    // 클라이언트 초기화: 네트워크, 모델 등 모든 구성요소를 시작합니다.
    ErrorCode begin();

    // 메인 루프에서 주기적으로 호출되어 클라이언트의 상태를 업데이트하고 필요한 작업을 수행합니다.
    void loop();

    // MQTT 콜백 함수 (NetworkManager에서 호출될 예정)
    // static으로 선언하고 FLClient 인스턴스에 대한 포인터를 저장하여 간접적으로 멤버 함수를 호출합니다.
    static void staticMqttCallback(char* topic, byte* payload, unsigned int length);

private:
    NetworkManager _networkManager; // 네트워크 관리자 인스턴스
    ModelManager   _modelManager;   // 모델 관리자 인스턴스

    ClientStatus _currentStatus; // 현재 클라이언트 상태
    volatile bool _startLocalTraining; // SOT 명령 수신 플래그 (volatile 필요)

    volatile bool _hasPendingGlobal = false;
    String _pendingGlobalB64;

    // 이 클래스의 인스턴스를 static 콜백에서 접근하기 위한 포인터
    static FLClient* _instance;

    // 실제 MQTT 메시지 처리 멤버 함수
    void handleMqttMessage(char* topic, byte* payload, unsigned int length);

    // 클라이언트 상태 업데이트 및 로그 출력
    void updateStatus(ClientStatus newStatus);
};

#endif // FL_CLIENT_H