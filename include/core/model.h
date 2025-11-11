#ifndef MODEL_H
#define MODEL_H
#pragma once

#include <Arduino.h> // for ps_malloc, Serial etc.
#include <aifes.h>   // AIfES 라이브러리
#include <Base64.h>  // 가중치 인코딩을 위해 필요
#include "mbedtls/base64.h"  // 디코딩용 추가
#include "common/config.h"
#include "common/data/data_api.h"


class ModelManager {
public:
    // 생성자: 멤버 변수 초기화
    ModelManager();

    // 소멸자: 할당된 PSRAM 메모리 해제
    ~ModelManager();

    // 모델을 설정하고 필요한 모든 PSRAM 메모리를 할당합니다.
    // 성공 시 SUCCESS, 실패 시 해당 ErrorCode 반환.
    ErrorCode setupModel();

    // 모델 훈련을 시작합니다.
    // 훈련 중 상태 메시지를 시리얼로 출력합니다.
    float trainModel();

    // 현재 모델의 가중치를 직렬화하고 Base64로 인코딩하여 반환합니다.
    // 메모리 할당 실패 시 빈 문자열을 반환합니다.
    String getEncodedWeights();

    // (TODO) 글로벌 모델 가중치를 디코딩하여 현재 모델에 적용합니다.
    // 이 메서드는 MQTT_TOPIC_IN으로 수신된 데이터를 처리하는 데 사용됩니다.
    bool applyGlobalWeights(const String& encodedWeights);

    // 현재 파라미터 버퍼의 앞/뒤 값을 디버깅 용도로 출력합니다.
    void logParameterPreview(const char* label = nullptr) const;

private:
    // AIfES 라이브러리에서 사용할 모델 레이어 및 관련 객체 포인터들
    ailayer_input_f32_t* _inputLayer;
    ailayer_dense_f32_t* _denseLayers[N_HIDDEN];
    ailayer_leaky_relu_f32_t* _actLayers[N_HIDDEN];
    ailayer_dense_f32_t* _outputLayer;
    ailayer_softmax_f32_t* _softmaxLayer;
    ailoss_crossentropy_t* _lossLayer;
    aimodel_t* _model;
    aiopti_t* _optimizer;
    aiopti_adam_f32_t _adam_opti; // Adam 옵티마이저 설정을 위한 구조체

    // 모델 파라미터(가중치, 바이어스)와 훈련 중 중간 결과 저장을 위한 PSRAM 버퍼
    uint8_t* _parameterMemory;
    uint8_t* _trainMemory;
    uint32_t _parameterSize;
    uint32_t _trainMemorySize;

    // 입력 레이어의 shape 데이터를 저장할 PSRAM 버퍼 추가
    uint16_t* _inputShapeData;

    // AIfES 레이어 및 모델 구조체에 필요한 PSRAM 메모리를 할당합니다.
    // 성공 시 true, 실패 시 false 반환.
    bool allocateAifesLayerMemory();
    static bool decodeBase64(const String& in, uint8_t** out_buf, size_t* out_len);

    // allocateAifesLayerMemory()에서 할당된 모든 메모리를 해제합니다.
    void deallocateAifesLayerMemory();

    bool findNonFiniteParameter(size_t* outIndex, float* outValue) const;
};

#endif // MODEL_H