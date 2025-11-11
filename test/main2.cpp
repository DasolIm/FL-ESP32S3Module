

#include <Arduino.h>
#include "data/data_0.h"
#include <aifes.h>
#include <Base64.h>
#include <WiFi.h>
#include <PubSubClient.h>

// ----------------------------- Configuration -----------------------------
#define WIFI_SSID         "ipTimeA1004NS"
#define WIFI_PASSWORD     "01066638380"
#define MQTT_SERVER       "192.168.100.20"
#define MQTT_PORT         1883
#define TOPIC_SOT         "fl/command/sot"
#define TOPIC_WEIGHT_FMT  "fl/%s/weight"
#define TOPIC_STATUS_FMT   "fl/%s/status"
#define MQTT_TOPIC_IN     "fl/global/wight"
#define MQTT_BUFFER_SIZE  32768

// Training parameters
#define TRAIN_EPOCHS      5
#define BATCH_SIZE        8

// Model configuration
#define N_HIDDEN          3
static const uint8_t HIDDEN_NEURONS[N_HIDDEN] = {32, 32, 32};

// Globals
WiFiClient    wifiClient;
PubSubClient  mqttClient(wifiClient);
char          mqttTopicWeight[32];
char          mqttTopicStatus[32];
volatile bool startLocalTraining = false;

// AIfES objects
static ailayer_input_f32_t*      inputLayer;
static ailayer_dense_f32_t*      denseLayers[N_HIDDEN];
static ailayer_batch_norm_f32_t* bnLayers[N_HIDDEN];
static ailayer_leaky_relu_f32_t* actLayers[N_HIDDEN];
static ailayer_dense_f32_t*      outputLayer;
static ailayer_softmax_f32_t*    softmaxLayer;
static ailoss_crossentropy_t*    lossLayer;
static aimodel_t*                model;
static aiopti_t*                 optimizer;
static aiopti_adam_f32_t         adam_opti; // 옵티마이저 설정을 위한 구조체

// PSRAM buffers
static uint8_t* parameterMemory;
static uint8_t* trainMemory;

// Wi-Fi 연결
void setup_wifi() {
  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.printf("\nWiFi connected: %s\n", WiFi.localIP().toString().c_str());
}

// MQTT 콜백
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  payload[length] = '\0';
  if (String(topic) == TOPIC_SOT) {
    startLocalTraining = true;
    Serial.println("Starting local training");
  }
}

// 훈련 루틴
void trainModel() {
  Serial.println("\n===== Starting Local Training =====");
  for (int e = 0; e < TRAIN_EPOCHS; e++) {
    yield();
    float loss = 0;
    unsigned t0 = millis();
    aialgo_train_model(model, &x_tensor, &y_tensor, optimizer, BATCH_SIZE);
    aialgo_calc_loss_model_f32(model, &x_tensor, &y_tensor, &loss);
    loss /= float(N_DATAS) * float(BATCH_SIZE);
    Serial.printf("Epoch %d: loss=%.6f, time=%dms\n", e+1, loss, millis()-t0);
  }
  Serial.println("===== Training Complete =====\n");
}

// 가중치 덤핑 + Base64 인코딩 + MQTT 발행
void publishWeight() {
  // 1) 총 바이트 계산
  uint32_t total_bytes = 0;
  for (int i = 0; i < N_HIDDEN; i++) {
    auto w = &denseLayers[i]->weights;
    uint32_t cnt = 1;
    for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
    total_bytes += cnt * sizeof(float);
  }
  { // output layer
    auto w = &outputLayer->weights;
    uint32_t cnt = 1;
    for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
    total_bytes += cnt * sizeof(float);
  }

  // 2) PSRAM에서 버퍼 할당
  float* buf = (float*) ps_malloc(total_bytes);
  float* p   = buf;

  // 3) 복사
  for (int i = 0; i < N_HIDDEN; i++) {
    auto w = &denseLayers[i]->weights;
    uint32_t cnt = 1;
    for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
    memcpy(p, w->data, cnt * sizeof(float));
    p += cnt;
  }
  { // output
    auto w = &outputLayer->weights;
    uint32_t cnt = 1;
    for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
    memcpy(p, w->data, cnt * sizeof(float));
  }

  // 4) Base64 인코딩
  String b64 = base64::encode((uint8_t*)buf, total_bytes);
  Serial.printf("Encoded data memory size : %u bytes\n", b64.length()*sizeof(char));

  // 5) MQTT 발행
  if (!mqttClient.connected()) {
    if (mqttClient.connect(WiFi.macAddress().c_str())) {
      mqttClient.subscribe(TOPIC_SOT);
      // TOPIC_IN (fl/global) 구독 추가 - 글로벌 모델 업데이트를 받아야 하므로 중요!
      mqttClient.subscribe(MQTT_TOPIC_IN, 2);
    }
  }
  boolean pub_state = mqttClient.publish(mqttTopicWeight, b64.c_str());
  Serial.printf("▶ Weights published %d (%u bytes → %u chars)\n", pub_state, total_bytes, b64.length());

  heap_caps_free(buf);
}

void setup() {
  // PSRAM 초기화
  if (!psramFound()) {
    Serial.println(F("ERROR: PSRAM not found!"));
    while (1);
  }
  Serial.println(F("PSRAM found"));

  // 시리얼 모니터 초기화
  Serial.begin(115200);
  while (!Serial) delay(10);
  delay(500);
  Serial.println("AIfES + PSRAM + MQTT Demo");

  // Wi-Fi ▶ MQTT 초기화
  setup_wifi();
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqtt_callback);
  mqttClient.setBufferSize(MQTT_BUFFER_SIZE);
  snprintf(mqttTopicWeight, sizeof(mqttTopicWeight),
           TOPIC_WEIGHT_FMT, WiFi.macAddress().c_str());
  snprintf(mqttTopicStatus, sizeof(mqttTopicStatus),
           TOPIC_STATUS_FMT, WiFi.macAddress().c_str());

  // 1) AIfES 모델 정의 및 레이어 PSRAM 할당
  uint16_t in_shape[2] = {BATCH_SIZE, INPUT_SHAPE}; // 입력 레이어를 배치 크기만큼 처리하도록 변경

  // inputLayer PSRAM 할당
  inputLayer = (ailayer_input_f32_t*)ps_malloc(sizeof(ailayer_input_f32_t));
  Serial.printf("Allocated inputLayer at PSRAM address: 0x%X, size: %u bytes\n", (uint32_t)inputLayer, sizeof(ailayer_input_f32_t));
  if (inputLayer == nullptr) { Serial.println(F("ERROR: Failed to allocate inputLayer in PSRAM!")); while(1); }
  *inputLayer = AILAYER_INPUT_F32_A(2, in_shape); // 할당된 메모리에 값 초기화

  // denseLayers, bnLayers, actLayers PSRAM 할당
  for (int i = 0; i < N_HIDDEN; i++) {
    denseLayers[i] = (ailayer_dense_f32_t*)ps_malloc(sizeof(ailayer_dense_f32_t));
    *denseLayers[i] = AILAYER_DENSE_F32_A(HIDDEN_NEURONS[i]);

    bnLayers[i] = (ailayer_batch_norm_f32_t*)ps_malloc(sizeof(ailayer_batch_norm_f32_t));
    *bnLayers[i] = AILAYER_BATCH_NORM_F32_A(0.1f, 0.00005f);

    actLayers[i] = (ailayer_leaky_relu_f32_t*)ps_malloc(sizeof(ailayer_leaky_relu_f32_t));
    *actLayers[i] = AILAYER_LEAKY_RELU_F32_A();
  }

  // outputLayer PSRAM 할당
  outputLayer = (ailayer_dense_f32_t*)ps_malloc(sizeof(ailayer_dense_f32_t));
  Serial.printf("Allocated outputLayer at PSRAM address: 0x%X, size: %u bytes\n", (uint32_t)outputLayer, sizeof(ailayer_dense_f32_t));
  if (outputLayer == nullptr) { Serial.println(F("ERROR: Failed to allocate outputLayer in PSRAM!")); while(1); }
  *outputLayer = AILAYER_DENSE_F32_A(TARGET_SHAPE);

  // softmaxLayer PSRAM 할당
  softmaxLayer = (ailayer_softmax_f32_t*)ps_malloc(sizeof(ailayer_softmax_f32_t));
  Serial.printf("Allocated softmaxLayer at PSRAM address: 0x%X, size: %u bytes\n", (uint32_t)softmaxLayer, sizeof(ailayer_softmax_f32_t));
  if (softmaxLayer == nullptr) { Serial.println(F("ERROR: Failed to allocate softmaxLayer in PSRAM!")); while(1); }
  *softmaxLayer = AILAYER_SOFTMAX_F32_A();

  // lossLayer PSRAM 할당
  lossLayer = (ailoss_crossentropy_t*)ps_malloc(sizeof(ailoss_crossentropy_t));
  Serial.printf("Allocated lossLayer at PSRAM address: 0x%X, size: %u bytes\n", (uint32_t)lossLayer, sizeof(ailoss_crossentropy_t));
  if (lossLayer == nullptr) { Serial.println(F("ERROR: Failed to allocate lossLayer in PSRAM!")); while(1); }

  // model struct PSRAM 할당 (이전과 동일)
  model = (aimodel_t*)ps_malloc(sizeof(aimodel_t));
  Serial.printf("Allocated model struct at PSRAM address: 0x%X, size: %u bytes\n", (uint32_t)model, sizeof(aimodel_t));
  if (model == nullptr) { Serial.println(F("ERROR: Failed to allocate model struct in PSRAM!")); while(1); }

  // 모델 연결 (이제 포인터 변수 자체를 전달합니다)
  model->input_layer = ailayer_input_f32_default(inputLayer);
  ailayer_t *x;     // Layer object from AIfES to connect the layers
  for (int i = 0; i < N_HIDDEN; i++) {
    if (i == 0) {
      x = ailayer_dense_f32_default(denseLayers[i], model->input_layer);
    } else {
      x = ailayer_dense_f32_default(denseLayers[i], x);
    }
    x = ailayer_batch_norm_f32_default(bnLayers[i], x);
    x = ailayer_leaky_relu_f32_default(actLayers[i], x);
  }
  x = ailayer_dense_f32_default(outputLayer, x);
  x = ailayer_softmax_f32_default(softmaxLayer, x);
  model->output_layer = x;

  model->loss = ailoss_crossentropy_f32_default(lossLayer, model->output_layer);
  aialgo_compile_model(model); // Compile the AIfES model

  Serial.println(F("-------------- Model structure ---------------"));
  Serial.println(F("Layer:"));
  aialgo_print_model_structure(model);
  Serial.print(F("\nLoss: "));
  aialgo_print_loss_specs(model->loss);
  Serial.println(F("\n----------------------------------------------\n"));

  // 2) PSRAM에 파라미터 메모리 할당 (이전과 동일)
  uint32_t psize = aialgo_sizeof_parameter_memory(model);
  Serial.print(F("Required memory for parameter (Weights, Biases): "));
  Serial.print(psize);
  Serial.print(F(" bytes"));
  Serial.println();
  parameterMemory = (uint8_t*) ps_malloc(psize);
  Serial.printf("Allocated parameterMemory at PSRAM address: 0x%X, size: %u bytes\n", (uint32_t)parameterMemory, psize);
  if (parameterMemory == nullptr) {
      Serial.println(F("ERROR: Failed to allocate parameterMemory in PSRAM!"));
      while(1);
  }
  aialgo_distribute_parameter_memory(model, parameterMemory, psize);
  for (int i = 0; i < N_HIDDEN; i++) {
    aimath_f32_default_init_he_uniform(&denseLayers[i]->weights); // 포인터 접근
    aimath_f32_default_init_he_uniform(&denseLayers[i]->bias);    // 포인터 접근
  }
  aimath_f32_default_init_he_uniform(&outputLayer->weights);     // 포인터 접근
  aimath_f32_default_init_he_uniform(&outputLayer->bias);       // 포인터 접근

  // 3) 트레이닝 메모리 + 옵티마이저 (이전과 동일)
  adam_opti = AIOPTI_ADAM_F32(0.0005f,0.9f,0.999f,1e-8f);
  optimizer   = aiopti_adam_f32_default(&adam_opti); // 전역 optimizer 변수에 할당
  uint32_t tsize = aialgo_sizeof_training_memory(model, optimizer);
  Serial.print(F("Required memory for the training (Intermediate results, gradients, optimization memory): "));
  Serial.print(tsize);
  Serial.print(F(" bytes"));
  Serial.println();
  trainMemory = (uint8_t*) ps_malloc(tsize);
  Serial.printf("Allocated trainMemory at PSRAM address: 0x%X, size: %u bytes\n", (uint32_t)trainMemory, tsize);
  if (trainMemory == nullptr) {
      Serial.println(F("ERROR: Failed to allocate trainMemory in PSRAM!"));
      while(1);
  }

  // AIfES 라이브러리에서 제공하는 메모리 부족 오류 메시지를 그대로 사용
  if(parameterMemory == nullptr || trainMemory == nullptr || model == nullptr || inputLayer == nullptr || outputLayer == nullptr || softmaxLayer == nullptr || lossLayer == nullptr){
    Serial.println(F("ERROR: Not enough memory (RAM) available for training! Try to use another optimizer (e.g. SGD) or make your net smaller."));
    while(1);
  }
  for(int i = 0; i < N_HIDDEN; ++i) {
      if (denseLayers[i] == nullptr || bnLayers[i] == nullptr || actLayers[i] == nullptr) {
          Serial.println(F("ERROR: Not enough memory (RAM) available for hidden layers!"));
          while(1);
      }
  }


  aialgo_schedule_training_memory(model, optimizer, trainMemory, tsize);
  aialgo_init_model_for_training(model, optimizer);

  Serial.println("Setup complete. Awaiting SOT...");
}

void loop() {
  // MQTT 재연결 & 루프
  if (!mqttClient.connected()) {
    if (mqttClient.connect(WiFi.macAddress().c_str())) {
      mqttClient.subscribe(TOPIC_SOT);
      // TOPIC_IN (fl/global) 구독 추가 - 글로벌 모델 업데이트를 받아야 하므로 중요!
      mqttClient.subscribe(MQTT_TOPIC_IN, 2);

      mqttClient.publish(mqttTopicStatus, "READY"); // 이 토픽은 가중치 전송용이므로 "READY"는 다른 토픽으로 보내는 것이 좋습니다.
                                                  // 예: "fl/%s/status" 같은 새로운 토픽
      Serial.println("MQTT connected & READY published");
    } else {
      Serial.printf("MQTT connection failed, rc=%d. Retrying in 2 seconds...\n", mqttClient.state());
      delay(2000);
      return;
    }
  }
  mqttClient.loop();

  // SOT 수신 시 → 훈련 + 가중치 발행
  if (startLocalTraining) {
    trainModel();
    publishWeight();
    startLocalTraining = false;
  }
}