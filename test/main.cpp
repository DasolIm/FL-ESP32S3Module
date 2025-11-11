// src/main.cpp (Arduino / PlatformIO)

#include <Arduino.h>
#include "../src/data/data_0.h"
#include <aifes.h>
#include <Base64.h>
#include <WiFi.h>
#include <PubSubClient.h>

// ----------------------------- Configuration -----------------------------
#define WIFI_SSID        "ipTimeA1004NS"
#define WIFI_PASSWORD    "01066638380"
#define MQTT_SERVER      "192.168.100.20"
#define MQTT_PORT        1883
#define MQTT_TOPIC_WEIGHT  "fl/%s/weight"
#define MQTT_TOPIC_SOT     "fl/command/sot"
#define MQTT_BUFFER_SIZE   16384  // adjust as needed

// Training parameters
#define TRAIN_EPOCHS     5
#define BATCH_SIZE       8

// Model configuration
#define N_HIDDEN         3
static const uint8_t HIDDEN_NEURONS[N_HIDDEN] = {32, 32, 32};

// MQTT client
WiFiClient    wifiClient;
PubSubClient  mqttClient(wifiClient);
char          mqttTopicOut[32];
volatile bool startLocalTraining = false;

// AIfES objects
static ailayer_input_f32_t      inputLayer;
static ailayer_dense_f32_t      denseLayers[N_HIDDEN];
static ailayer_batch_norm_f32_t bnLayers[N_HIDDEN];
static ailayer_leaky_relu_f32_t actLayers[N_HIDDEN];
static ailayer_dense_f32_t      outputLayer;
static ailayer_softmax_f32_t    softmaxLayer;
static ailoss_crossentropy_t    lossLayer;
static aimodel_t*               model;
static aiopti_t*                optimizer;

// Memory buffers
static uint8_t* parameterMemory;
static uint8_t* trainMemory;



void setup() {
  Serial.begin(115200);
  while (!Serial);
  delay(1000);
  Serial.println("AIfES training demo");

  // ---------------------------------- Layer definition ---------------------------------------
  uint16_t shape2d[2] = {1, INPUT_SHAPE};
  inputLayer = AILAYER_INPUT_F32_A( /*input dimension=*/ 2, /*input shape=*/ shape2d);   // Creation of the AIfES input layer
  for (int i = 0; i < N_HIDDEN; i++) {
    denseLayers[i] = AILAYER_DENSE_F32_A( /*neurons=*/ HIDDEN_NEURONS[i]); // Creation of the AIfES hidden dense layer with 3 neurons
    bnLayers[i] = AILAYER_BATCH_NORM_F32_A(/*momentum*/ 0.1, /*eps*/0.00005f); // Hidden batch normalization layer
    actLayers[i] = AILAYER_LEAKY_RELU_F32_A(); // Hidden activation function
  }
  outputLayer = AILAYER_DENSE_F32_A( /*neurons=*/ 26); // Creation of the AIfES output dense layer with 1 neuron
  softmaxLayer = AILAYER_SOFTMAX_F32_A(); // Output activation function

  ailayer_t *x;     // Layer object from AIfES to connect the layers
  model = new aimodel_t;

  model->input_layer = ailayer_input_f32_default(&inputLayer);
  for (int i = 0; i < N_HIDDEN; i++) {
    if (i == 0) {
      x = ailayer_dense_f32_default(&denseLayers[i], model->input_layer);
    } else {
      x = ailayer_dense_f32_default(&denseLayers[i], x);
    }
    x = ailayer_batch_norm_f32_default(&bnLayers[i], x);
    x = ailayer_leaky_relu_f32_default(&actLayers[i], x);
  }
  x = ailayer_dense_f32_default(&outputLayer, x);
  x = ailayer_softmax_f32_default(&softmaxLayer, x);
  model->output_layer = x;

  model->loss = ailoss_crossentropy_f32_default(&lossLayer, model->output_layer);
  aialgo_compile_model(model); // Compile the AIfES model

  Serial.println(F("-------------- Model structure ---------------"));
  Serial.println(F("Layer:"));
  aialgo_print_model_structure(model);
  Serial.print(F("\nLoss: "));
  aialgo_print_loss_specs(model->loss);
  Serial.println(F("\n----------------------------------------------\n"));

  uint32_t psize = aialgo_sizeof_parameter_memory(model);
  Serial.print(F("Required memory for parameter (Weights, Biases): "));
  Serial.print(psize);
  Serial.print(F(" bytes"));
  Serial.println();
  parameterMemory = (uint8_t*)malloc(psize);

  // Distribute the memory for the trainable parameters of the model
  aialgo_distribute_parameter_memory(model, parameterMemory, psize);

  for (int i = 0; i < N_HIDDEN; i++) {
    aimath_f32_default_init_he_uniform(&denseLayers[i].weights);
    aimath_f32_default_init_he_uniform(&denseLayers[i].bias);
  }
  aimath_f32_default_init_he_uniform(&outputLayer.weights);
  aimath_f32_default_init_he_uniform(&outputLayer.bias);

  aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(/*learning rate=*/ 0.0005f, /*beta_1=*/ 0.9f, /*beta_2=*/ 0.999f, /*eps=*/ 1e-8);
  aiopti_t *optimizer = aiopti_adam_f32_default(&adam_opti); // Initialize the optimizer

  uint32_t tsize = aialgo_sizeof_training_memory(model, optimizer);
  Serial.print(F("Required memory for the training (Intermediate results, gradients, optimization memory): "));
  Serial.print(tsize);
  Serial.print(F(" bytes"));
  Serial.println();
  trainMemory = (uint8_t*)malloc(tsize);

  if(parameterMemory == 0){
    Serial.println(F("ERROR: Not enough memory (RAM) available for training! Try to use another optimizer (e.g. SGD) or make your net smaller."));
    while(1);
  }

  // Schedule the memory over the model
  aialgo_schedule_training_memory(model, optimizer, trainMemory, tsize);

  // IMPORTANT: Initialize the AIfES model before training
  aialgo_init_model_for_training(model, optimizer);

  // ------------------------------------- Training configuration ------------------------------------

  uint32_t batch_size = 8; // Configuration tip: ADAM=4   / SGD=1
  uint16_t epochs = 5;   // Configuration tip: ADAM=100 / SGD=550

  Serial.println(F("\n------------ Training configuration ----------"));
  Serial.print(F("Epochs: "));
  Serial.print(epochs);
  Serial.println(F(" epochs)"));
  Serial.print(F("Batch size: "));
  Serial.println(batch_size);
  Serial.print(F("Optimizer: "));
  aialgo_print_optimizer_specs(optimizer);
  Serial.println(F("\n----------------------------------------------\n"));

  // ------------------------------------- Run the training ------------------------------------

  Serial.println(F("Start training"));

  unsigned total_time = 0;
  for(int i = 0; i < epochs; i++)
  {
    float loss = 0;
    unsigned prev_time = millis();
    // One epoch of training. Iterates through the whole data once
    aialgo_train_model(model, &x_tensor, &y_tensor, optimizer, batch_size);
    aialgo_calc_loss_model_f32(model, &x_tensor, &y_tensor, &loss);
    loss = loss / float(N_DATAS)/ float(batch_size); // Average loss over the batch size
    unsigned time = (millis() - prev_time);
    total_time += time;
    Serial.print(F("Epoch: "));
    Serial.print(i);
    Serial.print(F(" Loss: "));
    Serial.print(loss, 8);
    Serial.print(F(" Time: "));
    Serial.print(time);
    Serial.println(F(" ms"));

  }
  Serial.print(F("Finished training"));
  Serial.print(F(" Total time: "));
  Serial.print(total_time);
  Serial.println(F(" ms"));
  Serial.print(F(" Average time per epoch: "));
  Serial.print(float(total_time) / float(epochs));
  Serial.println(F(" ms"));

  // How to print the weights example
  Serial.println(F("Dense 1 - Weights:"));
  print_aitensor(&denseLayers[0].weights);
  Serial.println(F("Dense 1 - Bias:"));
  print_aitensor(&denseLayers[0].bias);

  for(int i = 0; i < denseLayers[0].weights.dim; i++) {
    Serial.print(denseLayers[0].weights.shape[i]);
    Serial.print(", ");
  }

  uint32_t total_weight_bytes = 0;
  uint8_t* weight_buffer = nullptr;

  // 히든 레이어
  for (int i = 0; i < N_HIDDEN; ++i) {
    aitensor_t* w = &denseLayers[i].weights;
    uint32_t count = 1;
    for (int d = 0; d < w->dim; ++d) count *= w->shape[d];
    total_weight_bytes += count * sizeof(float);
  }
  // 출력 레이어
  {
    aitensor_t* w = &outputLayer.weights;
    uint32_t count = 1;
    for (int d = 0; d < w->dim; ++d) count *= w->shape[d];
    total_weight_bytes += count * sizeof(float);
  }

  weight_buffer = (uint8_t*) malloc(total_weight_bytes);
  if (!weight_buffer) {
    Serial.println(F("ERROR: Not enough heap for weight buffer!"));
    return;
  }

  // 2) 가중치들을 순차적으로 복사
  uint8_t* ptr = weight_buffer;

  // 히든 레이어
  for (int i = 0; i < N_HIDDEN; ++i) {
    aitensor_t* w = &denseLayers[i].weights;
    uint32_t count = 1;
    for (int d = 0; d < w->dim; ++d) count *= w->shape[d];
    float* data = (float*) w->data;
    memcpy(ptr, data, count * sizeof(float));
    ptr += count * sizeof(float);
  }

  // 출력 레이어
  aitensor_t* w = &outputLayer.weights;
  uint32_t count = 1;
  for (int d = 0; d < w->dim; ++d) count *= w->shape[d];
  float* data = (float*) w->data;
  memcpy(ptr, data, count * sizeof(float));
  ptr += count * sizeof(float);

  // Base64 인코딩
  String b64 = base64::encode(weight_buffer, total_weight_bytes);

  // 3) 바이너리 전송
  Serial.println(F("\n--- Sending compressed weight buffer as Base64 ---"));
  Serial.println(b64);
  Serial.println(F("\n--- Size of Base64 encoded data ---"));
  Serial.print(b64.length());
  Serial.println(F("Training complete. Weights sent successfully."));


  // 4) 사용 후 해제
  free(weight_buffer);
  weight_buffer = nullptr;

}

void loop() {
  // Idle after training
}


// #include <Arduino.h>
// #include "data/data_0.h"
// #include <aifes.h>
// #include <Base64.h>
// #include <WiFi.h>
// #include <PubSubClient.h>
//
// // ----------------------------- Configuration -----------------------------
// #define WIFI_SSID         "ipTimeA1004NS"
// #define WIFI_PASSWORD     "01066638380"
// #define MQTT_SERVER       "192.168.100.20"
// #define MQTT_PORT         1883
// #define TOPIC_SOT         "fl/command/sot"
// #define TOPIC_WEIGHT_FMT  "fl/%s/weight"
// #define MQTT_BUFFER_SIZE  16384
//
// // Training parameters
// #define TRAIN_EPOCHS      5
// #define BATCH_SIZE        8
//
// // Model configuration
// #define N_HIDDEN          3
// static const uint8_t HIDDEN_NEURONS[N_HIDDEN] = {32, 32, 32};
//
// // Globals
// WiFiClient    wifiClient;
// PubSubClient  mqttClient(wifiClient);
// char          mqttTopicWeight[32];
// volatile bool startLocalTraining = false;
//
// // AIfES objects
// static ailayer_input_f32_t      inputLayer;
// static ailayer_dense_f32_t      denseLayers[N_HIDDEN];
// static ailayer_batch_norm_f32_t bnLayers[N_HIDDEN];
// static ailayer_leaky_relu_f32_t actLayers[N_HIDDEN];
// static ailayer_dense_f32_t      outputLayer;
// static ailayer_softmax_f32_t    softmaxLayer;
// static ailoss_crossentropy_t    lossLayer;
// static aimodel_t*               model;
// static aiopti_t*                optimizer;
//
// // PSRAM buffers
// static uint8_t* parameterMemory;
// static uint8_t* trainMemory;
//
// // Wi-Fi 연결
// void setup_wifi() {
//   Serial.printf("Connecting to %s", WIFI_SSID);
//   WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
//   while (WiFi.status() != WL_CONNECTED) {
//     delay(500);
//     Serial.print('.');
//   }
//   Serial.printf("\nWiFi connected: %s\n", WiFi.localIP().toString().c_str());
// }
//
// // MQTT 콜백
// void mqtt_callback(char* topic, byte* payload, unsigned int length) {
//   payload[length] = '\0';
//   if (String(topic) == TOPIC_SOT) {
//     startLocalTraining = true;
//     Serial.println("Starting local training");
//   }
// }
//
// // 훈련 루틴
// void trainModel() {
//   Serial.println("\n===== Local Training =====");
//   for (int e = 0; e < TRAIN_EPOCHS; e++) {
//     float loss = 0;
//     unsigned t0 = millis();
//     aialgo_train_model(model, &x_tensor, &y_tensor, optimizer, BATCH_SIZE);
//     aialgo_calc_loss_model_f32(model, &x_tensor, &y_tensor, &loss);
//     loss /= float(N_DATAS) * float(BATCH_SIZE);
//     Serial.printf("Epoch %d: loss=%.6f, time=%dms\n", e+1, loss, millis()-t0);
//   }
//   Serial.println("===== Training Complete =====\n");
// }
//
// // 가중치 덤핑 + Base64 인코딩 + MQTT 발행
// void publishWeight() {
//   // 1) 총 바이트 계산
//   uint32_t total_bytes = 0;
//   for (int i = 0; i < N_HIDDEN; i++) {
//     auto w = &denseLayers[i].weights;
//     uint32_t cnt = 1;
//     for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
//     total_bytes += cnt * sizeof(float);
//   }
//   { // output layer
//     auto w = &outputLayer.weights;
//     uint32_t cnt = 1;
//     for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
//     total_bytes += cnt * sizeof(float);
//   }
//
//   // 2) PSRAM에서 버퍼 할당
//   float* buf = (float*) ps_malloc(total_bytes);
//   float* p   = buf;
//
//   // 3) 복사
//   for (int i = 0; i < N_HIDDEN; i++) {
//     auto w = &denseLayers[i].weights;
//     uint32_t cnt = 1;
//     for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
//     memcpy(p, w->data, cnt * sizeof(float));
//     p += cnt;
//   }
//   { // output
//     auto w = &outputLayer.weights;
//     uint32_t cnt = 1;
//     for (int d = 0; d < w->dim; d++) cnt *= w->shape[d];
//     memcpy(p, w->data, cnt * sizeof(float));
//   }
//
//   // 4) Base64 인코딩
//   String b64 = base64::encode((uint8_t*)buf, total_bytes);
//
//   // 5) MQTT 발행
//   mqttClient.publish(mqttTopicWeight, b64.c_str());
//   Serial.printf("▶ Weights published (%u bytes → %u chars)\n", total_bytes, b64.length());
//
//   free(buf);
// }
//
// void setup() {
//   Serial.begin(115200);
//   while (!Serial) delay(10);
//   delay(500);
//   Serial.println("AIfES + PSRAM + MQTT Demo");
//
//   // Wi-Fi ▶ MQTT 초기화
//   setup_wifi();
//   mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
//   mqttClient.setCallback(mqtt_callback);
//   mqttClient.setBufferSize(MQTT_BUFFER_SIZE);
//   snprintf(mqttTopicWeight, sizeof(mqttTopicWeight),
//            TOPIC_WEIGHT_FMT, WiFi.macAddress().c_str());
//
//   // 1) AIfES 모델 정의
//   uint16_t in_shape[2] = {1, INPUT_SHAPE};
//   inputLayer    = AILAYER_INPUT_F32_A(2, in_shape);
//   for (int i = 0; i < N_HIDDEN; i++) {
//     denseLayers[i] = AILAYER_DENSE_F32_A(HIDDEN_NEURONS[i]);
//     bnLayers[i]    = AILAYER_BATCH_NORM_F32_A(0.1f, 0.00005f);
//     actLayers[i]   = AILAYER_LEAKY_RELU_F32_A();
//   }
//   outputLayer  = AILAYER_DENSE_F32_A(TARGET_SHAPE);
//   softmaxLayer = AILAYER_SOFTMAX_F32_A();
//
//   model = new aimodel_t;
//   model->input_layer = ailayer_input_f32_default(&inputLayer);
//   ailayer_t* lay = model->input_layer;
//   for (int i = 0; i < N_HIDDEN; i++) {
//     lay = ailayer_dense_f32_default(&denseLayers[i], lay);
//     lay = ailayer_batch_norm_f32_default(&bnLayers[i], lay);
//     lay = ailayer_leaky_relu_f32_default(&actLayers[i], lay);
//   }
//   lay = ailayer_dense_f32_default(&outputLayer, lay);
//   lay = ailayer_softmax_f32_default(&softmaxLayer, lay);
//   model->output_layer = lay;
//   model->loss         = ailoss_crossentropy_f32_default(&lossLayer, lay);
//   aialgo_compile_model(model);
//
//   // 2) PSRAM에 파라미터 메모리 할당
//   uint32_t psize = aialgo_sizeof_parameter_memory(model);
//   parameterMemory = (uint8_t*) ps_malloc(psize);
//   aialgo_distribute_parameter_memory(model, parameterMemory, psize);
//   for (int i = 0; i < N_HIDDEN; i++) {
//     aimath_f32_default_init_he_uniform(&denseLayers[i].weights);
//     aimath_f32_default_init_he_uniform(&denseLayers[i].bias);
//   }
//   aimath_f32_default_init_he_uniform(&outputLayer.weights);
//   aimath_f32_default_init_he_uniform(&outputLayer.bias);
//
//   // 3) 트레이닝 메모리 + 옵티마이저
//   aiopti_adam_f32_t cfg = AIOPTI_ADAM_F32(0.0005f,0.9f,0.999f,1e-8f);
//   optimizer   = aiopti_adam_f32_default(&cfg);
//   uint32_t tsize = aialgo_sizeof_training_memory(model, optimizer);
//   trainMemory = (uint8_t*) ps_malloc(tsize);
//   aialgo_schedule_training_memory(model, optimizer, trainMemory, tsize);
//   aialgo_init_model_for_training(model, optimizer);
//
//   Serial.println("Setup complete. Awaiting SOT...");
// }
//
// void loop() {
//   // MQTT 재연결 & 루프
//   if (!mqttClient.connected()) {
//     if (mqttClient.connect(WiFi.macAddress().c_str())) {
//       mqttClient.subscribe(TOPIC_SOT);
//       mqttClient.publish(mqttTopicWeight, "READY");
//       Serial.println("MQTT connected & READY published");
//     } else {
//       delay(2000);
//       return;
//     }
//   }
//   mqttClient.loop();
//
//   // SOT 수신 시 → 훈련 + 가중치 발행
//   if (startLocalTraining) {
//     trainModel();
//     publishWeight();
//     startLocalTraining = false;
//   }
// }