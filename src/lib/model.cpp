#include "core/model.h"
#include <cmath>
#include <vector>

// 생성자 구현
ModelManager::ModelManager()
        : _inputLayer(nullptr), _outputLayer(nullptr), _softmaxLayer(nullptr),
            _lossLayer(nullptr), _model(nullptr), _optimizer(nullptr), _adam_opti(),
            _parameterMemory(nullptr), _trainMemory(nullptr),
            _inputShapeData(nullptr), _parameterSize(0), _trainMemorySize(0) {

    // 모든 동적 할당 포인터를 nullptr로 초기화합니다.
    for (int i = 0; i < N_HIDDEN; ++i) {
        _denseLayers[i] = nullptr;
        _actLayers[i] = nullptr;
    }
}

// 소멸자 구현: 할당된 모든 PSRAM 메모리를 해제합니다.
ModelManager::~ModelManager() {
    DEBUG_VERBOSE_PRINTLN("ModelManager: Deallocating AIfES layer memory.");
    deallocateAifesLayerMemory(); // AIfES 레이어 구조체 메모리 해제

    // 모델 파라미터와 훈련 메모리 버퍼 해제
    if (_parameterMemory) {
        heap_caps_free(_parameterMemory);
        DEBUG_VERBOSE_PRINTLN("ModelManager: _parameterMemory freed.");
        _parameterMemory = nullptr;
        _parameterSize = 0;
    }
    if (_trainMemory) {
        heap_caps_free(_trainMemory);
        DEBUG_VERBOSE_PRINTLN("ModelManager: _trainMemory freed.");
        _trainMemory = nullptr;
        _trainMemorySize = 0;
    }
}

// AIfES 레이어 및 모델 구조체에 필요한 PSRAM 메모리를 할당하는 내부 도우미 함수
bool ModelManager::allocateAifesLayerMemory() {
    DEBUG_VERBOSE_PRINTLN("ModelManager: Allocating AIfES layer memory...");

    // Input Layer
    _inputShapeData = (uint16_t*)ps_malloc(sizeof(uint16_t) * 2); // 2개의 uint16_t (1, COMMON_INPUT_SHAPE)
    if (_inputShapeData == nullptr) {
        DEBUG_PRINTLN(F("ERROR: Failed to allocate _inputShapeData in PSRAM!"));
        return false;
    }
    _inputShapeData[0] = 1; // 첫 번째 차원 (단일 샘플)
    _inputShapeData[1] = COMMON_INPUT_SHAPE; // 두 번째 차원 (피처 수)

    _inputLayer = (ailayer_input_f32_t*)ps_malloc(sizeof(ailayer_input_f32_t));
    if (_inputLayer == nullptr) { DEBUG_PRINTLN(F("ERROR: Failed to allocate inputLayer in PSRAM!")); return false; }
    DEBUG_VERBOSE_PRINTF("Allocated inputLayer at 0x%X, size: %u bytes\n", (uint32_t)_inputLayer, sizeof(ailayer_input_f32_t));
    *_inputLayer = AILAYER_INPUT_F32_A(2, _inputShapeData);

    // Hidden Layers (Dense, LeakyReLU)
    for (int i = 0; i < N_HIDDEN; i++) {
        _denseLayers[i] = (ailayer_dense_f32_t*)ps_malloc(sizeof(ailayer_dense_f32_t));
        if (_denseLayers[i] == nullptr) { DEBUG_PRINTF("ERROR: Failed to allocate denseLayer[%d] in PSRAM!\n", i); return false; }
        *_denseLayers[i] = AILAYER_DENSE_F32_A(HIDDEN_NEURONS[i]);
        DEBUG_VERBOSE_PRINTF("Allocated denseLayer[%d] at 0x%X, size: %u bytes\n", i, (uint32_t)_denseLayers[i], sizeof(ailayer_dense_f32_t));

        _actLayers[i] = (ailayer_leaky_relu_f32_t*)ps_malloc(sizeof(ailayer_leaky_relu_f32_t));
        if (_actLayers[i] == nullptr) { DEBUG_PRINTF("ERROR: Failed to allocate actLayer[%d] in PSRAM!\n", i); return false; }
        *_actLayers[i] = AILAYER_LEAKY_RELU_F32_A();
        DEBUG_VERBOSE_PRINTF("Allocated actLayer[%d] at 0x%X, size: %u bytes\n", i, (uint32_t)_actLayers[i], sizeof(ailayer_leaky_relu_f32_t));
    }

    // Output Layer
    _outputLayer = (ailayer_dense_f32_t*)ps_malloc(sizeof(ailayer_dense_f32_t));
    if (_outputLayer == nullptr) { DEBUG_PRINTLN(F("ERROR: Failed to allocate outputLayer in PSRAM!")); return false; }
    DEBUG_VERBOSE_PRINTF("Allocated outputLayer at 0x%X, size: %u bytes\n", (uint32_t)_outputLayer, sizeof(ailayer_dense_f32_t));
    *_outputLayer = AILAYER_DENSE_F32_A(COMMON_TARGET_SHAPE);

    // Softmax Layer
    _softmaxLayer = (ailayer_softmax_f32_t*)ps_malloc(sizeof(ailayer_softmax_f32_t));
    if (_softmaxLayer == nullptr) { DEBUG_PRINTLN(F("ERROR: Failed to allocate softmaxLayer in PSRAM!")); return false; }
    DEBUG_VERBOSE_PRINTF("Allocated softmaxLayer at 0x%X, size: %u bytes\n", (uint32_t)_softmaxLayer, sizeof(ailayer_softmax_f32_t));
    *_softmaxLayer = AILAYER_SOFTMAX_F32_A();

    // Loss Layer
    _lossLayer = (ailoss_crossentropy_t*)ps_malloc(sizeof(ailoss_crossentropy_t));
    if (_lossLayer == nullptr) { DEBUG_PRINTLN(F("ERROR: Failed to allocate lossLayer in PSRAM!")); return false; }
    DEBUG_VERBOSE_PRINTF("Allocated lossLayer at 0x%X, size: %u bytes\n", (uint32_t)_lossLayer, sizeof(ailoss_crossentropy_t));

    // Model Structure
    _model = (aimodel_t*)ps_malloc(sizeof(aimodel_t));
    if (_model == nullptr) { DEBUG_PRINTLN(F("ERROR: Failed to allocate model struct in PSRAM!")); return false; }
    DEBUG_VERBOSE_PRINTF("Allocated model struct at 0x%X, size: %u bytes\n", (uint32_t)_model, sizeof(aimodel_t));

    return true;
}

// 할당된 AIfES 레이어 및 모델 구조체 메모리를 해제하는 내부 도우미 함수
void ModelManager::deallocateAifesLayerMemory() {
    // 할당된 역순으로 해제하는 것이 일반적이지만, nullptr 체크가 중요
    if (_lossLayer) { heap_caps_free(_lossLayer); _lossLayer = nullptr; }
    if (_softmaxLayer) { heap_caps_free(_softmaxLayer); _softmaxLayer = nullptr; }
    if (_outputLayer) { heap_caps_free(_outputLayer); _outputLayer = nullptr; }
    for (int i = N_HIDDEN - 1; i >= 0; --i) {
        if (_actLayers[i]) { heap_caps_free(_actLayers[i]); _actLayers[i] = nullptr; }
        if (_denseLayers[i]) { heap_caps_free(_denseLayers[i]); _denseLayers[i] = nullptr; }
    }
    if (_inputLayer) { heap_caps_free(_inputLayer); _inputLayer = nullptr; }
    if (_model) { heap_caps_free(_model); _model = nullptr; }
    if (_inputShapeData) { heap_caps_free(_inputShapeData); _inputShapeData = nullptr; }
}


// 모델 설정 및 PSRAM 할당 함수
ErrorCode ModelManager::setupModel() {
    DEBUG_PRINTLN("ModelManager: Setting up AIfES model...");

    randomSeed(RANDOM_SEED + SELECTED_DATASET_ID * 100);
    srand(RANDOM_SEED + SELECTED_DATASET_ID * 100);
    DEBUG_VERBOSE_PRINTF("ModelManager: Initialized with random seed: %d\n", RANDOM_SEED + SELECTED_DATASET_ID * 100);

    if (!psramFound()) {
        DEBUG_PRINTLN(F("ERROR: PSRAM not found! Model setup failed."));
        return ERROR_PSRAM_INIT;
    }
    DEBUG_PRINTLN(F("PSRAM found."));

    if (!init_dataset_buffers()) {
        DEBUG_PRINTLN(F("WARNING: Failed to allocate dataset buffers in PSRAM; using flash memory."));
    }

    // AIfES 레이어 메모리 할당
    if (!allocateAifesLayerMemory()) {
        DEBUG_PRINTLN(F("ERROR: Failed to allocate AIfES layer memory."));
        // 메모리 할당 실패 시, 부분적으로 할당된 메모리도 해제 시도 (소멸자가 처리)
        return ERROR_MEMORY_ALLOC;
    }

    // 모델 연결 (AIfES 라이브러리 함수 사용)
    _model->input_layer = ailayer_input_f32_default(_inputLayer);
    ailayer_t *x;
    for (int i = 0; i < N_HIDDEN; i++) {
        if (i == 0) {
            x = ailayer_dense_f32_default(_denseLayers[i], _model->input_layer);
        } else {
            x = ailayer_dense_f32_default(_denseLayers[i], x);
        }
        x = ailayer_leaky_relu_f32_default(_actLayers[i], x);
    }
    x = ailayer_dense_f32_default(_outputLayer, x);
    x = ailayer_softmax_f32_default(_softmaxLayer, x);
    _model->output_layer = x;

    _model->loss = ailoss_crossentropy_f32_default(_lossLayer, _model->output_layer);
    aialgo_compile_model(_model); // AIfES 모델 컴파일

    DEBUG_PRINTLN(F("-------------- Model structure ---------------"));
    aialgo_print_model_structure(_model);
    DEBUG_PRINT(F("\nLoss: "));
    aialgo_print_loss_specs(_model->loss);
    DEBUG_PRINTLN(F("\n----------------------------------------------\n"));

    // 모델 파라미터(가중치, 바이어스)를 위한 PSRAM 메모리 할당
    uint32_t psize = aialgo_sizeof_parameter_memory(_model);
    DEBUG_PRINTF("Required memory for parameters (Weights, Biases): %u bytes\n", psize);
    _parameterSize = psize;
    _parameterMemory = (uint8_t*) ps_malloc(psize);
    if (_parameterMemory == nullptr) {
        DEBUG_PRINTLN(F("ERROR: Failed to allocate parameterMemory in PSRAM!"));
        _parameterSize = 0;
        return ERROR_MEMORY_ALLOC;
    }
    DEBUG_VERBOSE_PRINTF("Allocated parameterMemory at 0x%X, size: %u bytes\n", (uint32_t)_parameterMemory, psize);
    aialgo_distribute_parameter_memory(_model, _parameterMemory, psize);

    // 초기 가중치 설정 (He Uniform 초기화)
    for (int i = 0; i < N_HIDDEN; i++) {
        aimath_f32_default_init_he_uniform(&_denseLayers[i]->weights);
        aimath_f32_default_init_he_uniform(&_denseLayers[i]->bias);
    }
    aimath_f32_default_init_he_uniform(&_outputLayer->weights);
    aimath_f32_default_init_he_uniform(&_outputLayer->bias);

    // 훈련 메모리 및 옵티마이저 설정 (Adam 옵티마이저 사용)
    _adam_opti = AIOPTI_ADAM_F32(LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    _optimizer = aiopti_adam_f32_default(&_adam_opti);
    uint32_t tsize = aialgo_sizeof_training_memory(_model, _optimizer);
    DEBUG_PRINTF("Required memory for training (Intermediate results, gradients, optimization memory): %u bytes\n", tsize);
    _trainMemorySize = tsize;
    _trainMemory = (uint8_t*) ps_malloc(tsize);
    if (_trainMemory == nullptr) {
        DEBUG_PRINTLN(F("ERROR: Failed to allocate trainMemory in PSRAM!"));
        _trainMemorySize = 0;
        return ERROR_MEMORY_ALLOC;
    }
    DEBUG_VERBOSE_PRINTF("Allocated trainMemory at 0x%X, size: %u bytes\n", (uint32_t)_trainMemory, tsize);

    // AIfES 라이브러리에서 제공하는 메모리 부족 오류 메시지를 그대로 사용
    if (_parameterMemory == nullptr || _trainMemory == nullptr || _model == nullptr ||
        _inputLayer == nullptr || _outputLayer == nullptr || _softmaxLayer == nullptr || _lossLayer == nullptr){
        DEBUG_PRINTLN(F("ERROR: Not enough memory (RAM) available for training! Try to use another optimizer (e.g. SGD) or make your net smaller."));
        return ERROR_MEMORY_ALLOC;
    }
    for(int i = 0; i < N_HIDDEN; ++i) {
        if (_denseLayers[i] == nullptr || _actLayers[i] == nullptr) {
            DEBUG_PRINTLN(F("ERROR: Not enough memory (RAM) available for hidden layers!"));
            return ERROR_MEMORY_ALLOC;
        }
    }

    aialgo_schedule_training_memory(_model, _optimizer, _trainMemory, tsize);
    aialgo_init_model_for_training(_model, _optimizer);

    DEBUG_PRINTLN("ModelManager: Model setup complete.");
    return SUCCESS;
}

// 모델 훈련 함수
float ModelManager::trainModel() {
    DEBUG_PRINTLN("Print Train dataset:");
    DEBUG_PRINT("tensor shape: ");
    for (int d = 0; d < x_tensor.dim; d++) {
        DEBUG_PRINTF("%d ", x_tensor.shape[d]);
    }
    DEBUG_PRINTLN();
    DEBUG_PRINTLN("Print Train Targets:");
    DEBUG_PRINT("tensor shape: ");
    for (int d = 0; d < y_tensor.dim; d++) {
        DEBUG_PRINTF("%d ", y_tensor.shape[d]);
    }
    DEBUG_PRINTLN();
    DEBUG_PRINTLN("\n===== Starting Local Training =====");

    const uint32_t totalSamples = static_cast<uint32_t>(x_tensor.shape[0]);
    if (totalSamples == 0) {
        DEBUG_PRINTLN("WARNING: No samples available for training.");
        return NAN;
    }

    const uint32_t baseBatchSize = static_cast<uint32_t>(BATCH_SIZE);
    const uint32_t totalBatches = (totalSamples + baseBatchSize - 1U) / baseBatchSize;
    const uint32_t remainder = totalSamples % baseBatchSize;

    if (remainder != 0U) {
        DEBUG_PRINTF("WARNING: Dataset size (%lu) not divisible by batch size (%lu); last batch will use %lu samples.\n",
                     static_cast<unsigned long>(totalSamples),
                     static_cast<unsigned long>(baseBatchSize),
                     static_cast<unsigned long>(remainder));
    }

    std::vector<uint16_t> batchXShape(x_tensor.dim);
    std::vector<uint16_t> batchYShape(y_tensor.dim);
    for (int d = 1; d < x_tensor.dim; ++d) {
        batchXShape[d] = x_tensor.shape[d];
    }
    for (int d = 1; d < y_tensor.dim; ++d) {
        batchYShape[d] = y_tensor.shape[d];
    }

    aitensor_t batchX = x_tensor;
    aitensor_t batchY = y_tensor;
    batchX.shape = batchXShape.data();
    batchY.shape = batchYShape.data();

    float* const xBase = reinterpret_cast<float*>(x_tensor.data);
    float* const yBase = reinterpret_cast<float*>(y_tensor.data);
    const size_t inputStride = static_cast<size_t>(COMMON_INPUT_SHAPE);
    const size_t targetStride = static_cast<size_t>(COMMON_TARGET_SHAPE);

    float lastLoss = NAN;
    bool abortTraining = false;

    for (int e = 0; e < TRAIN_EPOCHS && !abortTraining; e++) {
        float epochLossAccum = 0.0f;
        unsigned epochStart = millis();
        DEBUG_PRINTF("Epoch %d/%d\n", e + 1, TRAIN_EPOCHS);

        for (uint32_t b = 0; b < totalBatches; ++b) {
            yield(); // Avoid WDT resets while iterating batches

            const uint32_t currentBatchSize = (b == totalBatches - 1U && remainder != 0U)
                                              ? remainder
                                              : baseBatchSize;

            batchXShape[0] = static_cast<uint16_t>(currentBatchSize);
            batchYShape[0] = static_cast<uint16_t>(currentBatchSize);

            const size_t sampleOffset = static_cast<size_t>(b) * baseBatchSize;
            batchX.data = xBase + sampleOffset * inputStride;
            batchY.data = yBase + sampleOffset * targetStride;

            unsigned batchStart = millis();
            uint8_t trainRc = aialgo_train_model(_model, &batchX, &batchY, _optimizer, currentBatchSize);
            unsigned batchDuration = millis() - batchStart;

            if (trainRc != 0) {
                DEBUG_PRINTF("ERROR: aialgo_train_model failed for batch %lu (rc=%u)\n",
                             static_cast<unsigned long>(b), trainRc);
                abortTraining = true;
                break;
            }

            size_t badIndex = 0;
            float badValue = 0.0f;
            bool hasNonFiniteParam = findNonFiniteParameter(&badIndex, &badValue);

            float batchLossRaw = 0.0f;
            uint8_t lossRc = aialgo_calc_loss_model_f32(_model, &batchX, &batchY, &batchLossRaw);
            if (lossRc != 0) {
                DEBUG_PRINTF("ERROR: aialgo_calc_loss_model_f32 failed for batch %lu (rc=%u)\n",
                             static_cast<unsigned long>(b), lossRc);
                abortTraining = true;
                break;
            }

            bool hasNonFiniteLoss = !std::isfinite(batchLossRaw);
            if (hasNonFiniteLoss) {
                DEBUG_PRINTF("WARNING: Non-finite batch loss detected at epoch %d batch %lu (loss=%f)\n",
                             e + 1, static_cast<unsigned long>(b + 1U), batchLossRaw);
            }

            if (hasNonFiniteParam) {
                DEBUG_PRINTF("WARNING: Non-finite parameter detected at epoch %d batch %lu (index=%lu, value=%f)\n",
                             e + 1,
                             static_cast<unsigned long>(b + 1U),
                             static_cast<unsigned long>(badIndex),
                             badValue);
            }

            epochLossAccum += batchLossRaw;
            const float normalizedBatchLoss = batchLossRaw / (static_cast<float>(currentBatchSize) * static_cast<float>(BATCH_SIZE));

            DEBUG_VERBOSE_PRINTF("  Batch %lu/%lu (size=%lu): loss=%.6f, time=%ums%s%s\n",
                         static_cast<unsigned long>(b + 1U),
                         static_cast<unsigned long>(totalBatches),
                         static_cast<unsigned long>(currentBatchSize),
                         normalizedBatchLoss,
                         batchDuration,
                         hasNonFiniteLoss ? " [WARN non-finite loss]" : "",
                         hasNonFiniteParam ? " [WARN non-finite params]" : "");

            if (hasNonFiniteLoss || hasNonFiniteParam) {
                abortTraining = true;
                break;
            }
        }

        if (!abortTraining) {
            const float normalizedEpochLoss = epochLossAccum / (static_cast<float>(totalSamples) * static_cast<float>(BATCH_SIZE));
            lastLoss = normalizedEpochLoss;
            DEBUG_PRINTF("Epoch %d summary: loss=%.6f, time=%ums\n",
                         e + 1,
                         normalizedEpochLoss,
                         millis() - epochStart);
        }
    }

    DEBUG_PRINTLN("===== Training Complete =====\n");
    return lastLoss;
}

// 가중치 직렬화 및 Base64 인코딩 함수
String ModelManager::getEncodedWeights() {
    DEBUG_PRINTLN("ModelManager: Extracting and encoding weights...");
    if (_model == nullptr || _parameterMemory == nullptr || _parameterSize == 0) {
        DEBUG_PRINTLN(F("ERROR: Model parameter memory is not initialized."));
        return "";
    }

    const size_t floatCount = _parameterSize / sizeof(float);
    const float* paramFloats = reinterpret_cast<const float*>(_parameterMemory);

    DEBUG_PRINTLN("First parameters (up to 5):");
    for (size_t i = 0; i < floatCount && i < 5; ++i) {
        DEBUG_PRINTF("%f ", paramFloats[i]);
    }
    DEBUG_PRINTLN();

    if (floatCount >= 5) {
        DEBUG_PRINTLN("Last parameters (up to 5):");
        for (size_t i = floatCount - 5; i < floatCount; ++i) {
            DEBUG_PRINTF("%f ", paramFloats[i]);
        }
        DEBUG_PRINTLN();
    }

    double sum = 0.0;
    for (size_t i = 0; i < floatCount; ++i) {
        sum += paramFloats[i];
    }
    DEBUG_PRINTF("Sum: %.6f\n", sum);

    String b64 = base64::encode(_parameterMemory, _parameterSize);
    DEBUG_PRINTF("ModelManager: Encoded data memory size: %u bytes (%u chars)\n", _parameterSize, b64.length());

    return b64;
}

void ModelManager::logParameterPreview(const char* label) const {
    if (_parameterMemory == nullptr || _parameterSize == 0) {
        DEBUG_PRINTLN("ModelManager: Parameter preview unavailable (no parameter memory).");
        return;
    }

    const size_t floatCount = _parameterSize / sizeof(float);
    if (floatCount == 0) {
        DEBUG_PRINTLN("ModelManager: Parameter preview unavailable (empty parameter buffer).");
        return;
    }

    const float* params = reinterpret_cast<const float*>(_parameterMemory);
    if (label != nullptr) {
        DEBUG_PRINTF("ModelManager: Parameter preview [%s]\n", label);
    } else {
        DEBUG_PRINTLN("ModelManager: Parameter preview");
    }

    const size_t previewCount = floatCount < 10 ? floatCount : 10;

    DEBUG_PRINT("  First values: ");
    for (size_t i = 0; i < previewCount; ++i) {
        DEBUG_PRINTF("%f ", params[i]);
    }
    DEBUG_PRINTLN("");

    DEBUG_PRINT("  Last values:  ");
    const size_t startIdx = floatCount > 10 ? floatCount - 10 : 0;
    for (size_t i = startIdx; i < floatCount; ++i) {
        DEBUG_PRINTF("%f ", params[i]);
    }
    DEBUG_PRINTLN("");
}

bool ModelManager::findNonFiniteParameter(size_t* outIndex, float* outValue) const {
    if (_parameterMemory == nullptr || _parameterSize == 0) {
        return false;
    }

    const float* params = reinterpret_cast<const float*>(_parameterMemory);
    const size_t count = _parameterSize / sizeof(float);
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(params[i])) {
            if (outIndex != nullptr) {
                *outIndex = i;
            }
            if (outValue != nullptr) {
                *outValue = params[i];
            }
            return true;
        }
    }
    return false;
}

bool ModelManager::decodeBase64(const String& in, uint8_t** out_buf, size_t* out_len) {
    const size_t in_len = in.length();
    // Base64 디코딩 결과의 최대 크기 ≈ in_len * 3 / 4 (+약간의 여유)
    size_t cap = (in_len / 4) * 3 + 4;
    uint8_t* buf = (uint8_t*)ps_malloc(cap);
    if (!buf) return false;

    size_t decoded = 0;
    int rc = mbedtls_base64_decode(
        buf, cap, &decoded,
        (const unsigned char*)in.c_str(), in_len
    );
    if (rc != 0) {  // 실패 시 메모리 해제
        heap_caps_free(buf);
        return false;
    }
    *out_buf = buf;
    *out_len = decoded;
    return true;
}

bool ModelManager::applyGlobalWeights(const String& encodedWeights) {
    DEBUG_PRINTLN("ModelManager: Decoding and applying global weights...");

    // 0) 사전 조건 체크
    if (encodedWeights.length() == 0) {
        DEBUG_PRINTLN(F("ERROR: Empty Base64 payload."));
        return false;
    }
    if (_model == nullptr || _optimizer == nullptr ||
        _outputLayer == nullptr || _denseLayers[0] == nullptr) {
        DEBUG_PRINTLN(F("ERROR: Model/optimizer is not initialized."));
        return false;
    }

    if (_parameterMemory == nullptr || _parameterSize == 0) {
        DEBUG_PRINTLN(F("ERROR: Parameter memory is not initialized."));
        return false;
    }

    // 2) Base64 디코딩
    //  - densaugeo/base64 라이브러리의 decode()는 String을 반환(바이너리 포함 가능)
    uint8_t* decoded_buf = nullptr;
    size_t   decoded_len = 0;

    if (!this->decodeBase64(encodedWeights, &decoded_buf, &decoded_len)) {
        DEBUG_PRINTLN(F("ERROR: Base64 decode failed."));
        return false;
    }
    DEBUG_PRINTF("Decoded bytes: %u, Expected: %u\n", (unsigned)decoded_len, _parameterSize);

    if (decoded_len != _parameterSize) {
        DEBUG_PRINTLN(F("ERROR: Decoded length mismatch. Abort applying weights."));
        heap_caps_free(decoded_buf);
        return false;
    }
    memcpy(_parameterMemory, decoded_buf, _parameterSize);
    heap_caps_free(decoded_buf);

    // 파라미터 메모리는 레이어 텐서의 data 포인터와 공유되지만, 안전을 위해 재분배하여 포인터를 최신 상태로 유지합니다.
    aialgo_distribute_parameter_memory(_model, _parameterMemory, _parameterSize);

    // 5) 옵티마이저(Adam) 상태 리셋
    //  - 글로벌 가중치로 덮은 뒤엔 옵티마이저의 1차/2차 모멘트 등 상태를 초기화해
    //    다음 로컬 스텝이 안정적으로 시작되도록 함.
    aialgo_init_model_for_training(_model, _optimizer);
    DEBUG_PRINTLN("ModelManager: Global weights applied and optimizer reset.");

    return true;
}
