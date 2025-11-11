#include <aifes.h>
#include <cstring>
#ifdef ARDUINO_ARCH_ESP32
#include <esp32-hal-psram.h>
#endif
#include "common/config.h"
#include "common/data/data_api.h"

#if   SELECTED_DATASET_ID == 0
  #include "common/data/data_0.h"
#elif SELECTED_DATASET_ID == 1
  #include "common/data/data_1.h"
#elif SELECTED_DATASET_ID == 2
  #include "common/data/data_2.h"
#elif SELECTED_DATASET_ID == 3
  #include "common/data/data_3.h"
#elif SELECTED_DATASET_ID == 4
  #include "common/data/data_4.h"
#else
  #error "Invalid SELECTED_DATASET_ID"
#endif

extern const size_t N_DATAS               = N_DATAS_VALUE;
extern const uint16_t COMMON_INPUT_SHAPE  = INPUT_SHAPE;
extern const uint16_t COMMON_TARGET_SHAPE = TARGET_SHAPE;

uint16_t x_shape[2] = {
  static_cast<uint16_t>(N_DATAS),
  COMMON_INPUT_SHAPE
};
uint16_t y_shape[2] = {
  static_cast<uint16_t>(N_DATAS),
  COMMON_TARGET_SHAPE
};

static float *x_psram_buffer = nullptr;
static float *y_psram_buffer = nullptr;

aitensor_t x_tensor = AITENSOR_2D_F32(x_shape, const_cast<float*>(x_data));
aitensor_t y_tensor = AITENSOR_2D_F32(y_shape, const_cast<float*>(y_labels));

bool init_dataset_buffers() {
#ifdef ARDUINO_ARCH_ESP32
  const bool has_psram = psramFound();
#else
  const bool has_psram = false;
#endif

  if (has_psram) {
    const size_t x_bytes = static_cast<size_t>(N_DATAS) * COMMON_INPUT_SHAPE * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(N_DATAS) * COMMON_TARGET_SHAPE * sizeof(float);

    if (x_psram_buffer == nullptr) {
      x_psram_buffer = static_cast<float*>(ps_malloc(x_bytes));
      if (x_psram_buffer != nullptr) {
        memcpy(x_psram_buffer, x_data, x_bytes);
      }
    }

    if (y_psram_buffer == nullptr) {
      y_psram_buffer = static_cast<float*>(ps_malloc(y_bytes));
      if (y_psram_buffer != nullptr) {
        memcpy(y_psram_buffer, y_labels, y_bytes);
      }
    }

    if (x_psram_buffer != nullptr && y_psram_buffer != nullptr) {
      x_tensor.data = x_psram_buffer;
      y_tensor.data = y_psram_buffer;
      return true;
    }
  }

  // PSRAM not available or allocation failed → fall back to compile-time buffers.
  x_tensor.data = const_cast<float*>(x_data);
  y_tensor.data = const_cast<float*>(y_labels);

  return false;
}