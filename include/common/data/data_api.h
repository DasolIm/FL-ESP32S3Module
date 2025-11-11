//
// Created by 강예성 on 25. 8. 10.
//

#ifndef DATA_API_H
#define DATA_API_H

#pragma once
#include <aifes.h>
#include <stddef.h>
#include <stdint.h>

extern const size_t N_DATAS;
extern const uint16_t COMMON_INPUT_SHAPE;
extern const uint16_t COMMON_TARGET_SHAPE;

extern uint16_t x_shape[2];
extern uint16_t y_shape[2];

extern aitensor_t x_tensor;
extern aitensor_t y_tensor;

// Ensure dataset tensors point to RAM buffers (PSRAM if available).
bool init_dataset_buffers();

#endif //DATA_API_H
