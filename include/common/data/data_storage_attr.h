#pragma once

// Defines attribute macros that place the generated dataset arrays either in PSRAM
// (when available) or in flash/PROGMEM as a fallback.
//
// Usage:
//   DATA_STORAGE_FLOAT_ARRAY(name, length) = { ... };
//
// When PSRAM is present, the data is linked into external RAM so it is immediately
// accessible without performing a runtime memcpy from flash.

#if defined(ARDUINO_ARCH_ESP32)
  #include <esp32-hal-psram.h>
#endif

#ifndef PSRAM_DATA_ATTR
  // If the core does not expose PSRAM_DATA_ATTR we gracefully fall back to a no-op.
  #define PSRAM_DATA_ATTR
#endif

// Helper macros that control how the arrays are declared.
#if defined(BOARD_HAS_PSRAM) || defined(CONFIG_SPIRAM_SUPPORT) || defined(CONFIG_SPIRAM)
  // Place the data directly in PSRAM. The const qualifier keeps the original semantics.
  #define DATA_ATTR_PRE  PSRAM_DATA_ATTR const
  #define DATA_ATTR_POST
#else
  // Keep the arrays in flash using PROGMEM on platforms without PSRAM support.
  #include <pgmspace.h>
  #define DATA_ATTR_PRE  const
  #define DATA_ATTR_POST PROGMEM
#endif

#define DATA_STORAGE_FLOAT_ARRAY(name, length) DATA_ATTR_PRE float name[length] DATA_ATTR_POST
