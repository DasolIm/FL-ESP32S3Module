#include "core/fl_client.h"

FLClient* flClient;

void setup() {
    flClient = new FLClient();
    if (flClient->begin() != SUCCESS) {
        Serial.println("FATAL ERROR: FLClient initialization failed! Halting.");
        while (true);
    }
}

void loop() {
    flClient->loop();
}