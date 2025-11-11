#include <WiFiNINA.h>
#include <PubSubClient.h>
#include <Wire.h>
#include "am1008w_k_i2c.h"
#include <MQUnifiedsensor.h>
#include <time.h>

const char* WIFI_SSID  = "Yonsei-IoT-2G";
const char* WIFI_PASS  = "yonseiiot209";
const char* MQTT_HOST  = "192.168.0.12";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_TOPIC = "generative-ai/testbed";

#define PIN_MP801     A3
#define PIN_MICS2714  A0
#define PIN_MQ7       A1
#define PIN_MQ131     A2
#define I2C_SDA       A4
#define I2C_SCL       A5
#define PERIOD_MS     60000

const bool DEBUG = true;

WiFiClient net;
PubSubClient mqtt(net);
AM1008W_K_I2C AM;

#define BOARD_NAME            "Nano RP2040 Connect"
#define VOLTAGE_RESOLUTION    3.3
#define ADC_BIT_RESOLUTION    12
#define MQ131_TYPE            "MQ-131"
#define MQ131_RL_KOHM         10
#define MQ131_CLEAN_AIR_RATIO 15
MQUnifiedsensor MQ131(BOARD_NAME, VOLTAGE_RESOLUTION, ADC_BIT_RESOLUTION, PIN_MQ131, MQ131_TYPE);

String MAC_ID;

String get_mac_id() {
  byte mac[6]; WiFi.macAddress(mac);
  char buf[13]; for (int i=0;i<6;i++) sprintf(&buf[i*2], "%02X", mac[i]);
  return String(buf);
}

void connect_wifi() {
  if (DEBUG) Serial.println("[WiFi] connecting...");
  while (WiFi.begin(WIFI_SSID, WIFI_PASS) != WL_CONNECTED) { delay(300); }
  if (DEBUG) {
    Serial.println("[WiFi] OK");
    Serial.print("[WiFi] IP="); Serial.println(WiFi.localIP());
  }
}

void connect_mqtt() {
  while (!mqtt.connected()) {
    String cid = "RP2040_" + MAC_ID;
    if (DEBUG) { Serial.print("[MQTT] connect... "); }
    if (mqtt.connect(cid.c_str())) {
      if (DEBUG) Serial.println("OK");
      break;
    }
    if (DEBUG) Serial.println("retry");
    delay(1000);
  }
}

void init_time() {
  if (DEBUG) Serial.println("[TIME] sync...");
  setenv("TZ", "KST-9", 1);
  tzset();
  for (int i = 0; i < 50; ++i) {
    if (WiFi.getTime() >= 1700000000UL) {
      if (DEBUG) Serial.println("[TIME] OK");
      return;
    }
    delay(100);
  }
  if (DEBUG) Serial.println("[TIME] not synced");
}

String now_iso8601() {
  unsigned long epoch = WiFi.getTime();
  if (epoch < 1700000000UL) return "1970-01-01T00:00:00+09:00";
  time_t t = (time_t)epoch;
  struct tm kst;
  gmtime_r(&t, &kst);
  kst.tm_hour += 9;
  mktime(&kst);
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &kst);
  String s(buf);
  s += "+09:00";
  return s;
}

void setup_mq131() {
  MQ131.setRegressionMethod(1);
  MQ131.setA(23.943); MQ131.setB(-1.11);
  MQ131.init();
  MQ131.setRL(MQ131_RL_KOHM);

  if (DEBUG) Serial.print("[MQ131] calibrating");
  float r0_sum = 0;
  for (int i=0; i<10; i++) {
    MQ131.update();
    r0_sum += MQ131.calibrate(MQ131_CLEAN_AIR_RATIO);
    if (DEBUG) Serial.print(".");
    delay(200);
  }
  float R0 = r0_sum / 10.0f;
  MQ131.setR0(R0);
  if (DEBUG) {
    Serial.println(" done");
    Serial.print("[MQ131] R0="); Serial.println(R0);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  connect_wifi();
  MAC_ID = get_mac_id();
  if (DEBUG) { Serial.print("[BOOT] MAC="); Serial.println(MAC_ID); }

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setBufferSize(512);

  AM.begin();
  setup_mq131();
  init_time();

  if (DEBUG) Serial.println("[BOOT] ready");
}

void loop() {
  if (!mqtt.connected()) connect_mqtt();
  mqtt.loop();

  static uint32_t last = 0;
  uint32_t nowms = millis();
  if (nowms - last < PERIOD_MS) return;
  last = nowms;

  int mp801_raw = analogRead(PIN_MP801);
  int mics_raw  = analogRead(PIN_MICS2714);
  int mq7_raw   = analogRead(PIN_MQ7);

  MQ131.update();
  float o3_ppm_f = MQ131.readSensor();        
  int   o3_ppb   = (int)(o3_ppm_f * 1000.0f);

  int co2 = 0, pm10 = 0, pm25 = 0;
  float temp = 0, hum = 0;
  bool am_ok = (AM.read_data_command() == 0);
  if (am_ok) {
    co2  = AM.get_co2();
    temp = AM.get_temperature();
    hum  = AM.get_humidity();
    pm10 = AM.get_pm10();
    pm25 = AM.get_pm2p5();
  }

  String ts = now_iso8601();

  String payload = "{";
  payload += "\"mac\":\"" + MAC_ID + "\",";
  payload += "\"ts\":\"" + ts + "\",";
  payload += "\"Temperature\":" + String(temp, 2) + ",";
  payload += "\"Humidity\":"    + String(hum, 2)  + ",";
  payload += "\"CO2\":"         + String(co2)     + ",";
  payload += "\"PM10\":"        + String(pm10)    + ",";
  payload += "\"PM2.5\":"       + String(pm25)    + ",";
  payload += "\"MP801_raw\":"   + String(mp801_raw) + ",";
  payload += "\"MiCS2714_raw\":"+ String(mics_raw)  + ",";
  payload += "\"MQ7_raw\":"     + String(mq7_raw)   + ",";
  payload += "\"MQ131_O3_ppb\":"+ String(o3_ppb);
  payload += "}";

  bool ok = mqtt.publish(MQTT_TOPIC, payload.c_str());
  if (DEBUG) {
    Serial.print("[MQTT] publish "); Serial.println(ok ? "OK" : "FAIL");
    Serial.print("[PAYLOAD] "); Serial.println(payload);
  }
}
