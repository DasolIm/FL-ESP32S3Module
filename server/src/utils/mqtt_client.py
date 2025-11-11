# mqtt_client.py

import json
import logging
import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, broker_host: str, broker_port: int, client_id: str, username: str = None, password: str = None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.client = mqtt.Client(client_id=self.client_id)

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.logger = logging.getLogger("MQTTClient")
        logging.basicConfig(level=logging.INFO)

    def connect(self):
        self.logger.info("Connecting to MQTT broker at %s:%s", self.broker_host, self.broker_port)
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_start()

    def subscribe(self, topic: str):
        self.client.subscribe(topic)
        self.logger.info("Subscribed to topic: %s", topic)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Connected to MQTT broker successfully")
        else:
            self.logger.error("Failed to connect to MQTT broker, return code: %s", rc)

    def on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        self.logger.info("Received message on topic %s: %s", msg.topic, payload)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.logger.info("Disconnected from MQTT broker")
        
        
class ServerConfig:
    """MQTT 서버 연결 및 연합학습 설정"""
    
    def __init__(self, broker_host='broker.emqx.io', broker_port=1883, rounds=100, min_clients=1, 
                 topic_sot="yonseiiot/ysk/fl/command/sot",
                 topic_global_weight="yonseiiot/ysk/fl/global/weight",
                 topic_status_wildcard="yonseiiot/ysk/fl/+/status",
                 topic_weight_wildcard="yonseiiot/ysk/fl/+/weight",
                 client_id=None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.rounds = rounds
        self.min_clients = min_clients
        self.topic_sot = topic_sot
        self.topic_global_weight = topic_global_weight
        self.topic_status_wildcard = topic_status_wildcard
        self.topic_weight_wildcard = topic_weight_wildcard
        self.client_id = client_id