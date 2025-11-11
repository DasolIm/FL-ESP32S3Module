from src.server.fedavg import FederatedAveragingServer
from src.utils.mqtt_client import ServerConfig
import logging
import time
import paho.mqtt.client as mqtt
import numpy as np
import json
import base64
import io
from collections import defaultdict
from threading import Event

# 글로벌 변수
clients_training = set()
clients_ready = set()
client_weights = {}
training_rounds = 0
total_rounds = 100
mqtt_client = None
training_complete = Event()

# 토픽 설정
TOPIC_SOT = "yonseiiot/ysk/fl/command/sot"
TOPIC_GLOBAL = "yonseiiot/ysk/fl/global/weight"
TOPIC_STATUS_WILDCARD = "yonseiiot/ysk/fl/+/status"
TOPIC_WEIGHT_WILDCARD = "yonseiiot/ysk/fl/+/weight"

# 콜백 함수
def on_connect(client, userdata, flags, rc):
    logging.info("MQTT 브로커에 연결됨")
    client.subscribe(TOPIC_STATUS_WILDCARD)
    client.subscribe(TOPIC_WEIGHT_WILDCARD)
    print(f"subscribed: {TOPIC_STATUS_WILDCARD}")
    print(f"subscribed: {TOPIC_WEIGHT_WILDCARD}")

def on_message(client, userdata, msg):
    global clients_training, clients_ready, client_weights, training_rounds
    
    try:
        # 상태 메시지 처리
        if msg.topic.endswith("/status"):
            client_id = msg.topic.split('/')[-2]
            process_status_message(client_id, msg.payload)
        
        # 가중치 메시지 처리
        elif msg.topic.endswith("/weight"):
            client_id = msg.topic.split('/')[-2]
            process_weight_message(client_id, msg.payload)
    
    except Exception as e:
        logging.error(f"Exception err: {e}")

def process_status_message(client_id, payload):
    global clients_training, clients_ready
    
    try:
        status_data = json.loads(payload)
        status = status_data.get("Status")
        
        if status == "READY":
            dataset_id = status_data.get("DataSet ID")
            clients_ready.add(client_id)
            logging.info(f"클라이언트 {client_id}가 데이터셋 {dataset_id}로 준비됨")
            logging.info(f"num of clients: {len(clients_ready)}")
        
        elif status == "TRAINING":
            # 클라이언트가 훈련 중임을 표시
            clients_training.add(client_id)
            logging.info(f"클라이언트 {client_id}가 훈련 중. 훈련 중인 클라이언트: {len(clients_training)}개")
    
    except json.JSONDecodeError:
        logging.error(f"잘못된 JSON 형식: {payload}")

def process_weight_message(client_id, payload):
    global client_weights, clients_training
    
    try:
        # 가중치 디코딩 및 저장
        weights = base64_to_numpy(payload.decode('utf-8'))
        client_weights[client_id] = weights
        logging.info(f"클라이언트 {client_id}에서 가중치 수신 ({len(client_weights)}/{len(clients_training)})")
        
        # 모든 훈련 중인 클라이언트의 가중치를 받았는지 확인
        if len(client_weights) == len(clients_training) and len(client_weights) > 0:
            aggregate_and_send_weights()
    
    except Exception as e:
        logging.error(f"가중치 처리 중 오류: {e}")

def aggregate_and_send_weights():
    global client_weights, training_rounds
    
    if not client_weights:
        logging.warning("집계할 가중치가 없음")
        return
    
    # FedAvg 알고리즘 적용
    weights_list = list(client_weights.values())
    global_weights = np.mean(weights_list, axis=0)
    
    # 라운드 증가
    training_rounds += 1
    logging.info(f"라운드 {training_rounds}/{total_rounds} 완료")
    
    # 글로벌 가중치 전송
    encoded_weights = numpy_to_base64(global_weights)
    mqtt_client.publish(TOPIC_GLOBAL, encoded_weights)
    logging.info(f"글로벌 가중치 발행 완료")
    
    # 모든 라운드가 완료되었는지 확인
    if training_rounds >= total_rounds:
        logging.info("연합학습 완료!")
        training_complete.set()
        return
    
    # 다음 라운드를 위한 초기화
    client_weights.clear()
    clients_training.clear()
    
    # 새 라운드 시작
    logging.info(f"\n----- 라운드 {training_rounds + 1}/{total_rounds} 시작 -----")
    mqtt_client.publish(TOPIC_SOT, "true")

def numpy_to_base64(arr):
    """NumPy 배열을 base64 인코딩 문자열로 변환"""
    with io.BytesIO() as buf:
        np.save(buf, arr, allow_pickle=False)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

def base64_to_numpy(b64_str):
    """base64 인코딩 문자열을 NumPy 배열로 변환"""
    decoded = base64.b64decode(b64_str)
    with io.BytesIO(decoded) as buf:
        return np.load(buf, allow_pickle=False)

def main():
    global mqtt_client
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("TinyFL")
    
    # MQTT 클라이언트 설정
    mqtt_client = mqtt.Client(
        client_id=f"fl-server-{int(time.time())}",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION1
    )
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    try:
        # MQTT 브로커에 연결
        mqtt_client.connect("broker.emqx.io", 1883, 60)
        mqtt_client.loop_start()
        
        logger.info("클라이언트 연결을 대기 중입니다...")
        
        # 사용자 입력 대기
        start_training = input('100 라운드 학습을 시작하시겠습니까? (y/n): ')
        if start_training.lower() == 'y':
            logger.info("연합학습을 시작합니다...")
            
            # SOT 명령 전송 - 클라이언트 수와 관계없이 바로 시작
            mqtt_client.publish(TOPIC_SOT, "true")
            logger.info("SOT 신호를 전송했습니다. 클라이언트 응답을 기다립니다...")
            
            # 학습이 완료될 때까지 대기
            training_complete.wait()
            
        else:
            logger.info("학습이 취소되었습니다.")
    
    except KeyboardInterrupt:
        logger.info("프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    finally:
        # 연결 종료
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        logger.info("MQTT 브로커 연결이 종료되었습니다.")

if __name__ == '__main__':
    main()