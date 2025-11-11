import paho.mqtt.client as mqtt
import numpy as np
import base64
import json
import time
from collections import defaultdict
from threading import Event
import matplotlib.pyplot as plt

class FederatedAveragingServer:
    """연합학습 서버 (FedAvg 알고리즘 구현)"""
    
    def __init__(self, config):
        self.config = config
        self.client = mqtt.Client(
            client_id=self.config.client_id or f"fl-server-{int(time.time())}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        # 클라이언트 관리
        self.ready_clients = set()
        self.active_clients = set()
        self.client_weights = {}
        self.client_losses = {}
        
        # 학습 상태
        self.current_round = 0
        self.round_start_time = 0
        self.training_in_progress = False
        self.training_complete = Event()
        
        # 결과 추적
        self.round_times = []
        self.avg_losses = []
        self.client_losses_history = defaultdict(list)

        # 가중치 검증용 메타데이터
        self._weight_vector_bytes = None
        self._weight_vector_count = None

    def connect(self):
        """MQTT 브로커에 연결"""
        try:
            self.client.connect(self.config.broker_host, self.config.broker_port, 60)
            self.client.loop_start()
            print(f"MQTT 브로커에 연결됨: {self.config.broker_host}:{self.config.broker_port}")
            return True
        except Exception as e:
            print(f"MQTT 브로커 연결 실패: {e}")
            return False
    
    def close(self):
        """MQTT 브로커 연결 종료"""
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT 브로커 연결 종료됨")
    
    def start_training(self):
        """연합학습 시작"""
        if not self.ready_clients:
            print("준비된 클라이언트가 없습니다. 클라이언트 연결 대기 중...")
            return False
        
        self.training_in_progress = True
        self.current_round = 0
        self._reset_weight_state()
        self._start_next_round()
        
        # 학습 완료까지 대기
        self.training_complete.wait()
        
        # 결과 시각화
        self._visualize_results()
        return True
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT 브로커 연결 콜백"""
        if rc == 0:
            print("MQTT 브로커에 연결됨")
            client.subscribe(self.config.topic_status_wildcard)
            client.subscribe(self.config.topic_weight_wildcard)
            print(f"구독 중: {self.config.topic_status_wildcard}")
            print(f"구독 중: {self.config.topic_weight_wildcard}")
        else:
            print(f"연결 실패, 반환 코드: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT 메시지 수신 콜백"""
        try:
            # 상태 메시지 처리
            if msg.topic.endswith("/status"):
                client_id = msg.topic.split('/')[-2]
                self._process_status_message(client_id, msg.payload)
            
            # 가중치 메시지 처리
            elif msg.topic.endswith("/weight") and self.training_in_progress:
                client_id = msg.topic.split('/')[-2]
                self._process_weight_message(client_id, msg.payload)
                
        except Exception as e:
            print(f"메시지 처리 중 오류: {e}")
    
    def _process_status_message(self, client_id, payload):
        """클라이언트 상태 메시지 처리"""
        try:
            status_data = json.loads(payload)
            status = status_data.get("Status")
            
            if status == "READY":
                dataset_id = status_data.get("DataSet ID")
                self.ready_clients.add(client_id)
                print(f"클라이언트 {client_id}가 데이터셋 {dataset_id}로 준비됨")
                print(f"준비된 클라이언트 수: {len(self.ready_clients)}")
            
            elif status == "TRAINING_RESULT" and self.training_in_progress:
                if "Loss" in status_data:
                    loss = float(status_data.get("Loss", 0))
                    self.client_losses[client_id] = loss
                    print(f"클라이언트 {client_id} 손실: {loss:.6f}")
        
        except json.JSONDecodeError:
            print(f"클라이언트 {client_id}로부터 잘못된 JSON 형식")
    
    def _process_weight_message(self, client_id, payload):
        """학습 중 클라이언트 가중치 메시지 처리"""
        if client_id not in self.active_clients:
            print(f"비활성 클라이언트 {client_id}의 가중치 무시")
            return
            
        try:
            # 가중치 디코딩 및 저장
            weights = self._base64_to_numpy(payload.decode('utf-8'))
            self.client_weights[client_id] = weights
            print(f"클라이언트 {client_id}에서 가중치 수신 ({len(self.client_weights)}/{len(self.active_clients)})")
            
            # 모든 활성 클라이언트의 가중치를 받았는지 확인
            if len(self.client_weights) == len(self.active_clients):
                self._aggregate_and_send_weights()
        
        except Exception as e:
            print(f"클라이언트 {client_id}의 가중치 처리 중 오류: {e}")
    
    def _start_next_round(self):
        """다음 학습 라운드 시작"""
        self.current_round += 1
        
        # 모든 라운드가 완료되었는지 확인
        if self.current_round > self.config.rounds:
            self.training_in_progress = False
            self.training_complete.set()
            print("연합학습 완료!")
            return
        
        # 새 라운드를 위한 초기화
        self._reset_weight_state()
        self.active_clients = set(self.ready_clients)
        self.round_start_time = time.time()
        
        # 충분한 클라이언트가 있는지 확인
        if len(self.active_clients) < self.config.min_clients:
            print(f"클라이언트 부족 ({len(self.active_clients)}/{self.config.min_clients}). 대기 중...")
            return
        
        # 학습 시작 명령 전송
        print(f"\n----- 라운드 {self.current_round}/{self.config.rounds} -----")
        print(f"활성 클라이언트: {len(self.active_clients)}개")
        self.client.publish(self.config.topic_sot, "true")
    
    def _aggregate_and_send_weights(self):
        """가중치 집계 및 글로벌 모델 전송"""
        if not self.client_weights:
            print("집계할 가중치 없음")
            self._start_next_round()
            return
        
        # FedAvg 알고리즘 적용
        weights_list = list(self.client_weights.values())
        global_weights = np.mean(weights_list, axis=0)
        global_weights = global_weights.astype(np.float32)
        
        # 학습 메트릭 기록
        if self.client_losses:
            avg_loss = np.mean(list(self.client_losses.values()))
            self.avg_losses.append(avg_loss)
            
            for client_id, loss in self.client_losses.items():
                self.client_losses_history[client_id].append(loss)
            
            print(f"라운드 {self.current_round} 평균 손실: {avg_loss:.6f}")
        
        # 라운드 시간 기록
        round_time = time.time() - self.round_start_time
        self.round_times.append(round_time)
        print(f"라운드 {self.current_round} 완료 시간: {round_time:.2f}초")
        
        # 글로벌 가중치를 클라이언트에 전송
        encoded_weights = self._numpy_to_base64(global_weights)
        self.client.publish(self.config.topic_global_weight, encoded_weights)
        print(f"라운드 {self.current_round} 글로벌 가중치 발행 완료")
        
        # 다음 라운드 시작
        self._start_next_round()

    def _reset_weight_state(self):
        """누적된 가중치/손실 및 메타데이터 초기화"""
        self.client_weights.clear()
        self.client_losses.clear()
        self._weight_vector_bytes = None
        self._weight_vector_count = None
        print("서버: 가중치/손실 버퍼 초기화 완료")
    
    def _numpy_to_base64(self, arr):
        """NumPy 배열을 base64 인코딩 문자열로 변환"""
        arr = np.asarray(arr, dtype=np.float32)
        if self._weight_vector_count is not None and arr.size != self._weight_vector_count:
            raise ValueError(
                f"글로벌 가중치 크기 불일치: {arr.size} (expected {self._weight_vector_count})"
            )

        raw_bytes = arr.tobytes(order='C')
        if self._weight_vector_bytes is None:
            self._weight_vector_bytes = len(raw_bytes)
            self._weight_vector_count = arr.size
        elif len(raw_bytes) != self._weight_vector_bytes:
            raise ValueError(
                f"가중치 직렬화 크기 불일치: {len(raw_bytes)} (expected {self._weight_vector_bytes})"
            )

        return base64.b64encode(raw_bytes).decode('utf-8')
    
    def _base64_to_numpy(self, b64_str):
        """base64 인코딩 문자열을 NumPy 배열로 변환"""
        decoded = base64.b64decode(b64_str)

        if len(decoded) % 4 != 0:
            raise ValueError(
                f"가중치 바이트 길이({len(decoded)})가 4의 배수가 아님"
            )

        if self._weight_vector_bytes is None:
            self._weight_vector_bytes = len(decoded)
            self._weight_vector_count = len(decoded) // 4
        elif len(decoded) != self._weight_vector_bytes:
            raise ValueError(
                f"가중치 바이트 길이 불일치: {len(decoded)} (expected {self._weight_vector_bytes})"
            )

        return np.frombuffer(decoded, dtype='<f4').copy()
    
    def _visualize_results(self):
        """학습 결과 시각화"""
        if not self.round_times or not self.avg_losses:
            print("시각화할 데이터 없음")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 라운드 시간 그래프
        rounds = range(1, len(self.round_times) + 1)
        ax1.plot(rounds, self.round_times, 'b-', label='라운드 시간')
        ax1.set_title('라운드별 소요 시간')
        ax1.set_xlabel('라운드')
        ax1.set_ylabel('시간 (초)')
        ax1.grid(True)
        ax1.legend()
        
        # 손실 그래프
        loss_rounds = range(1, len(self.avg_losses) + 1)
        ax2.plot(loss_rounds, self.avg_losses, 'r-', label='평균 손실')
        
        # 클라이언트별 손실 그래프
        for client_id, losses in self.client_losses_history.items():
            client_rounds = range(1, len(losses) + 1)
            ax2.plot(client_rounds, losses, alpha=0.3, label=f'클라이언트 {client_id}')
        
        # Min/Max 손실 범위
        if len(self.client_losses_history) > 1:
            all_losses = []
            for losses in self.client_losses_history.values():
                if losses:  # 빈 리스트 건너뛰기
                    all_losses.append(losses)
            
            if all_losses:
                all_losses_array = np.array([losses for losses in all_losses if len(losses) == len(all_losses[0])])
                if len(all_losses_array) > 0:
                    min_losses = np.min(all_losses_array, axis=0)
                    max_losses = np.max(all_losses_array, axis=0)
                    min_max_rounds = range(1, len(min_losses) + 1)
                    ax2.fill_between(min_max_rounds, min_losses, max_losses, color='r', alpha=0.2, label='Min/Max 범위')
        
        ax2.set_title('학습 손실')
        ax2.set_xlabel('라운드')
        ax2.set_ylabel('손실')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('federated_learning_results.png')
        print("결과가 'federated_learning_results.png'에 저장됨")
        plt.show()