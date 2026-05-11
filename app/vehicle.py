import app.config as cfg

from app.request_status import RequestStatus
from app.vehicle_status import VehicleStatus, VEHICLE_STATUS_NUM_CLASSES


class Vehicle:
    def __init__(self, veh_id, curr_node, network):
        self.id = veh_id        # 0부터 N-1까지 값을 가짐
        self.network = network

        self.status = VehicleStatus.IDLE
        self.curr_node = curr_node
        self.next_node = 0
        self.active_request_list = []
        self.target_request = None
        self.target_arrival_time = -1
        self.num_passengers = 0

        # Logging
        self.num_accept = 0
        self.num_serve = 0
        self.idle_time = 0
        self.on_service_driving_time = 0

    def __str__(self):
        return (f"[V](id={self.id} / "
                f"{self.curr_node} -> {self.next_node} / "
                f"target_r={self.target_request.id if self.target_request else 'None'} / "
                f"status={self.status} / "
                f"at={self.target_arrival_time} / "
                f"np={self.num_passengers} / "
                f"active_r_num={len(self.active_request_list)})"
                )

    def has_pickedup_request(self):
        for r in self.active_request_list:
            if r.status == RequestStatus.PICKEDUP:
                return True
        return False

    def get_static_features(self):
        """노드 정보를 뺀 차량의 작은 raw feature 벡터.
        status one-hot(4) + capacity(1) = VEHICLE_RAW_DIM=5"""
        vec_status = [0] * VEHICLE_STATUS_NUM_CLASSES
        if 1 <= self.status <= VEHICLE_STATUS_NUM_CLASSES:
            vec_status[self.status - 1] = 1
        vec_capa = [(cfg.VEH_CAPACITY - self.num_passengers) / cfg.VEH_CAPACITY]
        return vec_status + vec_capa

    def get_node_ids(self):
        """노드 ID 튜플 (curr_node, next_node). 노드는 1..NUM_NODES, 0은 'no node'."""
        curr = self.curr_node if 1 <= self.curr_node <= cfg.NUM_NODES else 0
        nxt = self.next_node if 1 <= self.next_node <= cfg.NUM_NODES else 0
        return [curr, nxt]

