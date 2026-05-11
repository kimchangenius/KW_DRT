import app.config as cfg

from app.request_status import RequestStatus, REQUEST_STATUS_NUM_CLASSES


class Request:
    PICKUP_TOLERANCE_TIME = 10
    ARRIVAL_TOLERANCE_TIME = 20

    def __init__(self, request_id, from_node_id, to_node_id, request_time, network,
                 num_passengers=1):
        self.num_passengers = int(num_passengers)

        # 불변
        self.id = request_id
        self.network = network
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.request_time = request_time
        self.travel_time = -10000000
        self.pickup_due = -1
        self.arrival_due = -1

        # 가변 (매 시간 업데이트 필요)
        self.status = RequestStatus.PENDING
        self.waiting_time = -1
        self.in_vehicle_time = 0
        self.arrival_due_left = -1
        self.assigned_v_id = -1
        self.slot_idx = -1

        # 기록용
        self.detour_time = -1
        self.pickup_at = None
        self.dropoff_at = None

    def __str__(self):
        return (f"<R>(id={self.id} / "
                f"{self.from_node_id} -> {self.to_node_id} / "
                f"status={self.status} / "
                f"veh={self.assigned_v_id} / "
                f"rt={self.request_time} / "
                f"wt={self.waiting_time} / "
                f"ivt={self.in_vehicle_time} / "
                f"odt={self.travel_time} / "
                f"pt={self.pickup_at} / "
                f"dt={self.dropoff_at} / "
                f"p_due={self.pickup_due} / "
                f"a_due={self.arrival_due})"
                )

    def set_travel_time(self, travel_time):
        self.travel_time = travel_time
        self.pickup_due = self.request_time + Request.PICKUP_TOLERANCE_TIME
        self.arrival_due = self.request_time + self.travel_time + Request.ARRIVAL_TOLERANCE_TIME

    def get_static_features(self):
        """노드 정보를 뺀 요청의 작은 raw feature 벡터.
        status(3) + passengers(1) + travel(1) + waiting(1) + arrival_due_left(1) = REQUEST_RAW_DIM=7"""
        vec_status = [0] * REQUEST_STATUS_NUM_CLASSES
        if 1 <= self.status <= REQUEST_STATUS_NUM_CLASSES:
            vec_status[self.status - 1] = 1
        denom = (self.network.max_duration + Request.ARRIVAL_TOLERANCE_TIME)
        vec_pass = [self.num_passengers / cfg.VEH_CAPACITY]
        vec_travel = [self.travel_time / max(self.network.max_duration, 1)]
        vec_wait = [max(0.0, self.waiting_time) / cfg.MAX_WAIT_TIME]
        vec_due = [max(0, self.arrival_due_left) / max(denom, 1)]
        return vec_status + vec_pass + vec_travel + vec_wait + vec_due

    def get_node_ids(self):
        """(from_node_id, to_node_id). 0은 'no node' 센티넬."""
        f = self.from_node_id if 1 <= self.from_node_id <= cfg.NUM_NODES else 0
        t = self.to_node_id if 1 <= self.to_node_id <= cfg.NUM_NODES else 0
        return [f, t]
