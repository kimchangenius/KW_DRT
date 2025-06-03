import app.config as cfg

from app.request_status import RequestStatus


class Request:
    PICKUP_TOLERANCE_TIME = 10
    ARRIVAL_TOLERANCE_TIME = 20

    def __init__(self, request_id, from_node_id, to_node_id, request_time, network):
        self.num_passengers = 1

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
        self.in_vehicle_time = -1
        self.arrival_due_left = -1
        self.assigned_v_id = -1
        self.slot_idx = -1

        # 기록용
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
                f"p_due={self.pickup_due} / "
                f"a_due={self.arrival_due})"
                )

    def set_travel_time(self, travel_time):
        self.travel_time = travel_time
        self.pickup_due = self.request_time + Request.PICKUP_TOLERANCE_TIME
        self.arrival_due = self.request_time + self.travel_time + Request.ARRIVAL_TOLERANCE_TIME

    def get_vector(self):
        num_nodes = self.network.num_nodes

        vec_status = [0] * RequestStatus.NUM_CLASSES
        if 1 <= self.status <= RequestStatus.NUM_CLASSES:
            vec_status[self.status - 1] = 1

        vec_from = [0] * num_nodes
        if 1 <= self.from_node_id <= num_nodes:
            vec_from[self.from_node_id - 1] = 1

        vec_to = [0] * num_nodes
        if 1 <= self.to_node_id <= num_nodes:
            vec_to[self.to_node_id - 1] = 1

        vec_passengers = [self.num_passengers / cfg.VEH_CAPACITY]
        vec_travel = [self.travel_time / self.network.max_duration]
        vec_wait = [self.waiting_time / cfg.MAX_WAIT_TIME]
        vec_deadline = [self.arrival_due_left / (self.network.max_duration + Request.ARRIVAL_TOLERANCE_TIME)]
        vec_all = vec_status + vec_from + vec_to + vec_passengers + vec_travel + vec_wait + vec_deadline
        return vec_all
