import app.config as cfg

from app.request_status import RequestStatus


class Request:
    ARRIVAL_TOLERANCE_TIME = 20

    def __init__(self, request_id, from_node_id, to_node_id, request_time, network):
        self.num_passengers = 1

        # 불변
        self.request_id = request_id
        self.network = network
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.request_time = request_time
        self.travel_time = -10000000
        self.deadline = -1

        # 가변 (매 시간 업데이트 필요)
        self.status = RequestStatus.PENDING
        self.waiting_time = -1
        self.time_to_deadline = -1
        self.assigned_v_id = -1

    def __str__(self):
        return (f"Request(request_id={self.request_id}, "
                f"from={self.from_node_id}, to={self.to_node_id}, "
                f"request_time={self.request_time}, travel_time={self.travel_time})")

    def __repr__(self):
        return (f"Request({self.request_id}, {self.from_node_id}, "
                f"{self.to_node_id}, {self.request_time}, {self.travel_time})")

    def set_travel_time(self, travel_time):
        self.travel_time = travel_time
        self.deadline = self.request_time + self.travel_time + Request.ARRIVAL_TOLERANCE_TIME

    def get_state(self):
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
        vec_deadline = [self.time_to_deadline / (self.network.max_duration + Request.ARRIVAL_TOLERANCE_TIME)]
        vec_all = vec_status + vec_from + vec_to + vec_passengers + vec_travel + vec_wait + vec_deadline
        return vec_all
