import app.config as cfg

from app.vehicle_status import VehicleStatus


class Vehicle:
    def __init__(self, veh_id, curr_node, network):
        self.id = veh_id        # 0부터 N-1까지 값을 가짐
        self.network = network

        self.status = VehicleStatus.IDLE
        self.curr_node = curr_node
        self.next_node = 0

        self.active_requests = []
        self.target_request = None
        self.target_arrival_time = -1

    def __str__(self):
        return (f"[V](id={self.id} / "
                f"{self.curr_node} -> {self.next_node} / "
                f"target_r={self.target_request.request_id if self.target_request else 'None'} / "
                f"status={self.status} / "
                f"arr_time={self.target_arrival_time} / "
                f"active_r_num={len(self.active_requests)})"
                )

    def get_available_seats(self):
        num_curr_passengers = 0
        for r in self.active_requests:
            num_curr_passengers += r.num_passengers
        return cfg.VEH_CAPACITY - num_curr_passengers

    def get_vector(self):
        num_nodes = self.network.num_nodes

        vec_status = [0] * VehicleStatus.NUM_CLASSES
        if 1 <= self.status <= VehicleStatus.NUM_CLASSES:
            vec_status[self.status - 1] = 1

        vec_from = [0] * num_nodes
        if 1 <= self.curr_node <= num_nodes:
            vec_from[self.curr_node - 1] = 1

        vec_to = [0] * num_nodes
        if 1 <= self.next_node <= num_nodes:
            vec_to[self.next_node - 1] = 1

        vec_capa = [self.get_available_seats() / cfg.VEH_CAPACITY]

        vec_all = vec_status + vec_from + vec_to + vec_capa
        return vec_all

