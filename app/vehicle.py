import app.config as cfg

from app.vehicle_status import VehicleStatus


class Vehicle:
    def __init__(self, veh_id, curr_node, network):
        self.id = veh_id        # 0부터 N-1까지 값을 가짐
        self.network = network

        self.status = VehicleStatus.IDLE
        self.curr_node = curr_node
        self.next_node = 0
        self.curr_requests = []

        # 기존 환경 변수
        self.remaining_travel_time = 0
        self.current_path = []
        self.idle_time = 0
        self.dropped_passengers = 0

    def __str__(self):
        return (f"Vehicle(veh_id={self.id}, "
                f"curr={self.curr_node}, next={self.next_node}, "
                f"status={self.status}")

    def get_available_seats(self):
        num_curr_passengers = 0
        for r in self.curr_requests:
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

    def add_passenger(self, passenger):
        if len(self.passengers) < cfg.VEH_CAPACITY:
            self.passengers.append(passenger)
            self.idle_time = 0
            # self.status = self.env.status_map.get(self.status, 2)
            self.curr_request = {'type': 'pickup', 'passenger_id': passenger.id}
            return True
        return False

    def remove_passenger(self, passenger):
        if passenger in self.passengers:
            self.passengers.remove(passenger)

    def move_to_next_location(self):
        dropped = []

        if self.remaining_travel_time > 0:
            self.remaining_travel_time -= 1

        if self.remaining_travel_time == 0 and self.current_path:
            self.curr_node = self.current_path.pop(0)

            # if self.current_path:
            #     next_node = self.current_path[0]
            #     self.remaining_travel_time = self.network.get_travel_time([self.curr_node, next_node])
            # else:
            #     self.remaining_travel_time = 0

        if not self.current_path and self.remaining_travel_time == 0:
            self.idle_time += 1

        for passenger in self.passengers[:]:
            if passenger.end == self.curr_node:
                self.remove_passenger(passenger)
                # passenger.dropoff_time = self.env.curr_time
                self.dropped_passengers += 1
                # self.env.dropped_passengers += 1
                dropped.append(passenger)

        return dropped

