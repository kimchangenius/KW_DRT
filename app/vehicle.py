class Vehicle:
    def __init__(self, id, capacity, current_location, network, env):
        self.id = id
        self.capacity = capacity
        self.current_location = current_location
        self.network = network
        self.env = env
        self.passengers = []
        self.status = 'idle'
        self.current_request = None

        # 기존 환경 변수
        self.remaining_travel_time = 0
        self.current_path = []
        self.rebalance_target = None
        self.rebalance_count = 0
        self.idle_time = 0
        self.dropped_passengers = 0

    def add_passenger(self, passenger):
        if len(self.passengers) < self.capacity:
            self.passengers.append(passenger)
            self.idle_time = 0
            self.status = self.env.status_map.get(self.status, 2)
            self.current_request = {'type': 'pickup', 'passenger_id': passenger.id}
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
            self.current_location = self.current_path.pop(0)

            if self.current_path:
                next_node = self.current_path[0]
                self.remaining_travel_time = self.network.get_travel_time([self.current_location, next_node])
            else:
                self.remaining_travel_time = 0

        if not self.current_path and self.remaining_travel_time == 0:
            self.idle_time += 1

        for passenger in self.passengers[:]:
            if passenger.end == self.current_location:
                self.remove_passenger(passenger)
                passenger.dropoff_time = self.env.time
                self.dropped_passengers += 1
                self.env.dropped_passengers += 1
                dropped.append(passenger)

        return dropped

