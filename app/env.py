import random
import numpy as np

from app.passenger import Passenger
from app.vehicle import Vehicle
from app.action_type import ActionType

class RideSharingEnvironment:
    def __init__(self, network, capacity, passenger_data, vehicle_positions):
        self.network = network
        self.vehicles = self.initialize_vehicles(capacity, vehicle_positions)
        # self.passengers = self.initialize_passengers(passenger_data)
        self.passengers = []
        self.passengers_data = passenger_data
        # 기존 환경 변수
        self.time = 0
        self.canceled_passengers = 0
        self.rebalancing_count = 0
        self.old_rebalance = 0
        self.total_passenger_id = 0
        self.matched_passengers = 0
        self.dropped_passengers = 0
        self.last_dropped = []
        self.high_demand_nodes = []
        self.max_request_time = self.passengers_data['Request_time'].max()
        self.waiting_passenger_count = np.zeros(len(self.network.graph.nodes))

        self.matched_ids = set()
        self.canceled_ids = set()

        self.base_fare = 1
        self.VOT = 1.0

        self.status_map = {'idle': 0, 'reject': 1, 'pickup': 2, 'dropoff': 3, 'rebalance': 4}

    def initialize_vehicles(self, capacity, vehicle_position):
        vehicle = []
        for idx, pos in enumerate(vehicle_position):
            vehicle.append(Vehicle(idx, capacity, pos, self.network, self))
        return vehicle

    def get_high_demand_nodes(self):
        recent_requests = [p.start for p in self.passengers if self.time - p.request_time <= 10]
        demand_count = {node: recent_requests.count(node) for node in self.network.graph.nodes}
        historical_data_factor = {node: random.uniform(1.0, 1.5) for node in self.network.graph.nodes}
        for node in demand_count:
            demand_count[node] *= historical_data_factor[node]
        if sum(demand_count.values()) == 0:
            return random.sample(list(self.network.graph.nodes), 3)
        return sorted(demand_count, key=demand_count.get, reverse=True)[:3]

    def generate_passengers_for_current_time(self):
        if self.time <= self.max_request_time:
            new = self.passengers_data[self.passengers_data['Request_time'] == self.time]
            existing_ids = {p.id for p in self.passengers}
            for _, row in new.iterrows():
                if row['User_ID'] not in existing_ids:
                    p = Passenger(row['User_ID'], row['Start_node'], row['End_node'], row['Request_time'], self.network)
                    self.passengers.append(p)
                    self.waiting_passenger_count[p.start] += 1

    def get_state(self):
        """
        V_t
        차량의 상태: vehicle.status
        현재 차량의 위치: vehicle.current_location
        차량이 이동해야하는 리스트: vehicle.current_path[0] == next_node
        남은 좌석수: vehicle.capacity - len(vehicle.passengers)
        재배치 목표: vehicle.rebalance_target

        R_t
        요청 최대 20개 queue: 대기 중(request_time None)인 승객을 request_time 순으로 정렬한 후 상위 20개
        부족 시 [0,0,-1]로 패딩
        """
        # 차량 상태 V_t
        vehicle_features = []
        for v in self.vehicles:
            next_node = v.current_path[0] if v.current_path else -1
            vehicle_features.append([
                v.current_location,
                next_node,
                v.capacity - len(v.passengers),
                self.status_map.get(v.status, 0),
                v.rebalance_target or -1
            ])
        vehicle_array = np.array(vehicle_features, dtype=float)

        # 요청 상태 R_t
        waiting = [p for p in self.passengers if p.pickup_time is None]
        waiting.sort(key=lambda p: p.request_time, reverse=True)
        top20 = waiting[:20]
        req_feats = [[p.start, p.end, p.request_time] for p in top20]
        while len(req_feats) < 20:
            req_feats.append([0, 0, -1])
        requests_array = np.array(req_feats, dtype=float)

        return {'vehicles': vehicle_array, 'requests': requests_array}

    def flatten_state(self, state):
        vehicles_flat = state['vehicles'].flatten()
        requests_flat = state['requests'].flatten()

        return np.concatenate([vehicles_flat, requests_flat])

    def single_action(self, v_idx, action):
        """
        차량 v_idx에 대해 단일 action 적용, 반환 보상
        """
        vehicle = self.vehicles[v_idx]

        step_matched = 0
        qos_reward = 0.0

        if action['action_type'] == ActionType.REJECT:
            self.status_map.get(vehicle.status, 1)
            vehicle.current_request = {'type': 'reject'}

        elif action['action_type'] == ActionType.MATCHING:
            p_idx = action['discrete_index']
            if 0 <= p_idx < len(self.passengers):
                p = self.passengers[p_idx]
                if p.pickup_time is None:
                    p.pickup_time = self.time
                    vehicle.add_passenger(p)
                    self.matched_ids.add(p.id)
                    step_matched += 1
                    self.passengers.remove(p)
                    self.waiting_passenger_count[p.start] -= 1

                    path_to_pickup = self.network.get_shortest_path(
                        vehicle.current_location, p.start
                    )
                    path_to_dropoff = self.network.get_shortest_path(
                        p.start, p.end
                    )
                    full_path = path_to_pickup + path_to_dropoff[1:]
                    vehicle.current_path = full_path

                    if len(full_path) > 1:
                        vehicle.remaining_travel_time = self.network.get_travel_time(
                            [full_path[0], full_path[1]]
                        )
                    else:
                        vehicle.remaining_travel_time = 0

                    self.status_map.get(vehicle.status, 2)
                    vehicle.current_request = {'type': 'pickup', 'passenger_id': p.id}


        elif action['action_type'] == ActionType.REBALANCING:
            high_nodes = self.get_high_demand_nodes()
            r = action['discrete_index']
            target = high_nodes[r]
            vehicle.rebalance_target = target

            full_path = self.network.get_shortest_path(
                vehicle.current_location, target
            )
            vehicle.current_path = full_path
            if len(full_path) > 1:
                vehicle.remaining_travel_time = self.network.get_travel_time(
                    [full_path[0], full_path[1]]
                )
            else:
                vehicle.remaining_travel_time = 0

            self.status_map.get(vehicle.status, 4)
            vehicle.current_request = {'type': 'rebalance', 'target': target}
            self.rebalancing_count += 1

        dropped = vehicle.move_to_next_location()
        if dropped:
            self.status_map.get(vehicle.status, 3)
            vehicle.current_request = {'type': 'dropoff',
                                       'passenger_ids': [p.id for p in dropped]}

        for p in list(self.passengers):
            if p.start == vehicle.current_location and p.pickup_time is None:
                if vehicle.add_passenger(p):
                    p.pickup_time = self.time
                    self.matched_ids.add(p.id)
                    step_matched += 1
                    self.passengers.remove(p)
                    self.status_map.get(vehicle.status, 2)

        cumulative = len(self.matched_ids) + len(self.canceled_ids) + len(self.passengers)
        matched_ratio = len(self.matched_ids) / cumulative if cumulative > 0 else 0
        matched_reward = step_matched * self.base_fare * matched_ratio

        step_drop = len(dropped)
        dropoff_ratio = (self.dropped_passengers / len(self.matched_ids)
                         if len(self.matched_ids) > 0 else 0)
        dropoff_reward = step_drop * self.base_fare * 1.2 * dropoff_ratio

        for p in dropped:
            if p.pickup_time is not None and p.dropoff_time == self.time:
                t_wait = max(0, p.pickup_time - p.request_time)
                trav = p.dropoff_time - p.pickup_time
                detour = max(0, trav - p.direct_route_time)
                qos_reward -= (t_wait + detour)

        # moving_vehicles = sum(1 for v in self.vehicles if v.remaining_travel_time > 0)
        # emmision_reward = - moving_vehicles

        reward = matched_reward + dropoff_reward + qos_reward

        # reward = np.clip(reward, -5.0, 5.0)
        return reward

    def update_current_requirement(self):
        if self.time <= self.max_request_time:
            self.generate_passengers_for_current_time()
        cancel_limit = 10
        for p in list(self.passengers):
            if p.pickup_time is None and (self.time - p.request_time) > cancel_limit:
                self.canceled_ids.add(p.id)
                self.passengers.remove(p)

    def step(self, veh_i, act):
        # self.get_current_requirement()
        reward = self.single_action(veh_i, act)

        next_state = self.get_state()
        return reward, next_state