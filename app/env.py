import numpy as np
import app.config as cfg

from pprint import pprint
from app.passenger import Passenger
from app.request_status import RequestStatus
from app.vehicle import Vehicle
from app.action_type import ActionType


class RideSharingEnvironment:
    def __init__(self, network, request_list, vehicle_positions):
        self.network = network
        self.all_request_list = request_list
        self.todo_request_list = request_list.copy()

        self.curr_time = 0
        self.vehicles = []
        self.requests = []
        self.vehicle_np_states = []
        self.request_np_states = []
        self.relation_np_states = []

        self.initialize_vehicles(vehicle_positions)

        self.passengers = []
        self.dropped_passengers = 0
        self.last_dropped = []
        self.max_request_time = max(self.all_request_list, key=lambda r: r.request_time).request_time

        self.matched_ids = set()
        self.canceled_ids = set()

        self.base_fare = 1
        self.VOT = 1.0

        self.status_map = {'idle': 0, 'reject': 1, 'pickup': 2, 'dropoff': 3}

    def initialize_vehicles(self, vehicle_position):
        for idx, pos in enumerate(vehicle_position):
            veh = Vehicle(idx, pos, self.network)
            self.vehicles.append(veh)

    def enqueue_requests(self):
        while self.todo_request_list and self.todo_request_list[0].request_time <= self.curr_time:
            r = self.todo_request_list.pop(0)
            r.waiting_time = self.curr_time - r.request_time
            r.time_to_deadline = r.deadline - self.curr_time
            self.requests.append(r)

    def update_np_states(self):
        all_list = []
        for v in self.vehicles:
            print(v)
            all_list.append(v.get_state())
        self.vehicle_np_states = np.array(all_list, dtype=np.float32)
        # print(self.vehicle_np_states)
        # print(self.vehicle_np_states.shape)
        # print(self.vehicle_np_states.dtype)

        all_list = []
        for r in self.requests:
            print(r)
            all_list.append(r.get_state())

        missing = cfg.NUM_REQUEST - len(all_list)
        if missing > 0:
            zero_vec = [0.0] * cfg.REQUEST_INPUT_DIM
            all_list.extend([zero_vec] * missing)

        self.request_np_states = np.array(all_list, dtype=np.float32)
        # print(self.request_np_states)
        # print(self.request_np_states.shape)
        # print(self.request_np_states.dtype)

        all_list = []
        for v in self.vehicles:
            v_list = []
            for r in self.requests:
                need_drop_off = 0
                if r in v.curr_requests:
                    need_drop_off = 1

                if r.status == RequestStatus.PENDING:
                    dur = self.network.get_duration(v.curr_node, r.from_node_id)
                elif r.status == RequestStatus.PICKEDUP and need_drop_off == 1:
                    dur = self.network.get_duration(v.curr_node, r.to_node_id)
                else:
                    dur = 0
                dur = dur / self.network.max_duration
                vec = [need_drop_off, dur]
                v_list.append(vec)

            missing = cfg.NUM_REQUEST - len(v_list)
            if missing > 0:
                zero_vec = [0.0] * cfg.RELATION_INPUT_DIM
                v_list.extend([zero_vec] * missing)

            all_list.append(v_list)
        self.relation_np_states = np.array(all_list, dtype=np.float32)
        # print(self.relation_np_states)
        # print(self.relation_np_states.shape)
        # print(self.relation_np_states.dtype)


    def get_action_mask(self):
        """
        현재의 아래 state를 기준으로 q-value mask를 계산
        - self.request_states
        - self.vehicle_states

        Masking Rule
        - dummy request
        - non-idle vehicle
        - impossible request
            - 최대 대기 시간 = 10분 내에 도달 못감
            - 좌석 부족

        """
        return


    def get_state(self):
        """
        V_t
        현재 차량의 위치: vehicle.current_location
        차량이 이동해야하는 리스트: vehicle.current_path[0] == next_node
        남은 좌석수: vehicle.capacity - len(vehicle.passengers)
        차량의 상태: vehicle.status

        R_t
        요청 최대 20개 queue: 대기 중(request_time None)인 승객을 request_time 순으로 정렬한 후 상위 20개
        부족 시 [0,0,-1]로 패딩
        """
        # 차량 상태 V_t
        vehicle_features = []
        for v in self.vehicles:
            vehicle_features.append([
                v.curr_node,
                v.next_node,
                v.get_remaining_seats(),
                v.status
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
                    p.pickup_time = self.curr_time
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

        dropped = vehicle.move_to_next_location()
        if dropped:
            self.status_map.get(vehicle.status, 3)
            vehicle.current_request = {'type': 'dropoff',
                                       'passenger_ids': [p.id for p in dropped]}

        for p in list(self.passengers):
            if p.start == vehicle.current_location and p.pickup_time is None:
                if vehicle.add_passenger(p):
                    p.pickup_time = self.curr_time
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
            if p.pickup_time is not None and p.dropoff_time == self.curr_time:
                t_wait = max(0, p.pickup_time - p.request_time)
                trav = p.dropoff_time - p.pickup_time
                detour = max(0, trav - p.direct_route_time)
                qos_reward -= (t_wait + detour)

        # moving_vehicles = sum(1 for v in self.vehicles if v.remaining_travel_time > 0)
        # emmision_reward = - moving_vehicles

        reward = matched_reward + dropoff_reward + qos_reward

        # reward = np.clip(reward, -5.0, 5.0)
        return reward

    def generate_passengers_for_current_time(self):
        if self.curr_time <= self.max_request_time:
            new = self.all_request_list[self.all_request_list['Request_time'] == self.curr_time]
            existing_ids = {p.id for p in self.passengers}
            for _, row in new.iterrows():
                if row['User_ID'] not in existing_ids:
                    p = Passenger(row['User_ID'], row['Start_node'], row['End_node'], row['Request_time'], self.network)
                    self.passengers.append(p)
                    self.waiting_passenger_count[p.start] += 1

    def update_current_requirement(self):
        if self.curr_time <= self.max_request_time:
            self.generate_passengers_for_current_time()
        cancel_limit = 10
        for p in list(self.passengers):
            if p.pickup_time is None and (self.curr_time - p.request_time) > cancel_limit:
                self.canceled_ids.add(p.id)
                self.passengers.remove(p)

    def step(self, veh_i, act):
        # self.get_current_requirement()
        reward = self.single_action(veh_i, act)

        next_state = self.get_state()
        return reward, next_state