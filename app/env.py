import copy
import numpy as np
import app.config as cfg

from pprint import pprint
from app.request_status import RequestStatus
from app.vehicle import Vehicle
from app.vehicle_status import VehicleStatus


class RideSharingEnvironment:
    def __init__(self, network, original_request_list, vehicle_init_pos):
        self.network = network
        self.original_request_list = original_request_list
        self.vehicle_init_pos = vehicle_init_pos

        self.curr_time = None
        self.curr_step = None
        self.todo_request_list = None   # request들 중 미래에 들어올 것들 (정렬되어 있음)

        self.vehicle_list = None
        self.request_list = None        # 현재 request 슬롯에 들어갈 것들 (최대 개수가 정해져있음)

        self.vehicle_state = None
        self.request_state = None
        self.relation_state = None
        self.state = None

    def reset(self):
        self.curr_time = 0
        self.curr_step = 0
        self.todo_request_list = copy.deepcopy(self.original_request_list)

        self.vehicle_list = []
        self.request_list = []

        self.initialize_vehicles()
        self.handle_time_update()
        self.sync_state()
        return self.state

    def print_vehicles(self):
        for v in self.vehicle_list:
            print(v)

    def print_requests(self):
        for r in self.request_list:
            print(r)

    def initialize_vehicles(self):
        for idx, pos in enumerate(self.vehicle_init_pos):
            veh = Vehicle(idx, pos, self.network)
            self.vehicle_list.append(veh)

    # 시간이 업데이트 될 때 필요한 모든 것들을 업데이트 함
    def handle_time_update(self):
        # 현재 시간에 들어올 새로운 요청을 추가
        while self.todo_request_list and self.todo_request_list[0].request_time <= self.curr_time:
            r = self.todo_request_list.pop(0)
            r.waiting_time = self.curr_time - r.request_time
            r.time_to_deadline = r.arrival_deadline - self.curr_time
            self.request_list.append(r)
        for idx, r in enumerate(self.request_list):
            r.slot_idx = idx

        # Vehicle 상태 업데이트
        for v in self.vehicle_list:
            if v.status == VehicleStatus.REJECT:
                # REJECT이면 IDLE로 전환
                v.status = VehicleStatus.IDLE
            if v.status == VehicleStatus.PICKUP:
                if v.target_arrival_time == self.curr_time:
                    # 이번 시간에 pickup 도착했으면
                    v.status = VehicleStatus.IDLE
                    v.target_arrival_time = -1
                    v.curr_node = v.next_node
                    v.next_node = 0

            if v.status == VehicleStatus.DROPOFF:
                if v.target_arrival_time == self.curr_time:
                    # 이번 시간에 dropoff 도착했으면
                    v.status = VehicleStatus.IDLE
                    v.target_arrival_time = -1
                    v.active_requests.remove(v.target_request)
                    v.curr_node = v.next_node
                    v.next_node = 0

        # TODO: Request 상태 업데이트 (대기 시간 등...)





    # 기존에 가진 자료구조들을 토대로 state 형태로 만들어주기만 하는 역할
    # 이 안에서 상태가 바뀌거나 업데이트가 되어서는 안됨
    def sync_state(self):
        all_list = []
        for v in self.vehicle_list:
            all_list.append(v.get_vector())
        self.vehicle_state = np.array(all_list, dtype=np.float32)
        # print(self.vehicle_state)
        # print(self.vehicle_state.shape)
        # print(self.vehicle_state.dtype)

        all_list = []
        for r in self.request_list:
            all_list.append(r.get_vector())

        missing = cfg.NUM_REQUEST - len(all_list)
        if missing > 0:
            zero_vec = [0.0] * cfg.REQUEST_INPUT_DIM
            all_list.extend([zero_vec] * missing)

        self.request_state = np.array(all_list, dtype=np.float32)
        # print(self.request_state)
        # print(self.request_state.shape)
        # print(self.request_state.dtype)

        all_list = []
        for v in self.vehicle_list:
            v_list = []
            for r in self.request_list:
                need_drop_off = 0
                if r in v.active_requests:
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
        self.relation_state = np.array(all_list, dtype=np.float32)
        # print(self.relation_state)
        # print(self.relation_state.shape)
        # print(self.relation_state.dtype)

        self.state = [
            np.expand_dims(self.vehicle_state, axis=0),
            np.expand_dims(self.request_state, axis=0),
            np.expand_dims(self.relation_state, axis=0)
        ]

    def get_action_mask(self):
        """
        Masking Rule
        - Non-idle vehicle
        - Request
            - Non-Pending vehicle
            - Seat Not available
        - Dummy request
        """
        all_list = []
        for v in self.vehicle_list:

            # 현재 차량이 Non-idle
            if v.status != VehicleStatus.IDLE:
                all_list.append([0] * cfg.POSSIBLE_ACTION)
                continue

            v_row = []
            # 현재 차량이 Idle, 현재 request가 Dummy가 아닐 경우
            for r in self.request_list:
                # 현재 request가 이미 해당 차량에 assigned된 경우, 즉 drop off 대상
                if r.status == RequestStatus.PICKEDUP:
                    if r.assigned_v_id == v.id:
                        v_row.append(1)
                    else:
                        v_row.append(0)
                elif r.status == RequestStatus.PENDING:
                    if v.get_available_seats() >= r.num_passengers:
                        v_row.append(1)
                    else:
                        v_row.append(0)
                else:
                    v_row.append(0)

            # 현재 request가 Dummy 일 경우
            missing = cfg.NUM_REQUEST - len(self.request_list)
            if missing > 0:
                v_row.extend([0] * missing)

            # Reject 추가
            v_row.append(1)
            assert len(v_row) == cfg.POSSIBLE_ACTION, "Action mask length mismatch"
            all_list.append(v_row)
        return np.array(all_list, dtype=np.float32)

    # next_state, reward, done 을 기본적으로 리턴
    def step(self, action):
        print('Env: current action : {}'.format(action))
        vehicle_idx = action[0]
        action_idx = action[1]
        v = self.vehicle_list[vehicle_idx]

        curr_reward = 0
        done = False

        if action_idx == 20:
            # Reject
            v.status = VehicleStatus.REJECT
        else:
            # Matching
            # 어떤 요청이 채택된 경우 - Pickup 하러 가거나 Dropoff 하러 가야 함
            r = self.request_list[action_idx]
            if r not in v.active_requests:
                # Pickup 하러 가야하는 경우
                v.status = VehicleStatus.PICKUP
                v.active_requests.append(r)
                v.next_node = r.from_node_id
                v.target_request = r
                v.target_arrival_time = self.curr_time + self.network.get_duration(v.curr_node, v.next_node)

                r.status = RequestStatus.ACCEPTED
                r.assigned_v_id = v.id

                curr_reward += 1
            else:
                # Dropoff 하러 가야하는 경우
                v.status = VehicleStatus.DROPOFF
                v.next_node = r.to_node_id
                v.target_request = r
                v.target_arrival_time = self.curr_time + self.network.get_duration(v.curr_node, v.next_node)

        self.sync_state()
        self.curr_step += 1

        return self.state, curr_reward, done

    def has_idle_vehicle(self):
        has = False
        for v in self.vehicle_list:
            if v.status == VehicleStatus.IDLE:
                has = True
        return has
