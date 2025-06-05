import copy
import numpy as np
import app.config as cfg

from pprint import pprint
from app.request_status import RequestStatus
from app.vehicle import Vehicle
from app.vehicle_status import VehicleStatus
from app.action_type import ActionType


class RideSharingEnvironment:
    def __init__(self, network, original_request_list, vehicle_init_pos):
        self.network = network
        self.original_request_list = original_request_list
        self.vehicle_init_pos = vehicle_init_pos

        self.curr_time = None
        self.curr_step = None

        self.future_request_list = None     # request들 중 미래에 들어올 것들 (정렬되어 있음)
        self.active_request_list = None  # 현재 request 슬롯에 들어갈 것들 (최대 개수가 정해져있음)
        self.done_request_list = None

        self.vehicle_list = None

        self.vehicle_state = None
        self.request_state = None
        self.relation_state = None
        self.state = None

        self.logs = []

    def reset(self):
        self.curr_time = 0
        self.curr_step = 0

        self.future_request_list = copy.deepcopy(self.original_request_list)
        self.active_request_list = []
        self.done_request_list = []

        self.vehicle_list = []

        self.initialize_vehicles()
        self.handle_time_update()
        self.sync_state()
        return self.state

    def print_vehicles(self):
        for v in self.vehicle_list:
            print(v)

    def print_active_requests(self):
        print('Num Requests : {}'.format(len(self.active_request_list)))
        log = str(len(self.active_request_list))
        self.logs.append(log)
        for r in self.active_request_list:
            print(r)

    def print_done_requests(self):
        print('Num Requests : {}'.format(len(self.done_request_list)))
        for r in self.done_request_list:
            print(r)

    def print_statistics(self):
        num_served = 0
        num_cancelled = 0
        print("====================== Statistics ======================")
        print('Request Done at Time : {}'.format(self.curr_time))
        print('Num Requests : {}'.format(len(self.done_request_list)))
        for r in self.done_request_list:
            if r.status == RequestStatus.SERVED:
                num_served += 1
            if r.status == RequestStatus.CANCELLED:
                num_cancelled += 1
        print('Num Served : {}'.format(num_served))
        print('Num Cancelled : {}'.format(num_cancelled))

    def print_logs(self):
        for l in self.logs:
            print(l)
        self.logs = []

    def initialize_vehicles(self):

        for idx in range(cfg.MAX_NUM_VEHICLES):
            pos = self.vehicle_init_pos[idx]
            veh = Vehicle(idx, pos, self.network)
            self.vehicle_list.append(veh)

        # for idx, pos in enumerate(self.vehicle_init_pos):
        #     veh = Vehicle(idx, pos, self.network)
        #     self.vehicle_list.append(veh)

    # 시간이 업데이트 될 때 필요한 모든 것들을 업데이트 함
    def handle_time_update(self):
        d_reward_list = []

        # 현재 시간에 들어올 새로운 요청을 추가
        while self.future_request_list and self.future_request_list[0].request_time <= self.curr_time:
            r = self.future_request_list.pop(0)
            self.active_request_list.append(r)

        # Vehicle 업데이트
        for v in self.vehicle_list:
            if v.status == VehicleStatus.REJECT:
                # REJECT이면 IDLE로 전환
                v.status = VehicleStatus.IDLE

            if v.status == VehicleStatus.PICKUP:
                if v.target_arrival_time == self.curr_time:
                    # 이번 시간에 pickup 도착했으면
                    r = v.target_request

                    # V 업데이트
                    v.status = VehicleStatus.IDLE
                    v.curr_node = v.next_node
                    v.next_node = 0
                    v.target_request = None
                    v.target_arrival_time = -1

                    if r.status == RequestStatus.CANCELLED:
                        # Pickup 실패
                        v.active_request_list.remove(r)

                        # Cancel 페널티 (Pickup 결정, 지연 보상(페널티))
                        p_action_id = "{}_1".format(r.id)
                        d_reward_list.append([p_action_id, -1])
                    else:
                        # Pickup 성공
                        v.num_passengers += r.num_passengers
                        assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"

                        # R 업데이트
                        r.status = RequestStatus.PICKEDUP
                        r.waiting_time = self.curr_time - r.request_time    # 마지막 확정 업데이트
                        r.pickup_at = self.curr_time                      # 마지막 확정 업데이트

            if v.status == VehicleStatus.DROPOFF:
                if v.target_arrival_time == self.curr_time:
                    # 이번 시간에 dropoff 도착했으면
                    r = v.target_request

                    # V 업데이트
                    v.status = VehicleStatus.IDLE
                    v.curr_node = v.next_node
                    v.next_node = 0
                    v.target_request = None
                    v.target_arrival_time = -1
                    v.active_request_list.remove(r)
                    v.num_passengers -= r.num_passengers
                    assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"

                    # R 업데이트
                    r.status = RequestStatus.SERVED
                    r.arrival_due_left = r.arrival_due - self.curr_time     # 마지막 확정 업데이트
                    if r.arrival_due_left < 0:
                        r.arrival_due_left = 0
                    r.in_vehicle_time = self.curr_time - r.pickup_at         # 마지막 확정 업데이트
                    r.dropoff_at = self.curr_time                         # 마지막 확정 업데이트
                    self.active_request_list.remove(r)
                    self.done_request_list.append(r)

                    # Request 완료 보상 (Pickup and Dropoff, 지연 보상)
                    p_action_id = "{}_1".format(r.id)
                    d_action_id = "{}_2".format(r.id)
                    d_reward_list.append([p_action_id, 0.5])
                    d_reward_list.append([d_action_id, 0.5])

        # Request 업데이트
        cancelled_list = []
        for r in self.active_request_list:
            r.arrival_due_left = r.arrival_due - self.curr_time
            if r.arrival_due_left < 0:
                r.arrival_due_left = 0
            if r.status == RequestStatus.PENDING or r.status == RequestStatus.ACCEPTED:
                r.waiting_time = self.curr_time - r.request_time

                # waiting_time 초과 확인
                if r.waiting_time >= cfg.MAX_WAIT_TIME:
                    r.status = RequestStatus.CANCELLED
                    cancelled_list.append(r)
            if r.status == RequestStatus.PICKEDUP:
                r.in_vehicle_time = self.curr_time - r.pickup_at

        for cr in cancelled_list:
            self.active_request_list.remove(cr)
            self.done_request_list.append(cr)

        for idx, r in enumerate(self.active_request_list):
            r.slot_idx = idx

        return d_reward_list


    # 기존에 가진 자료구조들을 토대로 state 형태로 만들어주기만 하는 역할
    # 이 안에서 상태가 바뀌거나 업데이트가 되어서는 안됨
    def sync_state(self):
        # Vehicle State 생성
        all_list = []
        for v in self.vehicle_list:
            all_list.append(v.get_vector())
        self.vehicle_state = np.array(all_list, dtype=np.float32)
        # print(self.vehicle_state)
        # print(self.vehicle_state.shape)
        # print(self.vehicle_state.dtype)

        # Request State 생성
        all_list = []
        for idx, r in enumerate(self.active_request_list):
            if idx >= cfg.MAX_NUM_REQUEST:
                break
            all_list.append(r.get_vector())

        missing = cfg.MAX_NUM_REQUEST - len(all_list)
        if missing > 0:
            zero_vec = [0.0] * cfg.REQUEST_INPUT_DIM
            all_list.extend([zero_vec] * missing)

        assert len(all_list) == cfg.MAX_NUM_REQUEST, "MAX_NUM_REQUEST mismatch"
        self.request_state = np.array(all_list, dtype=np.float32)
        # print(self.request_state)
        # print(self.request_state.shape)
        # print(self.request_state.dtype)

        # Relation State 생성
        all_list = []
        for v in self.vehicle_list:
            v_list = []
            for idx, r in enumerate(self.active_request_list):
                if idx >= cfg.MAX_NUM_REQUEST:
                    break
                need_drop_off = 0
                if r in v.active_request_list:
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

            missing = cfg.MAX_NUM_REQUEST - len(v_list)
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
            # 현재 차량이 Idle할때
            for idx, r in enumerate(self.active_request_list):
                # 현재 request가 Dummy가 아닐 경우,

                # 가능한 액션 개수를 넘는 초과 요청은 상태로 다루지 않아, 마스킹 대상이 아님
                if idx >= cfg.MAX_NUM_REQUEST:
                    break

                # 현재 request가 이미 해당 차량에 assigned된 경우, 즉 drop off 대상
                if r.status == RequestStatus.PICKEDUP:
                    if r.assigned_v_id == v.id:
                        v_row.append(1)
                    else:
                        v_row.append(0)
                elif r.status == RequestStatus.PENDING:
                    v_empty_seat = cfg.VEH_CAPACITY - v.num_passengers
                    if v_empty_seat >= r.num_passengers:
                        v_row.append(1)
                    else:
                        v_row.append(0)
                else:
                    v_row.append(0)

            # 현재 request가 Dummy 일 경우
            missing = cfg.MAX_NUM_REQUEST - len(self.active_request_list)
            if missing > 0:
                v_row.extend([0] * missing)

            # Reject 추가
            v_row.append(1)
            assert len(v_row) == cfg.POSSIBLE_ACTION, "Action mask length mismatch"
            all_list.append(v_row)
        return np.array(all_list, dtype=np.float32)

    def enrich_action(self, action):
        vehicle_idx = action[0]
        action_idx = action[1]
        v = self.vehicle_list[vehicle_idx]
        if action_idx == cfg.POSSIBLE_ACTION - 1:
            action[2]['r_id'] = ""
            action[2]['type'] = ActionType.REJECT
        else:
            r = self.active_request_list[action_idx]
            action[2]['r_id'] = r.id
            if r not in v.active_request_list:
                action[2]['type'] = ActionType.PICKUP
            else:
                action[2]['type'] = ActionType.DROPOFF
        action[2]['id'] = "{}_{}".format(action[2]['r_id'], action[2]['type'].value)

    def step(self, action):
        # print('Env: Curr action : {}'.format(action))
        vehicle_idx = action[0]
        action_idx = action[1]
        assert action_idx < cfg.POSSIBLE_ACTION, "Invalid action"

        reward = 0
        info = {
            'is_pending': False,
            'has_delayed_reward': False,
            'action_id_list': None,
            'reward': None
        }
        v = self.vehicle_list[vehicle_idx]

        if action_idx == cfg.POSSIBLE_ACTION - 1:
            # Reject
            v.status = VehicleStatus.REJECT
        else:
            # Matching
            # 어떤 요청이 채택된 경우 - Pickup 하러 가거나 Dropoff 하러 가야 함
            r = self.active_request_list[action_idx]

            if r not in v.active_request_list:
                # Pickup 하러 가야하는 경우
                info['is_pending'] = True
                v.status = VehicleStatus.PICKUP
                v.active_request_list.append(r)
                v.next_node = r.from_node_id
                v.target_request = r
                pickup_duration = self.network.get_duration(v.curr_node, v.next_node)
                v.target_arrival_time = self.curr_time + pickup_duration

                r.status = RequestStatus.ACCEPTED
                r.assigned_v_id = v.id

                # Pickup 결정 보상 (최대 1점, 즉시 보상)
                reward = 0.5 * (1 - pickup_duration / self.network.max_duration)
                + 0.5 * (1 - r.waiting_time / cfg.MAX_WAIT_TIME)

                # 만약 Pickup이 즉시 완료된다면
                if v.curr_node == v.next_node:
                    v.status = VehicleStatus.IDLE
                    v.next_node = 0
                    v.target_request = None
                    v.target_arrival_time = -1
                    v.num_passengers += r.num_passengers
                    assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"

                    r.status = RequestStatus.PICKEDUP
                    r.waiting_time = self.curr_time - r.request_time  # 마지막 확정 업데이트
                    r.pickup_at = self.curr_time  # 마지막 확정 업데이트

                    # Pickup 결정 즉시 완료 보상
                    reward += 1
            else:
                # Dropoff 하러 가야하는 경우
                info['is_pending'] = True
                v.status = VehicleStatus.DROPOFF
                v.next_node = r.to_node_id
                v.target_request = r
                dropoff_duration = self.network.get_duration(v.curr_node, v.next_node)
                v.target_arrival_time = self.curr_time + dropoff_duration

                # Dropoff 결정 리워드 (즉시 보상)
                in_vehicle_ratio = r.in_vehicle_time / cfg.MAX_INVEHICLE_TIME
                if in_vehicle_ratio > 1:
                    in_vehicle_ratio = 1
                reward = 0.5 * (1 - dropoff_duration / self.network.max_duration)
                + 0.5 * (1 - in_vehicle_ratio)

                # 만약 Dropoff가 즉시 완료된다면
                if v.curr_node == v.next_node:
                    v.status = VehicleStatus.IDLE
                    v.next_node = 0
                    v.target_request = None
                    v.target_arrival_time = -1
                    v.active_request_list.remove(r)
                    v.num_passengers -= r.num_passengers
                    assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"

                    r.status = RequestStatus.SERVED
                    r.arrival_due_left = r.arrival_due - self.curr_time     # 마지막 확정 업데이트
                    r.in_vehicle_time = self.curr_time - r.pickup_at      # 마지막 확정 업데이트
                    r.dropoff_at = self.curr_time                         # 마지막 확정 업데이트
                    self.active_request_list.remove(r)
                    self.done_request_list.append(r)

                    # Dropoff 결정 즉시 완료 보상
                    reward += 1

                    # Request 완료 보상 (Dropoff, 즉시 보상)
                    reward += 0.5
                    info['is_pending'] = False

                    # Request 완료 보상 (Pickup, 지연 보상)
                    info['has_delayed_reward'] = True
                    action_id = "{}_1".format(r.id)
                    info['action_id_list'] = [action_id]
                    info['reward'] = 0.5

        self.sync_state()
        self.curr_step += 1

        return self.state, reward, info

    def has_idle_vehicle(self):
        has = False
        for v in self.vehicle_list:
            if v.status == VehicleStatus.IDLE:
                has = True
        return has

    def is_done(self):
        if len(self.active_request_list) == 0 and len(self.future_request_list) == 0:
            return True
        return False
