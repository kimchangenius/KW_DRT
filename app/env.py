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

        # Logging
        self.logs = []

    def reset(self):
        self.curr_time = 0
        self.curr_step = 0

        self.future_request_list = copy.deepcopy(self.original_request_list)
        self.active_request_list = []
        self.done_request_list = []

        self.vehicle_list = []
        
        # 이벤트 시퀀스 초기화
        self.event_sequences = [[] for _ in range(cfg.MAX_NUM_VEHICLES)]

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
                        # Pickup 실패 - 안전한 제거
                        if r in v.active_request_list:
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
                        
                        # 픽업 완료 이벤트 기록
                        if hasattr(self, 'event_sequences'):
                            seq = f"P_{r.id}"
                            self.event_sequences[v.id].append(seq)

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
                    
                    # 드롭오프 완료 이벤트 기록
                    if hasattr(self, 'event_sequences'):
                        seq = f"D_{r.id}"
                        self.event_sequences[v.id].append(seq)
                    
                    # 안전한 요청 제거 (중복 제거 방지)
                    if r in self.active_request_list:
                        self.active_request_list.remove(r)
                    self.done_request_list.append(r)

                    # Request 완료 보상 (Pickup and Dropoff, 지연 보상)
                    p_action_id = "{}_1".format(r.id)
                    d_action_id = "{}_2".format(r.id)
                    d_reward_list.append([p_action_id, 0.5])
                    d_reward_list.append([d_action_id, 0.5])

                    # Logging
                    v.num_serve += 1

        # Request 업데이트
        cancelled_list = []
        for r in self.active_request_list:
            r.arrival_due_left = r.arrival_due - self.curr_time
            if r.arrival_due_left < 0:
                r.arrival_due_left = 0
            if r.status == RequestStatus.PENDING or r.status == RequestStatus.ACCEPTED:
                r.waiting_time = self.curr_time - r.request_time

                # waiting_time 초과 확인 (PENDING)
                if r.status == RequestStatus.PENDING and r.waiting_time >= cfg.MAX_WAIT_TIME:
                    r.status = RequestStatus.CANCELLED
                    cancelled_list.append(r)

                # ACCEPTED 후 픽업까지의 강력 타임아웃
                # 차량에 assign되었지만 실제 PICKEDUP으로 전환되지 못하는 요청을 강제로 취소
                if r.status == RequestStatus.ACCEPTED:
                    # assign 이후 경과 시간 계산
                    if getattr(r, 'accepted_at', None) is None:
                        # 최초 ACCEPT 기록이 없으면 지금 시각으로 초기화
                        r.accepted_at = self.curr_time
                    accepted_elapsed = self.curr_time - r.accepted_at
                    if accepted_elapsed >= cfg.MAX_ACCEPT_WAIT:
                        r.status = RequestStatus.CANCELLED
                        cancelled_list.append(r)
            if r.status == RequestStatus.PICKEDUP:
                r.in_vehicle_time = self.curr_time - r.pickup_at

        for cr in cancelled_list:
            self.active_request_list.remove(cr)
            self.done_request_list.append(cr)

        # 취소된 요청이 차량 active 리스트에 남아있다면 안전하게 제거하고 지연보상 부여
        if len(cancelled_list) > 0:
            for v in self.vehicle_list:
                to_remove = []
                for r in v.active_request_list:
                    if r.status == RequestStatus.CANCELLED:
                        to_remove.append(r)
                        # Pickup 실패 지연보상 부여
                        p_action_id = "{}_1".format(r.id)
                        d_reward_list.append([p_action_id, -1])
                for r in to_remove:
                    if r in v.active_request_list:
                        v.active_request_list.remove(r)

        for idx, r in enumerate(self.active_request_list):
            r.slot_idx = idx

        return d_reward_list, [r.id for r in cancelled_list]


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
                    # 현재 승객 수 + 이미 수락한 요청들의 승객 수 계산
                    future_passengers = v.num_passengers
                    for pending_r in v.active_request_list:
                        if pending_r.status == RequestStatus.ACCEPTED:
                            future_passengers += pending_r.num_passengers
                    
                    v_empty_seat = cfg.VEH_CAPACITY - future_passengers
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
        """단일 액션 처리 (기존 방식)"""
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

            # Logging
            v.idle_time += 1
        else:
            # Matching
            # 어떤 요청이 채택된 경우 - Pickup 하러 가거나 Dropoff 하러 가야 함
            r = self.active_request_list[action_idx]
            
            # Request 액션은 즉시 완료되므로 여기서 기록
            if action_idx != cfg.POSSIBLE_ACTION - 1:  # REJECT가 아닌 경우
                request_id = action[2]['r_id'] if len(action) > 2 and 'r_id' in action[2] else None
                if request_id:
                    seq = f"R_{request_id}"  # Request 액션 기록
                    self.event_sequences[vehicle_idx].append(seq)

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

                # Logging
                v.num_accept += 1

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
                    info['is_pending'] = False

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

                    # Logging
                    v.num_serve += 1

        self.sync_state()
        self.curr_step += 1

        return self.state, reward, info
    
    def step_multi(self, actions):
        """여러 액션을 동시에 처리"""
        total_reward = 0
        all_info = {
            'is_pending': False,
            'has_delayed_reward': False,
            'action_id_list': [],
            'reward': 0,
            'individual_rewards': [],
            'individual_infos': []
        }
        
        # 각 액션을 개별적으로 처리
        for action in actions:
            vehicle_idx = action[0]
            action_idx = action[1]
            assert action_idx < cfg.POSSIBLE_ACTION, "Invalid action"
            
            reward, info = self.process_single_action(vehicle_idx, action_idx)
            total_reward += reward
            all_info['individual_rewards'].append(reward)
            all_info['individual_infos'].append(info)
            
            # Request 액션은 즉시 완료되므로 여기서 기록
            if action_idx != cfg.POSSIBLE_ACTION - 1:  # REJECT가 아닌 경우
                request_id = action[2]['r_id'] if len(action) > 2 and 'r_id' in action[2] else None
                if request_id:
                    seq = f"R_{request_id}"  # Request 액션 기록
                    self.event_sequences[vehicle_idx].append(seq)
            
            # 지연 보상 정보 수집
            if info['has_delayed_reward']:
                all_info['has_delayed_reward'] = True
                if all_info['action_id_list'] is None:
                    all_info['action_id_list'] = []
                all_info['action_id_list'].extend(info['action_id_list'])
            
            # Pending 정보 수집
            if info['is_pending']:
                all_info['is_pending'] = True
        
        all_info['reward'] = total_reward
        
        # 상태 동기화 및 스텝 카운터 증가
        self.sync_state()
        self.curr_step += 1
        
        return self.state, total_reward, all_info
    
    def process_single_action(self, vehicle_idx, action_idx):
        """단일 액션 처리 (내부 메서드)"""
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

            # Logging
            v.idle_time += 1
        else:
            # Matching
            # 어떤 요청이 채택된 경우 - Pickup 하러 가거나 Dropoff 하러 가야 함
            # IndexError 방지: action_idx가 유효한 범위인지 확인
            if action_idx >= len(self.active_request_list):
                # 요청이 이미 처리되어 제거된 경우
                reward = 0
                info['is_pending'] = False
                return reward, info
            
            r = self.active_request_list[action_idx]
            
            # 요청이 이미 처리된 경우 체크
            if r.status in [RequestStatus.SERVED, RequestStatus.CANCELLED]:
                # 이미 처리된 요청인 경우
                reward = 0
                info['is_pending'] = False
                return reward, info

            if r not in v.active_request_list:
                # Pickup 하러 가야하는 경우
                # 용량 체크: 승객을 태울 수 있는지 확인
                if v.num_passengers + r.num_passengers > cfg.VEH_CAPACITY:
                    # 용량 초과 - 이 액션은 무효
                    reward = 0
                    info['is_pending'] = False
                    return reward, info
                
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

                # Logging
                v.num_accept += 1

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
                    info['is_pending'] = False

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

                    # Logging
                    v.num_serve += 1

        return reward, info

    def has_idle_vehicle(self):
        has = False
        for v in self.vehicle_list:
            if v.status == VehicleStatus.IDLE:
                has = True
        return has

    def is_done(self):
        # 기본 종료 조건: 모든 요청이 처리됨
        if len(self.active_request_list) == 0 and len(self.future_request_list) == 0:
            # for v in self.vehicle_list:
            #     if v.status != VehicleStatus.IDLE:
            #         return False
            return True
        
        # 추가 종료 조건들 (DDQN을 위한 관대한 종료)
        
        # 1. 요청이 거의 처리되지 않고 시간이 많이 지난 경우
        # 근거: 정상적인 시뮬레이션은 60-80시간 내에 완료되므로, 80시간 후에도 5개 미만 처리되면 비효율적
        # if self.curr_time > 80 and len(self.done_request_list) < 5:
        #     return True
        
        # # 2. 활성 요청이 많지만 처리되지 않는 경우 (DDQN이 모든 요청을 거부할 때)
        # # 근거: 활성 요청이 15개 이상이고 평균 대기시간이 MAX_WAIT_TIME의 80% 이상이면 비효율적
        # if len(self.active_request_list) > 15 and self.curr_time > 40:
        #     avg_waiting_time = sum(r.waiting_time for r in self.active_request_list) / len(self.active_request_list)
        #     max_wait_threshold = cfg.MAX_WAIT_TIME * 0.8  # MAX_WAIT_TIME의 80%
        #     if avg_waiting_time > max_wait_threshold:
        #         return True
        
        # # 3. 모든 차량이 IDLE이고 요청이 있지만 처리되지 않는 경우
        # # 근거: 차량들이 일하지 않고 있는데 처리할 요청이 있다면 시스템이 멈춘 상태
        # all_idle = all(v.status == VehicleStatus.IDLE for v in self.vehicle_list)
        # if all_idle and len(self.active_request_list) > 0 and self.curr_time > 25:
        #     return True
        
        return False
