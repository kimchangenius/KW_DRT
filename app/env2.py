import copy
import numpy as np
import app.config as cfg

from app.action_type import ActionType
from app.request import Request
from app.request_status import RequestStatus
from app.vehicle import Vehicle
from app.vehicle_status import VehicleStatus


class RideSharingEnvironment:
    """
    env2: occupancy 개선 실험용 reward shaping 버전.

    env.py의 제약/후보 생성 로직은 유지하고, 보상만 승객 수와 탑승률을 더 직접
    반영하도록 조정한다.

    상태 표현이 (V, R) 통째 텐서가 아니라, agent 쪽에서 (v, r) 페어 후보를
    enumerate_pair_candidates()로 받아 페어 단위 forward를 하는 구조.

    따라서 sync_state/get_action_mask/enrich_action 같은 padded-state 인터페이스는
    더 이상 존재하지 않는다. agent는 vehicle_list / active_request_list를 직접 본다.
    """

    def __init__(self, network, original_request_list, vehicle_init_pos):
        self.network = network
        self.original_request_list = original_request_list
        self.vehicle_init_pos = vehicle_init_pos

        self.curr_time = None
        self.curr_step = None

        self.future_request_list = None
        self.active_request_list = None
        self.done_request_list = None

        self.vehicle_list = None

        # Logging
        self.logs = []

    @staticmethod
    def _penalty_time_over_cap(value, cap):
        v, c = float(value), float(cap)
        if v <= c:
            return 0.0
        scale = getattr(cfg, 'EXCESS_TIME_PENALTY_SCALE', 0.1)
        return -scale * (v - c)

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    def _load_ratio(self, passengers):
        return self._clip01(float(passengers) / max(float(cfg.VEH_CAPACITY), 1.0))

    def _passenger_ratio(self, request):
        return self._load_ratio(getattr(request, 'num_passengers', 1))

    def _distance_score(self, duration):
        return self._clip01(
            1.0 - float(duration) / max(float(self.network.max_duration), 1.0)
        )

    def _wait_score(self, request, extra_time=0.0):
        waited = float(request.waiting_time) + float(extra_time)
        return self._clip01(1.0 - waited / max(float(cfg.MAX_WAIT_TIME), 1.0))

    def _deadline_score(self, request, in_vehicle_time=None):
        if in_vehicle_time is None:
            in_vehicle_time = request.in_vehicle_time
        slack = self.in_vehicle_time_limit(request) - float(in_vehicle_time)
        return self._clip01(slack / max(float(cfg.MAX_INVEHICLE_TIME), 1.0))

    def _pickup_action_reward(self, vehicle, request, pickup_duration):
        load_before = float(vehicle.num_passengers)
        load_after = load_before + float(request.num_passengers)
        share_bonus = 0.15 if load_before > 0 else 0.0
        return (
            0.25 * self._distance_score(pickup_duration)
            + 0.15 * self._wait_score(request, pickup_duration)
            + 0.35 * self._load_ratio(load_after)
            + 0.20 * self._passenger_ratio(request)
            + share_bonus
        )

    def _dropoff_action_reward(self, vehicle, request, dropoff_duration):
        projected_in_vehicle = float(request.in_vehicle_time) + float(dropoff_duration)
        return (
            0.20 * self._distance_score(dropoff_duration)
            + 0.25 * self._deadline_score(request, projected_in_vehicle)
            + 0.25 * self._load_ratio(vehicle.num_passengers)
            + 0.30 * self._passenger_ratio(request)
        )

    def _service_completion_reward(self, vehicle, request):
        load_before_dropoff = vehicle.num_passengers
        shared_bonus = (
            0.25
            if load_before_dropoff > getattr(request, 'num_passengers', 1)
            else 0.0
        )
        return (
            0.50
            + 0.70 * self._passenger_ratio(request)
            + 0.60 * self._load_ratio(load_before_dropoff)
            + 0.25 * self._deadline_score(request, request.in_vehicle_time)
            + shared_bonus
        )

    def _cancel_penalty(self, request):
        return -(0.50 + 0.50 * self._passenger_ratio(request))

    def _wait_action_penalty(self, vehicle):
        return -(
            0.02
            + 0.03 * self._load_ratio(vehicle.num_passengers)
        )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------
    def reset(self):
        self.curr_time = 0
        self.curr_step = 0

        self.future_request_list = copy.deepcopy(self.original_request_list)
        self.active_request_list = []
        self.done_request_list = []

        self.vehicle_list = []
        self.initialize_vehicles()
        self.handle_time_update(count_idle=False)
        return None

    def initialize_vehicles(self):
        for idx in range(cfg.MAX_NUM_VEHICLES):
            pos = self.vehicle_init_pos[idx]
            veh = Vehicle(idx, pos, self.network)
            self.vehicle_list.append(veh)

    # -----------------------------------------------------------------------
    # Pretty-print helpers (디버깅 보조)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Time update — 미래 요청 유입 / 차량 상태 진행 / 취소 처리
    # -----------------------------------------------------------------------
    def handle_time_update(self, count_idle=True):
        d_reward_list = []

        # 새 요청 유입
        while self.future_request_list and self.future_request_list[0].request_time <= self.curr_time:
            r = self.future_request_list.pop(0)
            self.active_request_list.append(r)

        # Vehicle 진행
        for v in self.vehicle_list:
            if v.status == VehicleStatus.REJECT:
                v.status = VehicleStatus.IDLE
            elif count_idle and v.status == VehicleStatus.IDLE:
                v.idle_time += 1

            if v.status == VehicleStatus.PICKUP and v.target_arrival_time == self.curr_time:
                r = v.target_request

                v.status = VehicleStatus.IDLE
                v.curr_node = v.next_node
                v.next_node = 0
                v.target_request = None
                v.target_arrival_time = -1

                if r.status == RequestStatus.CANCELLED:
                    v.active_request_list.remove(r)
                    p_action_id = "{}_{}".format(r.id, ActionType.PICKUP.value)
                    d_reward_list.append([p_action_id, self._cancel_penalty(r)])
                else:
                    v.num_passengers += r.num_passengers
                    assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"
                    r.status = RequestStatus.PICKEDUP
                    r.waiting_time = self.curr_time - r.request_time
                    r.pickup_at = self.curr_time
                    p_action_id_pick = "{}_{}".format(r.id, ActionType.PICKUP.value)
                    wp = self._penalty_time_over_cap(r.waiting_time, cfg.MAX_WAIT_TIME)
                    if wp < 0:
                        d_reward_list.append([p_action_id_pick, wp])

            if v.status == VehicleStatus.DROPOFF and v.target_arrival_time == self.curr_time:
                r = v.target_request

                v.status = VehicleStatus.IDLE
                v.curr_node = v.next_node
                v.next_node = 0
                v.target_request = None
                v.target_arrival_time = -1

                r.status = RequestStatus.SERVED
                r.arrival_due_left = max(0, r.arrival_due - self.curr_time)
                r.in_vehicle_time = self.curr_time - r.pickup_at
                r.dropoff_at = self.curr_time
                service_reward = self._service_completion_reward(v, r)

                v.active_request_list.remove(r)
                v.num_passengers -= r.num_passengers
                assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"

                self.active_request_list.remove(r)
                self.done_request_list.append(r)

                p_action_id = "{}_{}".format(r.id, ActionType.PICKUP.value)
                d_action_id = "{}_{}".format(r.id, ActionType.DROPOFF.value)
                d_reward_list.append([p_action_id, 0.40 * service_reward])
                d_reward_list.append([d_action_id, 0.60 * service_reward])
                detour = max(0.0, float(r.in_vehicle_time - r.travel_time))
                dp = self._penalty_time_over_cap(detour, cfg.MAX_INVEHICLE_TIME)
                if dp < 0:
                    d_reward_list.append([p_action_id, dp])

                v.num_serve += 1

        # 활성 요청 시간 갱신 / 취소 처리
        cancelled_list = []
        for r in self.active_request_list:
            r.arrival_due_left = max(0, r.arrival_due - self.curr_time)
            if r.status == RequestStatus.PENDING or r.status == RequestStatus.ACCEPTED:
                r.waiting_time = self.curr_time - r.request_time
                if r.waiting_time >= cfg.MAX_WAIT_TIME:
                    r.status = RequestStatus.CANCELLED
                    cancelled_list.append(r)
            if r.status == RequestStatus.PICKEDUP:
                r.in_vehicle_time = self.curr_time - r.pickup_at

        for cr in cancelled_list:
            if cr.assigned_v_id >= 0:
                for v in self.vehicle_list:
                    if v.id == cr.assigned_v_id and v.status == VehicleStatus.PICKUP and v.target_request == cr:
                        v.status = VehicleStatus.IDLE
                        v.next_node = 0
                        v.target_request = None
                        v.target_arrival_time = -1
                        v.active_request_list.remove(cr)
                        p_action_id = "{}_{}".format(cr.id, ActionType.PICKUP.value)
                        d_reward_list.append([p_action_id, self._cancel_penalty(cr)])
                        break
            self.active_request_list.remove(cr)
            self.done_request_list.append(cr)

        for idx, r in enumerate(self.active_request_list):
            r.slot_idx = idx

        return d_reward_list

    # -----------------------------------------------------------------------
    # Snapshot (agent / replay 텐서 입력)
    # -----------------------------------------------------------------------
    def get_snapshot(self):
        """
        모델 입력용 스냅샷 dict.

        키:
            vehicle_static : (V, VEHICLE_RAW_DIM) float32
            vehicle_nodes  : (V, 2) int32  — (curr, next) 노드 ID, 0 = 센티넬
            request_static : (R, REQUEST_RAW_DIM) float32  (활성 요청 0이면 shape (0,D))
            request_nodes  : (R, 2) int32 — (from, to)
            time_norm      : float32 in [0,1] 대략적인 시뮬 시간 정규화
            global_stats   : (GLOBAL_STATS_DIM,) float32
            pair_agg       : (PAIR_AGG_DIM,) float32 — Option A 페어 MLP용 집계 스칼라
        """
        V = cfg.MAX_NUM_VEHICLES
        vehicle_static = np.zeros((V, cfg.VEHICLE_RAW_DIM), dtype=np.float32)
        vehicle_nodes = np.zeros((V, 2), dtype=np.int32)
        for i, v in enumerate(self.vehicle_list):
            if i >= V:
                break
            vehicle_static[i] = np.asarray(v.get_static_features(), dtype=np.float32)
            vehicle_nodes[i] = np.asarray(v.get_node_ids(), dtype=np.int32)

        R = len(self.active_request_list)
        if R > 0:
            request_static = np.zeros((R, cfg.REQUEST_RAW_DIM), dtype=np.float32)
            request_nodes = np.zeros((R, 2), dtype=np.int32)
            for j, r in enumerate(self.active_request_list):
                request_static[j] = np.asarray(r.get_static_features(), dtype=np.float32)
                request_nodes[j] = np.asarray(r.get_node_ids(), dtype=np.int32)
        else:
            request_static = np.zeros((0, cfg.REQUEST_RAW_DIM), dtype=np.float32)
            request_nodes = np.zeros((0, 2), dtype=np.int32)

        t_last = max((r.request_time for r in self.original_request_list), default=0)
        span = float(max(
            t_last + Request.ARRIVAL_TOLERANCE_TIME + max(self.network.max_duration, 1),
            1,
        ))
        time_norm = np.float32(min(1.0, float(self.curr_time) / span))

        max_r = float(max(len(self.original_request_list), 1))
        gstats = np.zeros(cfg.GLOBAL_STATS_DIM, dtype=np.float32)
        gstats[0] = np.float32(min(1.0, R / 32.0))
        gstats[1] = np.float32(min(1.0, len(self.future_request_list) / max_r))
        gstats[2] = np.float32(
            sum(1 for v in self.vehicle_list if v.status == VehicleStatus.IDLE) / cfg.MAX_NUM_VEHICLES
        )
        pend = [r for r in self.active_request_list if r.status == RequestStatus.PENDING]
        if pend:
            gstats[3] = np.float32(np.mean([r.waiting_time for r in pend]) / cfg.MAX_WAIT_TIME)
        cap_denom = max(cfg.VEH_CAPACITY * cfg.MAX_NUM_VEHICLES, 1)
        gstats[4] = np.float32(sum(v.num_passengers for v in self.vehicle_list) / cap_denom)
        gstats[5] = np.float32(
            sum(1 for v in self.vehicle_list if v.status == VehicleStatus.PICKUP) / cfg.MAX_NUM_VEHICLES
        )
        gstats[6] = np.float32(
            sum(1 for v in self.vehicle_list if v.status == VehicleStatus.DROPOFF) / cfg.MAX_NUM_VEHICLES
        )
        gstats[7] = np.float32(
            sum(1 for v in self.vehicle_list if v.status == VehicleStatus.REJECT) / cfg.MAX_NUM_VEHICLES
        )

        pair_agg = self._pair_aggregate_scalars(time_norm)

        return {
            'vehicle_static': vehicle_static,
            'vehicle_nodes': vehicle_nodes,
            'request_static': request_static,
            'request_nodes': request_nodes,
            'time_norm': time_norm,
            'global_stats': gstats,
            'pair_agg': pair_agg,
        }

    def _pair_aggregate_scalars(self, time_norm_scalar):
        """Option A: 페어 MLP 용 에피소드·큐 수준 스칼라 벡터 (공간 무관 집계)."""
        cap = float(max(getattr(cfg, 'PAIR_AGG_COUNT_NORM_CAP', 48), 1.0))
        duel_den = float(
            max(
                self.network.max_duration + Request.ARRIVAL_TOLERANCE_TIME,
                1.0,
            )
        )
        total_cap = max(cfg.VEH_CAPACITY * cfg.MAX_NUM_VEHICLES, 1)

        n_idle = sum(1 for v in self.vehicle_list if v.status == VehicleStatus.IDLE)
        pending = [r for r in self.active_request_list if r.status == RequestStatus.PENDING]
        picked = [r for r in self.active_request_list if r.status == RequestStatus.PICKEDUP]

        pend_n = np.float32(len(pending) / cap)
        pick_n = np.float32(len(picked) / cap)
        act_n = np.float32(len(self.active_request_list) / cap)

        if pending:
            waits = [float(r.waiting_time) for r in pending]
            mean_w = np.float32(np.mean(waits) / cfg.MAX_WAIT_TIME)
            max_w = np.float32(min(1.0, np.max(waits) / cfg.MAX_WAIT_TIME))
            dues = [float(max(0, r.arrival_due_left)) for r in pending]
            mean_due = np.float32(np.mean(dues) / duel_den)
        else:
            mean_w = max_w = mean_due = np.float32(0.0)

        idle_frac = np.float32(n_idle / cfg.MAX_NUM_VEHICLES)
        load_frac = np.float32(
            sum(v.num_passengers for v in self.vehicle_list) / total_cap
        )
        tot_req = len(self.original_request_list)
        fut_frac = np.float32(
            min(1.0, len(self.future_request_list) / max(tot_req, 1)),
        )

        prog = np.float32(min(1.0, float(time_norm_scalar)))

        out = np.array(
            [
                idle_frac,
                pend_n,
                pick_n,
                mean_w,
                max_w,
                mean_due,
                prog,
                load_frac,
                fut_frac,
                act_n,
            ],
            dtype=np.float32,
        )
        assert out.shape == (cfg.PAIR_AGG_DIM,), (
            'PAIR_AGG_DIM must match env._pair_aggregate_scalars'
        )
        return out

    # -----------------------------------------------------------------------
    # Pair candidate enumeration
    # -----------------------------------------------------------------------
    def _onboard_requests(self, vehicle):
        return [
            r for r in vehicle.active_request_list
            if r.status == RequestStatus.PICKEDUP
        ]

    def _request_travel_time(self, request):
        request_duration = getattr(request, 'travel_time', None)
        if request_duration is None or request_duration < 0:
            request_duration = self.network.get_duration(
                request.from_node_id, request.to_node_id
            )
        return float(request_duration)

    def _all_within_in_vehicle_limits(self, elapsed_by_request):
        for r, elapsed in elapsed_by_request.items():
            if float(elapsed) > self.in_vehicle_time_limit(r):
                return False
        return True

    def _has_feasible_dropoff_sequence(self, start_node, elapsed_by_request):
        """
        start_node에서 출발해 남은 탑승 요청들을 모두 각 request별
        travel_time + MAX_INVEHICLE_TIME 안에 하차시킬 순서가 있는지 검사한다.
        """
        if not elapsed_by_request:
            return True
        if not self._all_within_in_vehicle_limits(elapsed_by_request):
            return False

        requests = tuple(elapsed_by_request.keys())
        limits = {r: self.in_vehicle_time_limit(r) for r in requests}
        memo = {}

        def search(curr_node, remaining, elapsed_values):
            if not remaining:
                return True

            key = (
                curr_node,
                tuple(r.id for r in remaining),
                tuple(float(v) for v in elapsed_values),
            )
            if key in memo:
                return memo[key]

            drop_order = sorted(
                range(len(remaining)),
                key=lambda idx: (
                    limits[remaining[idx]]
                    - (
                        float(elapsed_values[idx])
                        + float(self.network.get_duration(
                            curr_node,
                            remaining[idx].to_node_id,
                        ))
                    )
                ),
            )
            for idx in drop_order:
                drop_r = remaining[idx]
                dur = self.network.get_duration(curr_node, drop_r.to_node_id)
                next_elapsed = tuple(float(v) + float(dur) for v in elapsed_values)
                if any(
                    next_elapsed[j] > limits[remaining[j]]
                    for j in range(len(remaining))
                ):
                    continue

                next_remaining = remaining[:idx] + remaining[idx + 1:]
                next_elapsed_remaining = (
                    next_elapsed[:idx] + next_elapsed[idx + 1:]
                )
                if search(drop_r.to_node_id, next_remaining, next_elapsed_remaining):
                    memo[key] = True
                    return True

            memo[key] = False
            return False

        elapsed_values = tuple(float(elapsed_by_request[r]) for r in requests)
        return search(start_node, requests, elapsed_values)

    def _can_serve_after_pickup(self, vehicle, request, pickup_dur):
        """
        vehicle이 request를 pickup하러 간 뒤, 현재 탑승 요청과 새 요청을 모두
        각 request별 travel_time + MAX_INVEHICLE_TIME 안에 하차시킬 수 있는지 검사한다.
        """
        onboard = [
            r for r in vehicle.active_request_list
            if r.status == RequestStatus.PICKEDUP
        ]
        if not onboard:
            direct_dur = self.network.get_duration(
                request.from_node_id,
                request.to_node_id,
            )
            return direct_dur <= self.in_vehicle_time_limit(request)

        initial_elapsed = {
            r: float(r.in_vehicle_time) + float(pickup_dur)
            for r in onboard
        }
        initial_elapsed[request] = 0.0
        return self._has_feasible_dropoff_sequence(
            request.from_node_id,
            initial_elapsed,
        )

    def _can_dropoff_next(self, vehicle, request, dropoff_dur):
        onboard = self._onboard_requests(vehicle)
        elapsed = {
            r: float(r.in_vehicle_time) + float(dropoff_dur)
            for r in onboard
        }
        if not self._all_within_in_vehicle_limits(elapsed):
            return False
        elapsed.pop(request, None)
        return self._has_feasible_dropoff_sequence(
            request.to_node_id,
            elapsed,
        )

    def _can_wait_with_onboard_limits(self, vehicle, wait_time=1.0):
        elapsed = {
            r: float(r.in_vehicle_time) + float(wait_time)
            for r in self._onboard_requests(vehicle)
        }
        return self._has_feasible_dropoff_sequence(
            vehicle.curr_node,
            elapsed,
        )

    def enumerate_pair_candidates(self, idle_vehicles, include_wait=True):
        """
        idle 차량들에 대해 가능한 (v, r) 페어 후보와 선택적 wait 페어를 생성.

        ActionType.REJECT는 학습/배정 인터페이스 호환을 위해 남겨 두지만,
        여기서는 "이 차량은 이번 의사결정 tick에 대기한다"는 no-op 의미다.

        Returns:
            dict[v.id -> List[candidate dict]]
                candidate dict 키:
                    'r'              : Request | None (reject은 None)
                    'r_slot_idx'     : int | None  (active_request_list 내 인덱스, reject은 None)
                    'action_type'    : ActionType
                    'is_reject'      : 0 | 1
                    'is_real'        : 1 if pickup/dropoff, 0 if wait
                    'v_feat'         : np.ndarray (Dv,)
                    'r_feat'         : np.ndarray (Dr,)  reject은 zero vector
                    'rel_feat'       : np.ndarray (Drel,)  reject은 zero vector
        """
        result = {}
        zero_r = np.zeros(cfg.REQUEST_RAW_DIM, dtype=np.float32)
        zero_rel = np.zeros(cfg.RELATION_INPUT_DIM, dtype=np.float32)
        max_dur = self.network.max_duration

        for v in idle_vehicles:
            v_feat = np.array(v.get_static_features(), dtype=np.float32)
            cands = []
            fallback_dropoff_cands = []

            for slot_idx, r in enumerate(self.active_request_list):
                if r.status == RequestStatus.PENDING:
                    # PICKUP 후보: 좌석/대기/승차시간 제약을 모두 만족해야 한다.
                    v_empty = cfg.VEH_CAPACITY - v.num_passengers
                    if v_empty < r.num_passengers:
                        continue
                    pickup_dur = self.network.get_duration(v.curr_node, r.from_node_id)
                    if r.waiting_time + pickup_dur >= cfg.MAX_WAIT_TIME:
                        continue
                    if not self._can_serve_after_pickup(v, r, pickup_dur):
                        continue
                    rel_feat = np.array(
                        [0.0, pickup_dur / max_dur if max_dur > 0 else 0.0],
                        dtype=np.float32,
                    )
                    cands.append({
                        'v_idx': v.id,
                        'r': r,
                        'r_slot_idx': slot_idx,
                        'action_type': ActionType.PICKUP,
                        'is_reject': 0,
                        'is_real': 1,
                        'v_feat': v_feat,
                        'r_feat': np.array(r.get_static_features(), dtype=np.float32),
                        'rel_feat': rel_feat,
                    })
                elif r.status == RequestStatus.PICKEDUP and r.assigned_v_id == v.id:
                    # DROPOFF 후보: 자기 차량에 실린 요청만
                    dropoff_dur = self.network.get_duration(v.curr_node, r.to_node_id)
                    rel_feat = np.array(
                        [1.0, dropoff_dur / max_dur if max_dur > 0 else 0.0],
                        dtype=np.float32,
                    )
                    cand = {
                        'v_idx': v.id,
                        'r': r,
                        'r_slot_idx': slot_idx,
                        'action_type': ActionType.DROPOFF,
                        'is_reject': 0,
                        'is_real': 1,
                        'v_feat': v_feat,
                        'r_feat': np.array(r.get_static_features(), dtype=np.float32),
                        'rel_feat': rel_feat,
                    }
                    if self._can_dropoff_next(v, r, dropoff_dur):
                        cands.append(cand)
                    else:
                        fallback_dropoff_cands.append(cand)

            has_real = any(c.get('is_real', 0) for c in cands)
            if not has_real and fallback_dropoff_cands:
                cands.extend(fallback_dropoff_cands)

            if include_wait and self._can_wait_with_onboard_limits(v):
                # WAIT 페어: ActionType.REJECT 값을 쓰되 의미는 no-op 차량 대기다.
                # r_slot_idx 는 reject/null request gather 용 placeholder 0.
                cands.append({
                    'v_idx': v.id,
                    'r': None,
                    'r_slot_idx': 0,
                    'action_type': ActionType.REJECT,
                    'is_reject': 1,
                    'is_real': 0,
                    'v_feat': v_feat,
                    'r_feat': zero_r,
                    'rel_feat': zero_rel,
                })
            result[v.id] = cands
        return result

    def has_dispatch_candidate(self):
        """현재 idle 차량 중 실제 pickup/dropoff 의사결정 후보가 있는지 확인."""
        idle_vehicles = [v for v in self.vehicle_list if v.status == VehicleStatus.IDLE]
        if not idle_vehicles:
            return False
        candidates_by_v = self.enumerate_pair_candidates(
            idle_vehicles, include_wait=False
        )
        return any(candidates_by_v[v.id] for v in idle_vehicles)

    def flatten_pair_candidates(self, idle_vehicles):
        """모든 idle 차량의 후보를 (v_feat, r_feat, rel_feat, is_reject) 튜플 리스트로 평탄화.
        replay buffer의 next_pairs 저장용."""
        if not idle_vehicles:
            return []
        candidates_by_v = self.enumerate_pair_candidates(idle_vehicles)
        flat = []
        for v in idle_vehicles:
            for c in candidates_by_v[v.id]:
                flat.append((c['v_feat'], c['r_feat'], c['rel_feat'], c['is_reject']))
        return flat

    # -----------------------------------------------------------------------
    # Step (action 처리)
    # -----------------------------------------------------------------------
    def step(self, action):
        """
        action: dict
            'vehicle_idx'  : int  (= vehicle.id, 0..N-1)
            'action_type'  : ActionType
            'request'      : Request | None
            (그 외 키는 agent 측 메타데이터, env는 사용 안 함)

        Returns:
            (reward, info)
                info 키:
                    'is_pending', 'has_delayed_reward', 'action_id_list', 'reward'
        """
        vehicle_idx = action['vehicle_idx']
        atype = action['action_type']
        r = action['request']

        v = self.vehicle_list[vehicle_idx]

        reward = 0.0
        info = {
            'is_pending': False,
            'has_delayed_reward': False,
            'action_id_list': None,
            'reward': None,
        }

        if atype == ActionType.REJECT:
            v.status = VehicleStatus.REJECT
            v.idle_time += 1
            reward = self._wait_action_penalty(v)

        elif atype == ActionType.PICKUP:
            assert r is not None and r.status == RequestStatus.PENDING, "Invalid PICKUP target"
            info['is_pending'] = True
            v.status = VehicleStatus.PICKUP
            v.active_request_list.append(r)
            v.next_node = r.from_node_id
            v.target_request = r
            pickup_duration = self.network.get_duration(v.curr_node, v.next_node)
            v.target_arrival_time = self.curr_time + pickup_duration

            r.status = RequestStatus.ACCEPTED
            r.assigned_v_id = v.id
            r.accepted_at = self.curr_time

            reward = self._pickup_action_reward(v, r, pickup_duration)
            v.num_accept += 1

            # 픽업이 즉시 완료되는 경우 (curr_node == from_node_id)
            if v.curr_node == v.next_node:
                v.status = VehicleStatus.IDLE
                v.next_node = 0
                v.target_request = None
                v.target_arrival_time = -1
                v.num_passengers += r.num_passengers
                assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"
                r.status = RequestStatus.PICKEDUP
                r.waiting_time = self.curr_time - r.request_time
                r.pickup_at = self.curr_time
                reward += 0.25 * self._load_ratio(v.num_passengers)
                reward += self._penalty_time_over_cap(r.waiting_time, cfg.MAX_WAIT_TIME)

        elif atype == ActionType.DROPOFF:
            assert r is not None and r in v.active_request_list, "Invalid DROPOFF target"
            info['is_pending'] = True
            v.status = VehicleStatus.DROPOFF
            v.next_node = r.to_node_id
            v.target_request = r
            dropoff_duration = self.network.get_duration(v.curr_node, v.next_node)
            v.target_arrival_time = self.curr_time + dropoff_duration

            reward = self._dropoff_action_reward(v, r, dropoff_duration)

            # 즉시 dropoff 완료되는 경우
            if v.curr_node == v.next_node:
                v.status = VehicleStatus.IDLE
                v.next_node = 0
                v.target_request = None
                v.target_arrival_time = -1

                r.status = RequestStatus.SERVED
                r.arrival_due_left = max(0, r.arrival_due - self.curr_time)
                r.in_vehicle_time = self.curr_time - r.pickup_at
                r.dropoff_at = self.curr_time
                service_reward = self._service_completion_reward(v, r)

                v.active_request_list.remove(r)
                v.num_passengers -= r.num_passengers
                assert 0 <= v.num_passengers <= cfg.VEH_CAPACITY, "Invalid Capacity"

                self.active_request_list.remove(r)
                self.done_request_list.append(r)

                reward += 0.60 * service_reward
                info['is_pending'] = False
                info['has_delayed_reward'] = True
                info['action_id_list'] = ["{}_{}".format(r.id, ActionType.PICKUP.value)]
                detour = max(0.0, float(r.in_vehicle_time - r.travel_time))
                info['reward'] = (
                    0.40 * service_reward
                    + self._penalty_time_over_cap(detour, cfg.MAX_INVEHICLE_TIME)
                )

                v.num_serve += 1

        else:
            raise ValueError(f"Unknown action_type: {atype}")

        # active_request_list가 변할 수 있으니 슬롯 인덱스를 다시 부여
        for idx, ar in enumerate(self.active_request_list):
            ar.slot_idx = idx

        self.curr_step += 1
        return reward, info

    # -----------------------------------------------------------------------
    # 보조 질의
    # -----------------------------------------------------------------------
    def has_idle_vehicle(self):
        return any(v.status == VehicleStatus.IDLE for v in self.vehicle_list)

    def in_vehicle_time_limit(self, request):
        return self._request_travel_time(request) + float(cfg.MAX_INVEHICLE_TIME)

    def find_dropoff_time_violation(self):
        requests = list(self.active_request_list) + list(self.done_request_list)
        for r in requests:
            if r.status not in (RequestStatus.PICKEDUP, RequestStatus.SERVED):
                continue
            if r.in_vehicle_time is None:
                continue

            request_duration = self._request_travel_time(r)
            limit = request_duration + float(cfg.MAX_INVEHICLE_TIME)
            if float(r.in_vehicle_time) > limit:
                return {
                    'request_id': r.id,
                    'status': str(r.status),
                    'request_duration': request_duration,
                    'in_vehicle_time': float(r.in_vehicle_time),
                    'limit': limit,
                    'max_invehicle_time': cfg.MAX_INVEHICLE_TIME,
                    'current_time': self.curr_time,
                }
        return None

    def find_in_vehicle_time_violation(self):
        return self.find_dropoff_time_violation()

    def is_done(self):
        return len(self.active_request_list) == 0 and len(self.future_request_list) == 0
