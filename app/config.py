import itertools

# ===========================================================================
# 시뮬레이션 / 환경
# ===========================================================================
MAX_NUM_VEHICLES = 4

VEH_CAPACITY = 5
MAX_WAIT_TIME = 10
MAX_INVEHICLE_TIME = 10

# waiting_time / detour_time 이 MAX_INVEHICLE_TIME 초과 시 (초과 step × scale) 페널티
EXCESS_TIME_PENALTY_SCALE = 0.1

# 노드 ID는 1..NUM_NODES 사용. 0은 "no node" 센티넬 (Vehicle.next_node 등).
NUM_NODES = 24

# ===========================================================================
# 모델 입력 차원
# ===========================================================================
# Vehicle / Request의 "노드 정보를 뺀" raw 특성 차원
#   Vehicle  : status one-hot(4) + capacity(1) = 5
#   Request  : status one-hot(3) + passengers(1) + travel(1) + waiting(1) + due_left(1) = 7
VEHICLE_RAW_DIM = 5
REQUEST_RAW_DIM = 7

# (v, r) 페어의 즉각 관계 정보: need_drop_off 플래그 + normalized pickup/dropoff duration
RELATION_INPUT_DIM = 2

# 글로벌 통계 차원 (env.get_snapshot이 채움)
GLOBAL_STATS_DIM = 8

# 페어 MLP에 추가 concat 되는 에피소드/큐 요약 스칼라(Option A).
# 미래 요청 비율(n_future/total)은 학습 시 시나리오가 알려져 있음 — 배포 환경에선 예측/고정값으로 대체해야 함.
PAIR_AGG_DIM = 10
PAIR_AGG_COUNT_NORM_CAP = 48.0

# 모델 내부 차원 (노드 임베딩 등)
NODE_EMB_DIM = 16

# ===========================================================================
# Hyperparameter grid
# batch_size: replay에서 한 번에 뽑는 transition 수 (effective batch)
# TRAIN_MICRO_BATCH_SIZE: GPU 역전파 시 한 번에 올리는 크기 (그래디언트 누적으로 effective batch 유지)
param_grid = {
    "hidden_dim": [128],
    "batch_size": [32],
    "learning_rate": [1e-4],
}

# gradient accumulation — VRAM 피크↓, 수학적으로 mean loss = sum_k (m_k/B)*mean_k 와 동일
TRAIN_MICRO_BATCH_SIZE = 8

keys = list(param_grid.keys())
values = list(param_grid.values())

config_list = [
    dict(zip(keys, combination))
    for combination in itertools.product(*values)
]
