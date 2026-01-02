import itertools

MAX_NUM_VEHICLES = 4

MAX_NUM_REQUEST = 12
VEH_CAPACITY = 5
MAX_WAIT_TIME = 10
MAX_INVEHICLE_TIME = 10
# MAX_ACCEPT_WAIT = 6  # ACCEPTED 상태 후 픽업까지 허용하는 최대 대기시간(강화된 타임아웃)

# 세이프가드 토글/파라미터
ENABLE_SAFEGUARD = True
STAGNATION_WINDOW = 50
MAX_STEPS_CAP = 500

VEHICLE_INPUT_DIM = 53
REQUEST_INPUT_DIM = 55
RELATION_INPUT_DIM = 2

POSSIBLE_ACTION = MAX_NUM_REQUEST + 1

# LLM 관련 설정
LLM_ENABLED = False
LLM_MODEL = "gemini-1.5-flash"
LLM_STEP_INTERVAL = 50

GEMINI_API_KEY = "AIzaSyBjFR4BMvW7gzguUMaKhB9Ch7TGkrOPHOw"


param_grid = {
    # "hidden_dim": [64, 128, 256],
    "hidden_dim": [256],
    # "batch_size": [32, 64, 128]
    "batch_size": [64],
    "ppo_batch_size": [128],
    # "learning_rate": [1e-4, 1e-5, 1e-6]
    "learning_rate": [1e-5],  # 기본값 
    "critic_learning_rate": [3e-4],
    "actor_learning_rate": [5e-4],
    "dqn_learning_rate": [1e-4],  # 더 공격적인 learning rate
    "mappo_learning_rate": [3e-4]  # MAPPO learning rate 추가
}

keys = list(param_grid.keys())
values = list(param_grid.values())

config_list = [
    dict(zip(keys, combination))
    for combination in itertools.product(*values)
]