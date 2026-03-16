import itertools

MAX_NUM_VEHICLES = 4

MAX_NUM_REQUEST = 8
VEH_CAPACITY = 5
MAX_WAIT_TIME = 10
MAX_INVEHICLE_TIME = 10

VEHICLE_INPUT_DIM = 53
REQUEST_INPUT_DIM = 55
RELATION_INPUT_DIM = 2

POSSIBLE_ACTION = MAX_NUM_REQUEST + 1

param_grid = {
    # "hidden_dim": [64, 128, 256],
    "hidden_dim": [256],
    # "mini_batch_size": [32, 64, 128],
    "mini_batch_size": [64],
    # pi_lr: Actor(Policy) 학습률 (표준 3e-4)
    "pi_lr": [3e-4],
    # vf_lr: Critic(Value) 학습률 (표준 1e-3)
    "vf_lr": [1e-3]
}

keys = list(param_grid.keys())
values = list(param_grid.values())

config_list = [
    dict(zip(keys, combination))
    for combination in itertools.product(*values)
]
