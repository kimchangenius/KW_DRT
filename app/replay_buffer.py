import random


class ReplayBuffer:
    """
    Pair-wise transition을 저장하는 ring buffer.

    transition: dict
        'pair_x'       : (v_feat, r_feat, rel_feat, is_reject_int)
        'reward'       : float (delayed reward 누적될 수 있음 → mutable)
        'next_pairs'   : List[(v_feat, r_feat, rel_feat, is_reject_int)]
        'done'         : bool
        'meta'         : {'id': int, 'action_id': str|None}
    """

    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.last_transition = None  # 항상 최신(=id가 가장 큰) transition을 별도 추적

    def append(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

        last_id = (
            self.last_transition['meta']['id']
            if self.last_transition is not None
            else -1
        )
        if transition['meta']['id'] > last_id:
            self.last_transition = transition

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def get_last(self):
        return self.last_transition

    def __len__(self):
        return len(self.buffer)
