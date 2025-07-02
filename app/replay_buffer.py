import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.last_transition = None  # 항상 최신 transition을 별도로 관리

    def append(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        # transition의 id가 더 크면 갱신
        if (
                self.last_transition is None or
                transition[-1]['id'] > self.last_transition[-1]['id']
        ):
            self.last_transition = transition

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def get_last(self):
        return self.last_transition

    def __len__(self):
        return len(self.buffer)
    