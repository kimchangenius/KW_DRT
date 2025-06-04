class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def append(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return None

    def get_last(self):
        if len(self.buffer) == 0:
            return None
        return self.buffer[(self.position - 1) % len(self.buffer)]

    def __len__(self):
        return len(self.buffer)
    