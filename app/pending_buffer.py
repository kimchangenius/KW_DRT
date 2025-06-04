class PendingBuffer:
    def __init__(self):
        self.pending = {}

    def add(self, action_id, transition):
        self.pending[action_id] = transition

    def confirm(self, action_id, reward):
        transition = self.pending.pop(action_id, None)
        if transition is not None:
            transition[2] += reward
        return transition

    def cancel(self, action_id):
        self.pending.pop(action_id, None)  # 또는 보상 -1 부여

    def __len__(self):
        return len(self.pending)
