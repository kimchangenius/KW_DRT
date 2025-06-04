class PendingBuffer:
    def __init__(self):
        self.pending = {}

    def add(self, action_id, transition):
        self.pending[action_id] = transition

    def confirm(self, action_id, reward, next_state):
        # 보상이 확정되면 replay buffer로 넘김
        # transition = self.pending.pop(action_id)
        # full_transition = (transition.state, transition.action, reward, next_state, done)
        # return full_transition
        return None

    def cancel(self, action_id):
        self.pending.pop(action_id, None)  # 또는 보상 -1 부여