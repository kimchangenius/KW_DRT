class PendingBuffer:
    """
    Delayed reward 처리 버퍼. action_id를 키로 transition을 보관하다가,
    환경에서 픽업/드롭오프가 확정될 때 reward를 누적한 뒤 replay buffer로 옮긴다.

    transition은 dict 형태이며 'reward' 키를 in-place로 갱신한다.
    """

    def __init__(self):
        self.pending = {}

    def add(self, action_id, transition):
        self.pending[action_id] = transition

    def confirm(self, action_id, reward):
        transition = self.pending.pop(action_id, None)
        if transition is not None:
            transition['reward'] += reward
        return transition

    def cancel(self, action_id):
        self.pending.pop(action_id, None)

    def clear(self):
        self.pending.clear()

    def __len__(self):
        return len(self.pending)
