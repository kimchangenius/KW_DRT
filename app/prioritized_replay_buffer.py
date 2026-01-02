import numpy as np
import random


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2016)
    
    주요 특징:
    - TD-error가 큰 경험을 더 자주 샘플링
    - Importance Sampling Weights로 bias 보정
    - 메모리 효율성: 중요한 경험 위주로 학습
    """
    
    def __init__(self, capacity=500, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """
        Args:
            capacity: 버퍼 크기
            alpha: 우선순위 강도 (0=uniform, 1=full prioritization)
            beta: Importance Sampling 보정 강도 (0=no correction, 1=full correction)
            beta_increment: 학습 진행에 따른 beta 증가량
            epsilon: TD-error가 0일 때를 위한 작은 값
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.last_transition = None

    def append(self, transition):
        """새로운 경험 추가 (최대 우선순위 부여)"""
        priority = self.max_priority
        self.tree.add(priority, transition)
        
        # 최신 transition 추적
        if self.last_transition is None or transition[-1]['id'] > self.last_transition[-1]['id']:
            self.last_transition = transition

    def sample(self, batch_size):
        """우선순위 기반 샘플링"""
        if self.tree.n_entries < batch_size:
            return None
        total = self.tree.total()
        if total <= 0:
            return None
        batch = []
        idxs = []
        priorities = []
        segment = total / batch_size

        # NOTE: beta 증가는 외부에서 에피소드 기준으로 설정합니다.

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Importance Sampling Weights 계산
        sampling_probabilities = np.array(priorities) / total
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # 정규화

        return batch, idxs, is_weights

    def set_beta(self, beta):
        """외부 진행도(에피소드)에 따라 beta를 직접 설정"""
        try:
            beta_f = float(beta)
        except Exception:
            beta_f = 1.0
        self.beta = max(0.0, min(1.0, beta_f))

    def update_priorities(self, idxs, td_errors):
        """TD-error 기반 우선순위 업데이트"""
        for idx, td_error in zip(idxs, td_errors):
            # TD-error의 절대값을 우선순위로 사용
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            # overflow/underflow 방지
            priority = np.clip(priority, 1e-8, 1e6)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def get_last(self):
        """마지막 transition 반환"""
        return self.last_transition

    def __len__(self):
        return self.tree.n_entries

