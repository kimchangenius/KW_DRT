# DDQN 성능 향상 방법들 상세 분석

## 목차
1. [보상 함수 개선 (+20-30%) ⭐⭐](#1-보상-함수-개선)
2. [하이퍼파라미터 최적화 (+15-25%) ⭐](#2-하이퍼파라미터-최적화)
3. [네트워크 구조 개선 (+10-20%) ⭐⭐⭐](#3-네트워크-구조-개선)
4. [멀티스텝 학습 (+15-25%) ⭐⭐⭐](#4-멀티스텝-학습)
5. [Prioritized Replay (+25-40%) ⭐⭐⭐⭐](#5-prioritized-replay)
6. [앙상블 학습 (+20-35%) ⭐⭐⭐⭐⭐](#6-앙상블-학습)

---

## 1. 보상 함수 개선 (+20-30%) ⭐⭐

### 현재 보상 함수의 문제점

```python
# 현재 구현 (env.py)
# Pickup 결정 보상
reward = 0.5 * (1 - pickup_duration / self.network.max_duration) \
         + 0.5 * (1 - r.waiting_time / cfg.MAX_WAIT_TIME)

# Dropoff 결정 보상  
reward = 0.5 * (1 - dropoff_duration / self.network.max_duration) \
         + 0.5 * (1 - in_vehicle_ratio)
```

**문제점들**:
- 🎯 **단순한 거리 기반**: 거리만 고려하여 복잡한 상황 무시
- ⏰ **시간 요소 부족**: 대기시간, 차량내시간, 우회시간의 복합적 고려 부족
- 🚗 **차량 효율성 무시**: 승객 수, 수용량 활용도 미고려
- 📍 **지역별 특성 무시**: 노드별 수요 패턴, 교통 상황 미반영

### 개선된 보상 함수 설계

#### 1. 다차원 보상 함수

```python
def calculate_improved_pickup_reward(self, v, r, pickup_duration):
    """개선된 Pickup 결정 보상 함수"""
    
    # 1. 거리 보상 (0.3): 픽업 거리가 짧을수록 높음
    distance_reward = 0.3 * (1 - pickup_duration / self.network.max_duration)
    
    # 2. 대기시간 보상 (0.25): 요청의 대기 시간이 짧을수록 높음
    waiting_reward = 0.25 * (1 - r.waiting_time / cfg.MAX_WAIT_TIME)
    
    # 3. 효율성 보상 (0.2): 차량의 현재 승객 수 대비 수용량 여유
    empty_seats_ratio = (cfg.VEH_CAPACITY - v.num_passengers) / cfg.VEH_CAPACITY
    efficiency_reward = 0.2 * empty_seats_ratio
    
    # 4. 지역 수요 보상 (0.15): 픽업 지역의 수요 밀도
    demand_density = self.get_demand_density(r.from_node_id)
    demand_reward = 0.15 * demand_density
    
    # 5. 시간대 보상 (0.1): 현재 시간대의 수요 패턴
    time_pattern = self.get_time_pattern_reward(self.curr_time)
    time_reward = 0.1 * time_pattern
    
    total_reward = distance_reward + waiting_reward + efficiency_reward + demand_reward + time_reward
    return total_reward

def calculate_improved_dropoff_reward(self, v, r, dropoff_duration):
    """개선된 Dropoff 결정 보상 함수"""
    
    # 1. 거리 보상 (0.3)
    distance_reward = 0.3 * (1 - dropoff_duration / self.network.max_duration)
    
    # 2. 차량 내 시간 보상 (0.25)
    in_vehicle_ratio = min(r.in_vehicle_time / cfg.MAX_INVEHICLE_TIME, 1.0)
    in_vehicle_reward = 0.25 * (1 - in_vehicle_ratio)
    
    # 3. 승객 수 보상 (0.2): 효율적인 경로
    passenger_ratio = v.num_passengers / cfg.VEH_CAPACITY
    passenger_reward = 0.2 * passenger_ratio
    
    # 4. 경로 효율성 보상 (0.15): 다른 승객들과의 경로 겹침
    route_efficiency = self.calculate_route_efficiency(v, r)
    route_reward = 0.15 * route_efficiency
    
    # 5. 목적지 수요 보상 (0.1)
    destination_demand = self.get_demand_density(r.to_node_id)
    destination_reward = 0.1 * destination_demand
    
    total_reward = distance_reward + in_vehicle_reward + passenger_reward + route_reward + destination_reward
    return total_reward
```

#### 2. 보조 함수들

```python
def get_demand_density(self, node_id):
    """특정 노드의 수요 밀도 계산"""
    # 현재 active_request_list에서 해당 노드 관련 요청 수 계산
    from_count = sum(1 for r in self.active_request_list if r.from_node_id == node_id)
    to_count = sum(1 for r in self.active_request_list if r.to_node_id == node_id)
    total_count = from_count + to_count
    
    # 정규화 (최대 요청 수로 나눔)
    max_demand = len(self.active_request_list) + 1
    density = total_count / max_demand
    return density

def calculate_route_efficiency(self, v, r):
    """경로 효율성 계산"""
    if len(v.active_request_list) == 0:
        return 1.0
    
    # 현재 차량이 가지고 있는 다른 요청들과의 경로 겹침 계산
    total_overlap = 0
    for other_r in v.active_request_list:
        if other_r.id != r.id:
            # 경로 겹침 정도 계산
            overlap = self.calculate_route_overlap(r, other_r)
            total_overlap += overlap
    
    # 정규화
    efficiency = total_overlap / max(len(v.active_request_list), 1)
    return efficiency

def get_time_pattern_reward(self, curr_time):
    """시간대별 수요 패턴 고려"""
    # 러시아워 시간대 정의 (예: 0-300, 600-900)
    rush_hours = [(0, 300), (600, 900)]
    
    for start, end in rush_hours:
        if start <= curr_time <= end:
            return 1.0  # 러시아워
    
    return 0.5  # 비러시아워
```

### 예상 효과
- **성능 향상**: 20-30% 보상 증가
- **서비스 품질**: 대기시간, 차량내시간 단축
- **차량 활용도**: 승객 수용 효율성 향상
- **지역 균형**: 수요 밀집 지역 우선 서비스

---

## 2. 하이퍼파라미터 최적화 (+15-25%) ⭐

### 현재 하이퍼파라미터 분석

```python
# 현재 설정 (dqn_agent.py)
self.epsilon = 1.0
self.epsilon_min = 0.01
self.epsilon_decay = 0.9995
self.update_target_freq = 200
self.batch_size = 32
self.learning_rate = 2e-4  # 적응형 스케줄링
self.gamma = 0.99
```

### 최적화 가능한 하이퍼파라미터들

#### 1. Discount Factor (γ) 조절

```python
# 현재: 고정된 gamma (0.99)
# 개선: 적응형 gamma
def adaptive_gamma(self, episode):
    """에피소드 진행에 따라 gamma 조절"""
    if episode < 20:
        return 0.95  # 초기에는 낮은 gamma (단기 보상 중시)
    elif episode < 50:
        return 0.97  # 중간에는 중간 gamma
    else:
        return 0.99  # 후반에는 높은 gamma (장기 보상 중시)
```

#### 2. Target Network 업데이트 전략

```python
# 현재: 고정된 주기 (200 steps)
# 개선 1: 적응형 업데이트 주기
def adaptive_target_update_freq(self, episode):
    if episode < 20:
        return 100  # 초기에는 자주 업데이트
    elif episode < 50:
        return 200  # 중간에는 보통
    else:
        return 500  # 후반에는 천천히

# 개선 2: Soft Update (Polyak averaging)
def soft_update_target_network(self, tau=0.001):
    """부드러운 타겟 네트워크 업데이트"""
    main_weights = self.main_network.get_weights()
    target_weights = self.target_network.get_weights()
    
    new_weights = []
    for main_w, target_w in zip(main_weights, target_weights):
        new_w = tau * main_w + (1 - tau) * target_w
        new_weights.append(new_w)
    
    self.target_network.set_weights(new_weights)
```

#### 3. Replay Buffer 크기 최적화

```python
# 현재: 고정된 버퍼 크기 (10000)
# 개선: 동적 버퍼 크기
def adaptive_buffer_size(self, episode):
    if episode < 10:
        return 5000  # 초기에는 작은 버퍼
    elif episode < 30:
        return 10000  # 중간에는 보통 버퍼
    else:
        return 20000  # 후반에는 큰 버퍼
```

#### 4. Batch Size 최적화

```python
# 현재: 고정된 batch_size (32)
# 개선: 적응형 batch size
def adaptive_batch_size(self, episode):
    if episode < 10:
        return 16  # 초기에는 작은 배치 (빠른 업데이트)
    elif episode < 30:
        return 32  # 중간에는 보통 배치
    else:
        return 64  # 후반에는 큰 배치 (안정적 학습)
```

### 예상 효과
- **학습 안정성**: 적응형 파라미터로 안정적 수렴
- **수렴 속도**: 초기 빠른 학습, 후반 정교한 조정
- **성능 향상**: 15-25% 보상 증가

---

## 3. 네트워크 구조 개선 (+10-20%) ⭐⭐⭐

### 현재 네트워크 구조

```python
# 현재 구현 (dqn_agent.py)
def _build_network(self):
    inputs = tf.keras.Input(shape=(self.state_dim,))
    x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
    x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
    outputs = tf.keras.layers.Dense(self.action_dim)(x)
    return tf.keras.Model(inputs, outputs)
```

**문제점들**:
- **깊이 부족**: 2개 은닉층으로 복잡한 패턴 학습 어려움
- **Gradient Vanishing**: 깊은 네트워크 학습 시 문제
- **Overfitting**: 정규화 기법 부재

### 개선된 네트워크 구조들

#### 1. Dueling DQN 구조

```python
def _build_dueling_network(self):
    """Dueling DQN: V(s)와 A(s,a)를 분리하여 학습"""
    inputs = tf.keras.Input(shape=(self.state_dim,))
    
    # 공통 특징 추출 레이어
    x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
    x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
    
    # Value Stream: V(s)
    value_stream = tf.keras.layers.Dense(self.hidden_dim // 2, activation='relu')(x)
    value = tf.keras.layers.Dense(1)(value_stream)
    
    # Advantage Stream: A(s,a)
    advantage_stream = tf.keras.layers.Dense(self.hidden_dim // 2, activation='relu')(x)
    advantage = tf.keras.layers.Dense(self.action_dim)(advantage_stream)
    
    # 결합: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    advantage_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
    q_values = tf.keras.layers.Add()([value, tf.keras.layers.Subtract()([advantage, advantage_mean])])
    
    return tf.keras.Model(inputs, q_values)
```

**장점**:
- **효율적 학습**: 상태 가치와 행동 우위를 분리하여 학습
- **안정성**: 더 안정적인 학습 곡선
- **성능**: 10-15% 성능 향상

#### 2. Noisy Network

```python
class NoisyLinear(tf.keras.layers.Layer):
    """Noisy Linear Layer for exploration"""
    def __init__(self, units, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.units = units
        self.sigma_init = sigma_init
    
    def build(self, input_shape):
        # Learnable parameters
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.w_sigma = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer=tf.constant_initializer(self.sigma_init),
                                        trainable=True)
        self.b_mu = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     trainable=True)
        self.b_sigma = self.add_weight(shape=(self.units,),
                                        initializer=tf.constant_initializer(self.sigma_init),
                                        trainable=True)
    
    def call(self, inputs, training=None):
        if training:
            # 노이즈 추가
            w_epsilon = tf.random.normal(shape=tf.shape(self.w_mu))
            b_epsilon = tf.random.normal(shape=tf.shape(self.b_mu))
            
            w = self.w_mu + self.w_sigma * w_epsilon
            b = self.b_mu + self.b_sigma * b_epsilon
        else:
            w = self.w_mu
            b = self.b_mu
        
        return tf.matmul(inputs, w) + b

def _build_noisy_network(self):
    """Noisy Network: Epsilon-greedy 없이 탐험"""
    inputs = tf.keras.Input(shape=(self.state_dim,))
    x = NoisyLinear(self.hidden_dim)(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = NoisyLinear(self.hidden_dim)(x)
    x = tf.keras.layers.Activation('relu')(x)
    outputs = NoisyLinear(self.action_dim)(x)
    return tf.keras.Model(inputs, outputs)
```

**장점**:
- **자동 탐험**: Epsilon-greedy 없이 자동으로 탐험
- **파라미터 노이즈**: 가중치에 노이즈 추가로 탐험
- **성능**: 5-10% 성능 향상

#### 3. ResNet 구조

```python
def _build_residual_network(self):
    """ResNet 구조로 깊은 네트워크 학습"""
    inputs = tf.keras.Input(shape=(self.state_dim,))
    
    # 첫 번째 레이어
    x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Residual 블록들
    for i in range(3):
        residual = x
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(self.hidden_dim)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation('relu')(x)
    
    outputs = tf.keras.layers.Dense(self.action_dim)(x)
    return tf.keras.Model(inputs, outputs)
```

**장점**:
- **깊은 네트워크**: Residual connection으로 깊은 네트워크 학습 가능
- **Gradient 안정성**: Gradient vanishing 문제 해결
- **정규화**: Batch Normalization, Dropout으로 overfitting 방지

### 예상 효과
- **복잡한 패턴 학습**: 10-20% 성능 향상
- **학습 안정성**: Gradient 문제 해결
- **일반화 능력**: Overfitting 방지

---

## 4. 멀티스텝 학습 (+15-25%) ⭐⭐⭐

### 개념

현재는 1-step TD 학습을 사용:
```
Q(s,a) = r + γ * max Q(s',a')
```

멀티스텝 학습은 n-step return을 사용:
```
G_t^(n) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * max Q(s_{t+n}, a)
```

### 구현

#### 1. N-Step Return 계산

```python
def calculate_n_step_return(self, transitions, n=3, gamma=0.99):
    """N-step return 계산"""
    if len(transitions) < n:
        n = len(transitions)
    
    # N-step return 계산
    n_step_return = 0
    for i in range(n):
        state, action, reward, next_state, done, info = transitions[i]
        n_step_return += (gamma ** i) * reward
        
        if done:
            return n_step_return
    
    # 마지막 상태의 Q값 추가
    last_state, _, _, _, _, _ = transitions[n-1]
    last_next_state = transitions[n-1][3]
    
    # Target Q값 계산
    target_q = self.target_network.predict(last_next_state, verbose=0)
    max_q = np.max(target_q[0])
    n_step_return += (gamma ** n) * max_q
    
    return n_step_return
```

#### 2. 멀티스텝 학습 통합

```python
def train_with_multistep(self, n_step=3):
    """멀티스텝 학습"""
    if self.replay_buffer.size() < self.batch_size:
        return None
    
    # 배치 샘플링
    batch = self.replay_buffer.sample(self.batch_size)
    
    states = []
    targets = []
    
    for i in range(len(batch)):
        state, action, reward, next_state, done, info = batch[i]
        
        # N-step return 계산
        if i + n_step <= len(batch):
            transitions = batch[i:i+n_step]
            n_step_return = self.calculate_n_step_return(transitions, n_step)
        else:
            # 1-step return으로 대체
            if done:
                n_step_return = reward
            else:
                target_q = self.target_network.predict(next_state, verbose=0)
                n_step_return = reward + self.gamma * np.max(target_q[0])
        
        # 현재 Q값
        current_q = self.main_network.predict(state, verbose=0)
        current_q[0][action[1]] = n_step_return
        
        states.append(state[0])
        targets.append(current_q[0])
    
    # 학습
    states = np.array(states)
    targets = np.array(targets)
    history = self.main_network.fit(states, targets, epochs=1, verbose=0)
    
    return history.history['loss'][0]
```

### 예상 효과
- **샘플 효율성**: 15-25% 성능 향상
- **보상 전파**: 빠른 보상 전파
- **학습 속도**: 빠른 수렴

---

## 5. Prioritized Experience Replay (+25-40%) ⭐⭐⭐⭐

### 개념

현재는 uniform random sampling:
- 모든 경험을 동일한 확률로 샘플링

Prioritized Replay는 중요한 경험을 우선적으로 샘플링:
- TD error가 큰 경험을 더 자주 샘플링
- 중요한 경험을 더 많이 학습

### 구현

#### 1. SumTree 자료구조

```python
class SumTree:
    """SumTree for efficient prioritized sampling"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """우선순위 변화를 트리에 전파"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """주어진 값에 해당하는 리프 노드 찾기"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """총 우선순위"""
        return self.tree[0]
    
    def add(self, priority, data):
        """새로운 경험 추가"""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """우선순위 업데이트"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """우선순위 기반 샘플링"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
```

#### 2. Prioritized Replay Buffer

```python
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta    # importance sampling 가중치
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.epsilon = 0.01  # small constant to avoid zero priority
    
    def add(self, transition):
        """새로운 경험 추가"""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size):
        """우선순위 기반 샘플링"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # Beta 증가 (importance sampling weight)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)
        
        # Importance sampling weights 계산
        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()
        
        return batch, idxs, is_weights
    
    def update_priorities(self, idxs, td_errors):
        """TD error 기반 우선순위 업데이트"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def size(self):
        return self.tree.n_entries
```

#### 3. DQN Agent 통합

```python
def train_with_prioritized_replay(self):
    """Prioritized Replay로 학습"""
    if self.replay_buffer.size() < self.batch_size:
        return None
    
    # Prioritized sampling
    batch, idxs, is_weights = self.replay_buffer.sample(self.batch_size)
    
    states = []
    targets = []
    td_errors = []
    
    for i in range(len(batch)):
        state, action, reward, next_state, done, info = batch[i]
        
        # 현재 Q값
        current_q = self.main_network.predict(state, verbose=0)[0]
        
        # 타겟 Q값
        if done:
            target_q = reward
        else:
            # Double DQN
            next_q_main = self.main_network.predict(next_state, verbose=0)[0]
            next_action = np.argmax(next_q_main)
            next_q_target = self.target_network.predict(next_state, verbose=0)[0]
            target_q = reward + self.gamma * next_q_target[next_action]
        
        # TD error 계산
        td_error = target_q - current_q[action[1]]
        td_errors.append(td_error)
        
        # 타겟 설정
        current_q[action[1]] = target_q
        
        states.append(state[0])
        targets.append(current_q)
    
    # Importance sampling weights 적용하여 학습
    states = np.array(states)
    targets = np.array(targets)
    is_weights = np.array(is_weights)
    
    history = self.main_network.fit(
        states, targets,
        sample_weight=is_weights,
        epochs=1, verbose=0
    )
    
    # 우선순위 업데이트
    self.replay_buffer.update_priorities(idxs, td_errors)
    
    return history.history['loss'][0]
```

### 예상 효과
- **샘플 효율성**: 25-40% 성능 향상
- **학습 속도**: 중요한 경험 집중 학습
- **안정성**: Importance sampling으로 bias 보정

---

## 6. 앙상블 학습 (+20-35%) ⭐⭐⭐⭐⭐

### 개념

여러 개의 독립적인 DDQN 모델을 학습하여 결합:
- 각 모델은 다른 초기화, 하이퍼파라미터로 학습
- 예측 시 모든 모델의 Q값을 평균 또는 투표

### 구현

#### 1. 앙상블 DQN Agent

```python
class EnsembleDQNAgent:
    """앙상블 DQN Agent"""
    def __init__(self, num_models=5, hidden_dim=256, batch_size=32, learning_rate=2e-4):
        self.num_models = num_models
        self.agents = []
        
        # 여러 개의 DQN Agent 생성
        for i in range(num_models):
            # 각 모델은 약간 다른 하이퍼파라미터 사용
            lr = learning_rate * np.random.uniform(0.8, 1.2)
            hd = hidden_dim + np.random.randint(-64, 64)
            agent = DQNAgent(hidden_dim=hd, batch_size=batch_size, learning_rate=lr)
            self.agents.append(agent)
    
    def act(self, state, action_mask):
        """앙상블 예측"""
        # 모든 모델의 Q값 예측
        all_q_values = []
        for agent in self.agents:
            q_values = agent.main_network.predict(state, verbose=0)[0]
            all_q_values.append(q_values)
        
        # Q값 평균
        avg_q_values = np.mean(all_q_values, axis=0)
        
        # Epsilon-greedy (앙상블 평균 epsilon 사용)
        avg_epsilon = np.mean([agent.epsilon for agent in self.agents])
        
        if np.random.rand() < avg_epsilon:
            # 탐험: 유효한 액션 중 랜덤 선택
            valid_actions = np.where(action_mask[0] == 1)[0]
            if len(valid_actions) > 0:
                action_idx = np.random.choice(valid_actions)
            else:
                action_idx = cfg.POSSIBLE_ACTION - 1
        else:
            # 활용: 최대 Q값 선택 (마스킹 적용)
            masked_q = np.where(action_mask[0] == 1, avg_q_values, -np.inf)
            action_idx = np.argmax(masked_q)
        
        vehicle_idx = 0  # DDQN은 single vehicle
        return [vehicle_idx, action_idx, {}]
    
    def remember(self, transition):
        """모든 모델에 경험 저장"""
        for agent in self.agents:
            agent.remember(transition)
    
    def pending(self, transition):
        """모든 모델에 pending 경험 저장"""
        for agent in self.agents:
            agent.pending(transition)
    
    def confirm_and_remember(self, action_id, delayed_reward):
        """모든 모델에 지연 보상 업데이트"""
        for agent in self.agents:
            agent.confirm_and_remember(action_id, delayed_reward)
    
    def train(self):
        """모든 모델 학습"""
        total_loss = 0
        count = 0
        
        for agent in self.agents:
            loss = agent.train()
            if loss is not None:
                total_loss += loss
                count += 1
        
        if count > 0:
            return total_loss / count
        return None
```

#### 2. 다양성 확보 전략

```python
def create_diverse_ensemble(num_models=5):
    """다양한 모델들로 앙상블 구성"""
    agents = []
    
    # 모델 1: 기본 DDQN
    agent1 = DQNAgent(hidden_dim=256, batch_size=32, learning_rate=2e-4)
    agents.append(agent1)
    
    # 모델 2: Dueling DQN
    agent2 = DuelingDQNAgent(hidden_dim=256, batch_size=32, learning_rate=1.5e-4)
    agents.append(agent2)
    
    # 모델 3: Noisy DQN
    agent3 = NoisyDQNAgent(hidden_dim=256, batch_size=32, learning_rate=2.5e-4)
    agents.append(agent3)
    
    # 모델 4: ResNet DQN
    agent4 = ResNetDQNAgent(hidden_dim=256, batch_size=64, learning_rate=1.8e-4)
    agents.append(agent4)
    
    # 모델 5: 큰 네트워크 DQN
    agent5 = DQNAgent(hidden_dim=512, batch_size=32, learning_rate=1e-4)
    agents.append(agent5)
    
    return agents
```

#### 3. 앙상블 예측 전략

```python
def ensemble_predict(agents, state, action_mask, strategy='average'):
    """다양한 앙상블 예측 전략"""
    
    # 모든 모델의 Q값 수집
    all_q_values = []
    for agent in agents:
        q_values = agent.main_network.predict(state, verbose=0)[0]
        all_q_values.append(q_values)
    
    all_q_values = np.array(all_q_values)
    
    if strategy == 'average':
        # 전략 1: 평균
        final_q = np.mean(all_q_values, axis=0)
    
    elif strategy == 'weighted_average':
        # 전략 2: 가중 평균 (최근 성능 기반)
        weights = []
        for agent in agents:
            if len(agent.recent_rewards) > 0:
                avg_reward = np.mean(agent.recent_rewards[-5:])
                weights.append(avg_reward)
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        final_q = np.average(all_q_values, axis=0, weights=weights)
    
    elif strategy == 'voting':
        # 전략 3: 투표
        votes = np.zeros(all_q_values.shape[1])
        for q_values in all_q_values:
            masked_q = np.where(action_mask[0] == 1, q_values, -np.inf)
            best_action = np.argmax(masked_q)
            votes[best_action] += 1
        final_q = votes
    
    elif strategy == 'max':
        # 전략 4: 최대값
        final_q = np.max(all_q_values, axis=0)
    
    elif strategy == 'ucb':
        # 전략 5: Upper Confidence Bound
        mean_q = np.mean(all_q_values, axis=0)
        std_q = np.std(all_q_values, axis=0)
        final_q = mean_q + 1.96 * std_q  # 95% confidence
    
    # 최종 액션 선택
    masked_q = np.where(action_mask[0] == 1, final_q, -np.inf)
    action_idx = np.argmax(masked_q)
    
    return action_idx
```

### 예상 효과
- **성능 향상**: 20-35% 보상 증가
- **안정성**: 여러 모델의 결합으로 안정적 예측
- **일반화**: 다양한 모델로 overfitting 방지
- **로버스트성**: 특정 모델의 실패에 강건

---

## 종합 비교

| 방법 | 성능 향상 | 구현 난이도 | 계산 비용 | 추천도 |
|------|-----------|-------------|-----------|--------|
| 보상 함수 개선 | +20-30% | ⭐⭐ (중간) | 낮음 | ⭐⭐ |
| 하이퍼파라미터 최적화 | +15-25% | ⭐ (쉬움) | 낮음 | ⭐ |
| 네트워크 구조 개선 | +10-20% | ⭐⭐⭐ (어려움) | 중간 | ⭐⭐⭐ |
| 멀티스텝 학습 | +15-25% | ⭐⭐⭐ (어려움) | 중간 | ⭐⭐⭐ |
| Prioritized Replay | +25-40% | ⭐⭐⭐⭐ (매우 어려움) | 높음 | ⭐⭐⭐⭐ |
| 앙상블 학습 | +20-35% | ⭐⭐⭐⭐⭐ (매우 어려움) | 매우 높음 | ⭐⭐⭐⭐⭐ |

## 구현 우선순위 추천

**단기 (빠른 효과)**:
1. 하이퍼파라미터 최적화 (⭐)
2. 보상 함수 개선 (⭐⭐)

**중기 (안정적 향상)**:
3. 네트워크 구조 개선 (⭐⭐⭐)
4. 멀티스텝 학습 (⭐⭐⭐)

**장기 (최고 성능)**:
5. Prioritized Replay (⭐⭐⭐⭐)
6. 앙상블 학습 (⭐⭐⭐⭐⭐)

## 결론

각 방법은 서로 독립적으로 적용 가능하며, 결합 시 상승 효과를 기대할 수 있습니다. 
프로젝트의 목표와 리소스에 따라 적절한 방법을 선택하시기 바랍니다.
