import os
import random
import numpy as np
import tensorflow as tf

from collections import deque
from app.action_type import ActionType
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.python.distribute.combinations import tf_function

class DQNAgent:
    def __init__(self, state_size, max_episodes=500):
        self.state_size = state_size

        self.num_reject = 1
        self.num_matching = 20
        self.num_rebalancing = 3
        self.param_dim_matching = 1
        self.param_dim_rebalancing = 1

        self.total_actions = (
            self.num_reject +
            self.num_matching +
            self.num_rebalancing
        )

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_delta = (self.epsilon - self.epsilon_min) / max_episodes
        self.batch_size = 32

        self.memory = deque(maxlen=20000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_update_counter = 0
        self.train_counter = 0
        self.target_update_freq = 1000

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, file_path):
        self.model.save_weights(file_path)
        print(f"Model weights saved at {file_path}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model.load_weights(file_path)
            self.update_target_model()
            print(f"Model weights loaded at {file_path}")
        else:
            print(f"No model weights loaded at{file_path}")

    def build_model(self):
        state = Input(shape=(self.state_size,), name='state')
        x = Dense(128, activation='relu')(state)
        x = Dense(128, activation='relu')(x)

        q_reject = Dense(self.num_reject, activation='linear', name='q_reject')(x)
        q_matching = Dense(self.num_matching, activation='linear', name='q_matching')(x)
        q_rebalance = Dense(self.num_rebalancing, activation='linear', name='q_rebalance')(x)

        q_all = Concatenate(name='q_all')(
            [q_reject, q_matching, q_rebalance]
        )

        model = Model(inputs=state, outputs=q_all)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=5e-5,
            clipnorm=0.5
        )
        model.compile(
            optimizer=optimizer,
            loss='huber'
        )
        return model

    @tf_function
    def predict_q_all(self, states: tf.Tensor) -> tf.Tensor:
        return self.model(states, training=False)

    @tf_function
    def predict_target_q_all(self, states: tf.Tensor) -> tf.Tensor:
        return self.target_model(states, training=False)

    def act(self, flat_states: np.ndarray, vehicles: list):
        batch = tf.convert_to_tensor(flat_states, dtype=tf.float32)
        q_all = self.predict_q_all(batch).numpy()

        n_v = flat_states.shape[0]
        req_flat_start = self.state_size - 20 * 3
        requests_flat = flat_states[:, req_flat_start:]
        request_times = requests_flat.reshape(n_v, 20, 3)[:, :, 2]

        for i in range(n_v):
            for m in range(self.num_matching):
                if request_times[i, m] < 0:
                    q_all[i, 1 + m] = -np.inf

        for i, v in enumerate(vehicles):
            if v.status != 'idle':
                q_all[i, :] = -np.inf

        idle_idxs = [i for i, v in enumerate(vehicles) if v.status == 'idle']
        valid_pairs = []
        for i in idle_idxs:
            for j, q in enumerate(q_all[i]):
                if not np.isneginf(q):
                    valid_pairs.append((i, j))

        # ε-greedy 탐색 or 활용
        if np.random.rand() < self.epsilon and valid_pairs:
            veh_i, idx = random.choice(valid_pairs)
        else:
            flat_q = q_all.reshape(-1)
            if np.all(np.isneginf(flat_q)):
                veh_i = random.choice(idle_idxs) if idle_idxs else 0
                idx = 0
            else:
                best_flat = int(np.argmax(flat_q))
                veh_i = best_flat // self.total_actions
                idx = best_flat % self.total_actions

        # action dict 생성
        if idx < self.num_reject:
            act = {
                'action_type': ActionType.REJECT,
                'discrete_index': 0,
                'parameter': None
            }
        elif idx < self.num_reject + self.num_matching:
            act = {
                'action_type': ActionType.MATCHING,
                'discrete_index': idx - self.num_reject,
                'parameter': None
            }
        else:
            act = {
                'action_type': ActionType.REBALANCING,
                'discrete_index': idx - self.num_reject - self.num_matching,
                'parameter': None
            }

        return veh_i, act

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        self.train_counter += 1
        # 매 5스텝마다, 메모리가 충분할 때만 학습
        if self.train_counter % 5 != 0 or len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        losses = []

        for state, action_list, reward, next_state, done in batch:
            # 0) 텐서 준비 (batch size=1)
            st_t = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            nxt_t = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)

            # 1) online 네트워크로부터 next Q 선택용
            q_next_on = self.predict_q_all(nxt_t).numpy().reshape(-1)  # shape=(A,)
            a_max = int(np.argmax(q_next_on))

            # 2) target 네트워크로부터 next Q 평가용
            q_next_tg = self.predict_target_q_all(nxt_t).numpy().reshape(-1)
            max_q_next = float(q_next_tg[a_max])

            # 3) 벨만 타깃값
            target_q_value = reward if done else reward + self.gamma * max_q_next

            # 4) 현재 상태에서 Q값
            q_current = self.predict_q_all(st_t).numpy().reshape(-1)

            # 5) 실제 취한 action index로 Q 업데이트
            for act in action_list:
                if act['action_type'] == ActionType.REJECT:
                    idx = 0
                elif act['action_type'] == ActionType.MATCHING:
                    idx = act['discrete_index'] + self.num_reject
                else:  # REBALANCING
                    idx = act['discrete_index'] + self.num_reject + self.num_matching
                q_current[idx] = target_q_value

            loss = self.model.train_on_batch(
                st_t,
                q_current[np.newaxis, :]  # shape=(1, A)
            )
            losses.append(loss)

        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.update_target_model()

        return float(np.mean(losses))

    def decay_epsilon(self, ep, episodes):
        if ep < 100:
            self.epsilon = 1.0
        else:
            self.epsilon = max(self.epsilon_min,
                               1.0 - (ep - 100) * (1.0 - self.epsilon_min) / (episodes - 100))