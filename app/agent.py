import os
import random
import numpy as np
import tensorflow as tf
import app.config as cfg

from collections import deque
from app.action_type import ActionType
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, RepeatVector, Reshape
from tensorflow.python.distribute.combinations import tf_function


class DQNAgent:
    def __init__(self, hidden_dim, max_episodes=500):
        self.hidden_dim = hidden_dim
        self.model = self.build_model()
        self.target_model = self.build_model()

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

        self.batch_size = 32
        self.memory = deque(maxlen=20000)

        # self.update_target_model()
        #  self.target_update_counter = 0
        # self.train_counter = 0
        # self.target_update_freq = 1000

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
        vehicle_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM), name="vehicle_input")  # (B, V, Dv)
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name="request_input")  # (B, R, Dr)
        relation_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM), name="relation_input") # (B, V, R, Drel)

        v_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(vehicle_input)  # (B, V, H)
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(request_input)  # (B, R, H)

        v_expand = tf.expand_dims(v_embed, axis=2)  # (B, V, 1, H)
        r_expand = tf.expand_dims(r_embed, axis=1)  # (B, 1, R, H)

        v_tiled = tf.tile(v_expand, [1, 1, cfg.MAX_NUM_REQUEST, 1])  # (B, V, R, H)
        r_tiled = tf.tile(r_expand, [1, cfg.MAX_NUM_VEHICLES, 1, 1])  # (B, V, R, H)

        # Broadcast concat to shape (B, V, R, 2H + Drel)
        pair_embed = Concatenate(axis=-1)([v_tiled, r_tiled, relation_input])  # (B, V, R, 2H + Drel)

        q_match = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation='relu')))(pair_embed)  # (B, V, R, H)
        q_match = TimeDistributed(TimeDistributed(Dense(1)))(q_match)  # (B, V, R, 1)
        q_match = Lambda(lambda x: tf.squeeze(x, axis=-1))(q_match)  # (B, V, R)

        r_summary = tf.reduce_mean(r_embed, axis=1)  # (B, H)
        r_summary = RepeatVector(cfg.MAX_NUM_VEHICLES)(r_summary)  # (B, V, H)
        reject_context = Concatenate(axis=-1)([v_embed, r_summary])  # (B, V, 2H)

        q_reject = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(reject_context)
        q_reject = TimeDistributed(Dense(1))(q_reject)  # (B, V, 1)

        # Concatenate along request dim → total 21 actions
        q_total = Concatenate(axis=-1)([q_match, q_reject])  # (B, V, R+1)

        return Model(inputs=[vehicle_input, request_input, relation_input], outputs=q_total)

    def act(self, state, action_mask):
        if np.random.rand() < self.epsilon:
            valid_actions = tf.where(action_mask == 1)
            rand_idx = tf.random.uniform(shape=(), maxval=tf.shape(valid_actions)[0], dtype=tf.int32)
            rand_action = valid_actions[rand_idx]
            rand_action = rand_action.numpy()
            return [int(rand_action[0]), int(rand_action[1]), 'explore']
        else:
            q_values = self.model.predict(state)
            masked_q = tf.where(action_mask == 1, q_values, tf.float32.min)
            flat_idx = tf.argmax(tf.reshape(masked_q, (-1,))).numpy()
            vehicle_idx = int(flat_idx // cfg.POSSIBLE_ACTION)
            action_idx = int(flat_idx % cfg.POSSIBLE_ACTION)
            return [vehicle_idx, action_idx, 'exploit']

    @tf_function
    def predict_q_all(self, states: tf.Tensor) -> tf.Tensor:
        return self.model(states, training=False)

    @tf_function
    def predict_target_q_all(self, states: tf.Tensor) -> tf.Tensor:
        return self.target_model(states, training=False)

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