import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from app.pending_buffer import PendingBuffer
from app.replay_buffer import ReplayBuffer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, RepeatVector, Reshape


class DQNAgent:
    def __init__(self, hidden_dim, batch_size, learning_rate):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.train_step = 0
        self.update_target_freq = 500
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.replay_buffer = ReplayBuffer()
        self.pending_buffer = PendingBuffer()

    def save_model(self, file_path):
        self.model.save_weights(file_path)
        print(f"Model weights saved at {file_path}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model.load_weights(file_path)
            self.target_model.set_weights(self.model.get_weights())
            print(f"Model weights loaded at {file_path}")
        else:
            print(f"No model weights loaded at {file_path}")

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
        q_match = TimeDistributed(TimeDistributed(Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')))(q_match)  # (B, V, R, 1)
        q_match = Lambda(lambda x: tf.squeeze(x, axis=-1))(q_match)  # (B, V, R)

        r_summary = tf.reduce_mean(r_embed, axis=1)  # (B, H)
        r_summary = RepeatVector(cfg.MAX_NUM_VEHICLES)(r_summary)  # (B, V, H)
        reject_context = Concatenate(axis=-1)([v_embed, r_summary])  # (B, V, 2H)

        q_reject = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(reject_context)
        q_reject = TimeDistributed(Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros'))(q_reject)  # (B, V, 1)

        # Concatenate along request dim → total 21 actions
        q_total = Concatenate(axis=-1)([q_match, q_reject])  # (B, V, R+1)

        # Output scaling to bound Q-values and prevent explosion
        q_scaled = Lambda(lambda x: 5.0 * tf.tanh(x))(q_total)  # (-5, 5)

        return Model(inputs=[vehicle_input, request_input, relation_input], outputs=q_scaled)

    def act(self, state, action_mask):
        info = {
            'mode': None
        }
        # 유효한 액션 먼저 확인
        valid_actions = tf.where(action_mask == 1)
        if tf.shape(valid_actions)[0] == 0:
            # 유효한 액션이 없는 경우 REJECT로 폴백
            vehicle_idx = 0
            action_idx = cfg.POSSIBLE_ACTION - 1
            info['mode'] = 'fallback'
            return [vehicle_idx, action_idx, info]

        if np.random.rand() < self.epsilon:
            info['mode'] = 'explore'
            rand_idx = tf.random.uniform(shape=(), maxval=tf.shape(valid_actions)[0], dtype=tf.int32)
            rand_action = valid_actions[rand_idx].numpy()
            vehicle_idx = int(rand_action[0])
            action_idx = int(rand_action[1])
        else:
            info['mode'] = 'exploit'
            q_values = self.model.predict(state, verbose=0)
            # 무효 액션에 작은 값 할당(너무 큰 음수는 불필요한 수치문제 유발)
            masked_q = tf.where(action_mask == 1, q_values, tf.constant(-1e1, dtype=tf.float32))
            flat_idx = tf.argmax(tf.reshape(masked_q, (-1,))).numpy()
            vehicle_idx = int(flat_idx // cfg.POSSIBLE_ACTION)
            action_idx = int(flat_idx % cfg.POSSIBLE_ACTION)
            # 최종 유효성 재검증
            if action_mask[vehicle_idx][action_idx] != 1:
                first_valid = valid_actions[0].numpy()
                vehicle_idx = int(first_valid[0])
                action_idx = int(first_valid[1])
                info['mode'] = 'exploit_safe_fallback'
        return [vehicle_idx, action_idx, info]

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        # print(f"[Agent] Epsilon decayed to {self.epsilon:.4f}")

    def remember(self, transition):
        self.replay_buffer.append(transition)

    def pending(self, transition):
        action = transition[1]
        action_id = action[2]['id']
        self.pending_buffer.add(action_id, transition)

    def confirm_and_remember(self, action_id, reward):
        transition = self.pending_buffer.confirm(action_id, reward)
        if transition is not None:
            self.remember(transition)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        try:
            batch = self.replay_buffer.sample(self.batch_size)

            # === Q(s', a') with numerical stability ===
            next_vehicle_tensor = np.array([b[3][0][0] for b in batch])
            next_request_tensor = np.array([b[3][1][0] for b in batch])
            next_relation_tensor = np.array([b[3][2][0] for b in batch])

            # Clean inputs
            next_vehicle_tensor = np.nan_to_num(next_vehicle_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            next_request_tensor = np.nan_to_num(next_request_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            next_relation_tensor = np.nan_to_num(next_relation_tensor, nan=0.0, posinf=5.0, neginf=-5.0)

            next_states = [next_vehicle_tensor, next_request_tensor, next_relation_tensor]

            next_q_values = self.target_model.predict(next_states, verbose=0)
            next_q_values = tf.where(tf.math.is_finite(next_q_values), next_q_values, tf.zeros_like(next_q_values))
            next_q_values = tf.clip_by_value(next_q_values, -5.0, 5.0)

            next_action_mask = np.array([b[5]['nm'] for b in batch])
            masked_next_q_values = tf.where(next_action_mask == 1, next_q_values, tf.constant(-1e1, dtype=tf.float32))
            max_next_q = np.max(masked_next_q_values, axis=(1, 2))
            max_next_q = np.nan_to_num(max_next_q, nan=0.0, posinf=5.0, neginf=-5.0)
            max_next_q = np.clip(max_next_q, -5.0, 5.0)

            rewards = np.array([b[2] for b in batch], dtype=np.float32)
            dones = np.array([b[4] for b in batch], dtype=np.float32)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=5.0, neginf=-5.0)
            rewards = np.clip(rewards, -5.0, 5.0)

            targets = rewards + self.gamma * max_next_q * (1 - dones)
            targets = np.nan_to_num(targets, nan=0.0, posinf=5.0, neginf=-5.0)
            targets = np.clip(targets, -5.0, 5.0)

            # === Q(s, a) with numerical stability ===
            vehicle_tensor = np.array([b[0][0][0] for b in batch])
            request_tensor = np.array([b[0][1][0] for b in batch])
            relation_tensor = np.array([b[0][2][0] for b in batch])

            vehicle_tensor = np.nan_to_num(vehicle_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            request_tensor = np.nan_to_num(request_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            relation_tensor = np.nan_to_num(relation_tensor, nan=0.0, posinf=5.0, neginf=-5.0)

            states = [vehicle_tensor, request_tensor, relation_tensor]

            with tf.GradientTape() as tape:
                q_values = self.model(states, training=True)
                q_values = tf.where(tf.math.is_finite(q_values), q_values, tf.zeros_like(q_values))
                q_values = tf.clip_by_value(q_values, -5.0, 5.0)

                action_mask = np.array([b[5]['m'] for b in batch])
                masked_q_values = tf.where(action_mask == 1, q_values, tf.constant(-1e1, dtype=tf.float32))

                actions_raw = np.array([b[1] for b in batch])
                actions = actions_raw[:, :2]
                indices = tf.constant(actions, dtype=tf.int32)
                batch_indices = tf.range(tf.shape(indices)[0], dtype=tf.int32)[:, tf.newaxis]
                full_indices = tf.concat([batch_indices, indices], axis=1)

                q_sa = tf.gather_nd(masked_q_values, full_indices)
                q_sa = tf.where(tf.math.is_finite(q_sa), q_sa, tf.zeros_like(q_sa))
                q_sa = tf.clip_by_value(q_sa, -5.0, 5.0)

                # Huber loss (more robust than MSE)
                diff = targets - q_sa.numpy()
                diff = np.nan_to_num(diff, nan=0.0, posinf=5.0, neginf=-5.0)
                diff = np.clip(diff, -10.0, 10.0)
                diff = tf.convert_to_tensor(diff, dtype=tf.float32)

                huber_delta = 0.5
                abs_diff = tf.abs(diff)
                is_small = abs_diff <= huber_delta
                small_loss = 0.5 * tf.square(diff)
                large_loss = huber_delta * abs_diff - 0.5 * huber_delta * huber_delta
                huber_loss = tf.where(is_small, small_loss, large_loss)
                raw_loss = tf.reduce_mean(huber_loss)
                loss = raw_loss * 0.1  # scale down

                if tf.math.is_nan(loss):
                    print("[Warning] Loss is NaN! Forcing 0.0")
                    return 0.0

            grads = tape.gradient(loss, self.model.trainable_variables)
            if grads is not None:
                safe_grads = []
                for grad in grads:
                    if grad is not None:
                        g = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
                        g = tf.clip_by_norm(g, 1.0)
                        safe_grads.append(g)
                    else:
                        safe_grads.append(None)
                pairs = [(g, v) for g, v in zip(safe_grads, self.model.trainable_variables) if g is not None]
                if pairs:
                    self.optimizer.apply_gradients(pairs)

            self.train_step += 1
            if self.train_step % self.update_target_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

            return loss.numpy()

        except Exception as e:
            print(f"[Error] Training failed: {e}")
            return 0.0