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
        
        # Q-값 범위 제한 (tanh로 -5~5 범위로 스케일링)
        q_scaled = Lambda(lambda x: 5.0 * tf.tanh(x))(q_total)  # (-5, 5) 범위

        return Model(inputs=[vehicle_input, request_input, relation_input], outputs=q_scaled)

    def act(self, state, action_mask):
        info = {
            'mode': None # 실제 사용 x, env.enrich_action에서 덮어짐.
        }
        
        # 유효한 액션 먼저 확인
        valid_actions = tf.where(action_mask == 1)
        
        if tf.shape(valid_actions)[0] == 0:
            # 유효한 액션이 없는 경우 REJECT 선택
            vehicle_idx = 0
            action_idx = cfg.POSSIBLE_ACTION - 1  # REJECT
            info['mode'] = 'fallback'
        elif np.random.rand() < self.epsilon:
            # 탐험: 유효한 액션 중에서 랜덤 선택
            info['mode'] = 'explore'
            rand_idx = tf.random.uniform(shape=(), maxval=tf.shape(valid_actions)[0], dtype=tf.int32)
            rand_action = valid_actions[rand_idx]
            rand_action = rand_action.numpy()
            vehicle_idx = int(rand_action[0])
            action_idx = int(rand_action[1])
        else:
            # 활용: Q-값 기반 선택 (안전한 마스킹)
            info['mode'] = 'exploit'
            q_values = self.model.predict(state, verbose=0)
            
            # 더 안전한 마스킹: 무효한 액션에 매우 작은 값 할당
            masked_q = tf.where(action_mask == 1, q_values, tf.constant(-1e1, dtype=tf.float32))
            
            # 전체에서 argmax 후 유효성 재검증
            flat_masked_q = tf.reshape(masked_q, (-1,))
            flat_idx = tf.argmax(flat_masked_q).numpy()
            
            vehicle_idx = int(flat_idx // cfg.POSSIBLE_ACTION)
            action_idx = int(flat_idx % cfg.POSSIBLE_ACTION)
            
            # 선택된 액션이 실제로 유효한지 최종 확인
            if action_mask[vehicle_idx][action_idx] != 1:
                # 무효한 액션이 선택된 경우 유효한 액션 중 첫 번째로 대체
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
            # print("\n\n================= Train : {} =================".format(self.train_step))
            batch = self.replay_buffer.sample(self.batch_size)

            # === Q(s', a') with numerical stability ===
            next_vehicle_tensor = np.array([b[3][0][0] for b in batch])
            next_request_tensor = np.array([b[3][1][0] for b in batch])
            next_relation_tensor = np.array([b[3][2][0] for b in batch])
            
            # NaN/Inf 체크 및 정리 (보상 범위에 맞는 클리핑)
            next_vehicle_tensor = np.nan_to_num(next_vehicle_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            next_request_tensor = np.nan_to_num(next_request_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            next_relation_tensor = np.nan_to_num(next_relation_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            
            next_states = [next_vehicle_tensor, next_request_tensor, next_relation_tensor]

            next_q_values = self.target_model.predict(next_states, verbose=0)  # (B, V, A)
            
            # Q-값 안정성 보장 (보상 범위에 맞는 강력한 클리핑)
            next_q_values = tf.where(tf.math.is_finite(next_q_values), next_q_values, tf.zeros_like(next_q_values))
            next_q_values = tf.clip_by_value(next_q_values, -5.0, 5.0)
            
            next_action_mask = np.array([b[5]['nm'] for b in batch])
            masked_next_q_values = tf.where(next_action_mask == 1, next_q_values, tf.constant(-1e1, dtype=tf.float32))
            max_next_q = np.max(masked_next_q_values, axis=(1, 2))
            
            # max_next_q 안정성 보장 (보상 범위에 맞는 강력한 클리핑)
            max_next_q = np.nan_to_num(max_next_q, nan=0.0, posinf=5.0, neginf=-5.0)
            max_next_q = np.clip(max_next_q, -5.0, 5.0)

            rewards = np.array([b[2] for b in batch], dtype=np.float32)
            dones = np.array([b[4] for b in batch], dtype=np.float32)
            
            # Rewards와 dones 안정성 보장 (실제 보상 범위 유지)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=5.0, neginf=-5.0)
            rewards = np.clip(rewards, -5.0, 5.0)
            
            targets = rewards + self.gamma * max_next_q * (1 - dones)
            targets = np.nan_to_num(targets, nan=0.0, posinf=5.0, neginf=-5.0)
            targets = np.clip(targets, -5.0, 5.0)

            # === Q(s, a) with numerical stability ===
            vehicle_tensor = np.array([b[0][0][0] for b in batch])  # (B, V, Dv)
            request_tensor = np.array([b[0][1][0] for b in batch])  # (B, R, Dr)
            relation_tensor = np.array([b[0][2][0] for b in batch])  # (B, V, R, Drel)
            
            # 입력 텐서 안정성 보장
            vehicle_tensor = np.nan_to_num(vehicle_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            request_tensor = np.nan_to_num(request_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            relation_tensor = np.nan_to_num(relation_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            
            states = [vehicle_tensor, request_tensor, relation_tensor]

            with tf.GradientTape() as tape:
                q_values = self.model(states, training=True)
                
                # Q-값 안정성 보장 (보상 범위에 맞는 강력한 클리핑)
                q_values = tf.where(tf.math.is_finite(q_values), q_values, tf.zeros_like(q_values))
                q_values = tf.clip_by_value(q_values, -5.0, 5.0)
                
                action_mask = np.array([b[5]['m'] for b in batch])
                masked_q_values = tf.where(action_mask == 1, q_values, tf.constant(-1e1, dtype=tf.float32))

                actions_raw = np.array([b[1] for b in batch])  # (B, 3)
                actions = actions_raw[:, :2]  # (B, 2)
                indices = tf.constant(actions, dtype=tf.int32)  # (B, 2)
                batch_indices = tf.range(tf.shape(indices)[0], dtype=tf.int32)[:, tf.newaxis]  # (B, 1)
                full_indices = tf.concat([batch_indices, indices], axis=1)  # (B, 3)

                q_sa = tf.gather_nd(masked_q_values, full_indices)  # (B,)
                
                # q_sa 안정성 보장 (보상 범위에 맞는 강력한 클리핑)
                q_sa = tf.where(tf.math.is_finite(q_sa), q_sa, tf.zeros_like(q_sa))
                q_sa = tf.clip_by_value(q_sa, -5.0, 5.0)

                # Huber Loss 계산 (MSE보다 안정적)
                diff = targets - q_sa
                diff = tf.where(tf.math.is_finite(diff), diff, tf.zeros_like(diff))
                
                # Huber Loss: 작은 오차에는 MSE, 큰 오차에는 MAE
                huber_delta = 0.5  # 임계값 (더 작게 조정)
                abs_diff = tf.abs(diff)
                is_small_error = abs_diff <= huber_delta
                
                small_error_loss = 0.5 * tf.square(diff)
                large_error_loss = huber_delta * abs_diff - 0.5 * huber_delta * huber_delta
                
                huber_loss = tf.where(is_small_error, small_error_loss, large_error_loss)
                raw_loss = tf.reduce_mean(huber_loss)
                
                # Loss 스케일링 (0.1배로 감소)
                loss = raw_loss * 0.1
                
                # Loss NaN 체크
                if tf.math.is_nan(loss):
                    print("[Warning] Loss is NaN! Setting to 0.")
                    return 0.0

            # Gradients 계산 및 안정성 보장
            grads = tape.gradient(loss, self.model.trainable_variables)
            if grads is not None:
                # 그래디언트 NaN/Inf 체크 및 클리핑
                safe_grads = []
                for grad in grads:
                    if grad is not None:
                        # NaN/Inf를 0으로 대체
                        safe_grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
                        # 그래디언트 노름 클리핑
                        safe_grad = tf.clip_by_norm(safe_grad, 1.0)
                        safe_grads.append(safe_grad)
                    else:
                        safe_grads.append(None)
                
                # 안전한 그래디언트만 적용
                valid_grads_and_vars = [(g, v) for g, v in zip(safe_grads, self.model.trainable_variables) if g is not None]
                if valid_grads_and_vars:
                    self.optimizer.apply_gradients(valid_grads_and_vars)

            self.train_step += 1
            if self.train_step % self.update_target_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

            return loss.numpy()

        except Exception as e:
            print(f"[Error] Training failed: {e}")
            return 0.0
