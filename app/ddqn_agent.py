import os
import numpy as np
import tensorflow as tf
import app.config as cfg
from tensorflow.keras import mixed_precision

from app.pending_buffer import PendingBuffer
from app.prioritized_replay_buffer import PrioritizedReplayBuffer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, RepeatVector, Reshape


class DDQNAgent:
    def __init__(self, hidden_dim, batch_size, learning_rate):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.train_step = 0
        self.update_target_freq = 200  # 더 자주 타겟 네트워크 업데이트
        # Mixed precision에서 손실 스케일링 적용
        base_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        try:
            self.optimizer = mixed_precision.LossScaleOptimizer(base_opt)
        except Exception:
            self.optimizer = base_opt

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # 더 낮은 최소 epsilon
        self.epsilon_decay = 0.9995  # 더 천천히 감소

        # Prioritized Experience Replay 사용
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=800,  # 1500 -> 800 (OOM 완화)
            alpha=0.6,  # 우선순위 강도
            beta=0.4,   # IS weights 보정 강도 (초기값)
            beta_increment=0.00015  # 5000 ep 기준: 4000 ep에서 1.0 도달
        )
        self.pending_buffer = PendingBuffer()
        
        # 학습률 스케줄링
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        
        # 성능 추적을 위한 변수들
        self.recent_rewards = []
        self.performance_window = 10
        # PER beta 스케줄 (에피소드 기반)
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.beta_target_episode = 4000  # 5000 ep의 80%

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
            masked_q = tf.where(action_mask == 1, q_values, tf.constant(-1e1, dtype=q_values.dtype))
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
    
    def adaptive_epsilon_decay(self, episode_reward):
        """성능 기반 적응형 Epsilon 감소"""
        # 최근 보상 기록
        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > self.performance_window:
            self.recent_rewards.pop(0)
        
        # 성능 기반 epsilon 조절
        if len(self.recent_rewards) >= 5:
            recent_avg = sum(self.recent_rewards[-5:]) / 5
            older_avg = sum(self.recent_rewards[-10:-5]) / 5 if len(self.recent_rewards) >= 10 else recent_avg
            
            if recent_avg > older_avg:
                # 성능이 향상되면 더 천천히 감소 (더 많은 탐험)
                self.epsilon_decay = 0.9998
            else:
                # 성능이 하락하면 더 빠르게 감소 (더 많은 활용)
                self.epsilon_decay = 0.9990
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_learning_rate(self, episode):
        """개선된 학습률 스케줄링 - 더 공격적인 학습"""
        if episode < 15:
            # 초기 빠른 학습 (더 높은 학습률)
            self.current_learning_rate = self.initial_learning_rate * 1.2
        elif episode < 30:
            # 중간 안정화
            self.current_learning_rate = self.initial_learning_rate * 0.8
        elif episode < 40:
            # 후반 정교화
            self.current_learning_rate = self.initial_learning_rate * 0.5
        else:
            # 최종 단계 (더 낮은 학습률로 안정화)
            self.current_learning_rate = self.initial_learning_rate * 0.2
        
        # 옵티마이저 학습률 업데이트 (LossScaleOptimizer 사용 시 내부 옵티마이저 갱신)
        try:
            self.optimizer.inner_optimizer.learning_rate.assign(self.current_learning_rate)
        except Exception:
            self.optimizer.learning_rate.assign(self.current_learning_rate)

        # PER beta를 에피소드 진행도에 맞춰 설정
        progress = min(1.0, max(0.0, episode / float(self.beta_target_episode)))
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        if hasattr(self, 'replay_buffer') and hasattr(self.replay_buffer, 'set_beta'):
            self.replay_buffer.set_beta(beta)

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
            # PER: batch, idxs, is_weights 반환
            result = self.replay_buffer.sample(self.batch_size)
            if result is None:
                return None
            batch, idxs, is_weights = result

            # === Q(s', a') with numerical stability ===
            next_vehicle_tensor = np.array([b[3][0][0] for b in batch], dtype=np.float32)
            next_request_tensor = np.array([b[3][1][0] for b in batch], dtype=np.float32)
            next_relation_tensor = np.array([b[3][2][0] for b in batch], dtype=np.float32)

            # Clean inputs
            next_vehicle_tensor = np.nan_to_num(next_vehicle_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            next_request_tensor = np.nan_to_num(next_request_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            next_relation_tensor = np.nan_to_num(next_relation_tensor, nan=0.0, posinf=5.0, neginf=-5.0)

            next_states = [next_vehicle_tensor, next_request_tensor, next_relation_tensor]

            # === Double DQN Implementation ===
            # Step 1: Main Network로 액션 선택
            next_q_main = self.model.predict(next_states, verbose=0)
            next_q_main = tf.where(tf.math.is_finite(next_q_main), next_q_main, tf.zeros_like(next_q_main))
            next_q_main = tf.clip_by_value(next_q_main, -5.0, 5.0)
            
            next_action_mask = np.array([b[5]['nm'] for b in batch])
            masked_next_q_main = tf.where(next_action_mask == 1, next_q_main, tf.constant(-1e1, dtype=next_q_main.dtype))
            
            # Main Network에서 최적 액션 선택
            max_indices = tf.argmax(tf.reshape(masked_next_q_main, (self.batch_size, -1)), axis=1)
            # GPU 친화적 연산: FloorMod 대신 floordiv와 수동 계산 (타입 일치 유지)
            max_indices_int64 = tf.cast(max_indices, dtype=tf.int64)
            possible_action_int64 = tf.constant(cfg.POSSIBLE_ACTION, dtype=tf.int64)
            vehicle_indices_int64 = tf.math.floordiv(max_indices_int64, possible_action_int64)
            action_indices_int64 = max_indices_int64 - vehicle_indices_int64 * possible_action_int64
            # 최종적으로 int32로 변환
            vehicle_indices = tf.cast(vehicle_indices_int64, dtype=tf.int32)
            action_indices = tf.cast(action_indices_int64, dtype=tf.int32)
            
            # Step 2: Target Network로 선택된 액션 평가
            next_q_target = self.target_model.predict(next_states, verbose=0)
            next_q_target = tf.where(tf.math.is_finite(next_q_target), next_q_target, tf.zeros_like(next_q_target))
            next_q_target = tf.clip_by_value(next_q_target, -5.0, 5.0)
            
            # 선택된 액션의 Target Network 값 추출
            batch_indices = tf.range(self.batch_size, dtype=tf.int32)
            selected_indices = tf.stack([batch_indices, vehicle_indices, action_indices], axis=1)
            max_next_q = tf.gather_nd(next_q_target, selected_indices)
            
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
            vehicle_tensor = np.array([b[0][0][0] for b in batch], dtype=np.float32)
            request_tensor = np.array([b[0][1][0] for b in batch], dtype=np.float32)
            relation_tensor = np.array([b[0][2][0] for b in batch], dtype=np.float32)

            vehicle_tensor = np.nan_to_num(vehicle_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            request_tensor = np.nan_to_num(request_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            relation_tensor = np.nan_to_num(relation_tensor, nan=0.0, posinf=5.0, neginf=-5.0)

            states = [vehicle_tensor, request_tensor, relation_tensor]

            with tf.GradientTape() as tape:
                q_values = self.model(states, training=True)
                q_values = tf.where(tf.math.is_finite(q_values), q_values, tf.zeros_like(q_values))
                q_values = tf.clip_by_value(q_values, -5.0, 5.0)

                action_mask = np.array([b[5]['m'] for b in batch])
                masked_q_values = tf.where(action_mask == 1, q_values, tf.constant(-1e1, dtype=q_values.dtype))

                actions_raw = np.array([b[1] for b in batch])
                actions = actions_raw[:, :2]
                indices = tf.constant(actions, dtype=tf.int32)
                batch_indices = tf.range(tf.shape(indices)[0], dtype=tf.int32)[:, tf.newaxis]
                full_indices = tf.concat([batch_indices, indices], axis=1)

                q_sa = tf.gather_nd(masked_q_values, full_indices)
                q_sa = tf.where(tf.math.is_finite(q_sa), q_sa, tf.zeros_like(q_sa))
                q_sa = tf.clip_by_value(q_sa, -5.0, 5.0)

                # Huber loss (more robust than MSE) - dtype 일치 유지
                targets_tf = tf.convert_to_tensor(targets, dtype=q_sa.dtype)
                diff = targets_tf - q_sa
                diff = tf.where(tf.math.is_finite(diff), diff, tf.zeros_like(diff))
                diff = tf.clip_by_value(diff, -10.0, 10.0)
                td_errors = diff  # PER: TD-error 저장 (TF 텐서)

                huber_delta = tf.cast(0.5, dtype=diff.dtype)
                abs_diff = tf.abs(diff)
                is_small = abs_diff <= huber_delta
                small_loss = tf.cast(0.5, diff.dtype) * tf.square(diff)
                large_loss = huber_delta * abs_diff - tf.cast(0.5, diff.dtype) * huber_delta * huber_delta
                huber_loss = tf.where(is_small, small_loss, large_loss)
                
                # PER: Importance Sampling Weights 적용 (dtype 정합)
                is_weights_tensor = tf.convert_to_tensor(is_weights, dtype=huber_loss.dtype)
                weighted_loss = huber_loss * is_weights_tensor
                raw_loss = tf.reduce_mean(weighted_loss)
                loss = raw_loss * 0.1  # scale down

                if tf.math.is_nan(loss):
                    print("[Warning] Loss is NaN! Forcing 0.0")
                    return 0.0

            # Mixed precision: scale/unscale gradients if 적용됨
            try:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
                grads = tape.gradient(scaled_loss, self.model.trainable_variables)
                grads = self.optimizer.get_unscaled_gradients(grads)
            except Exception:
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

            # PER: 우선순위 업데이트 (numpy로 변환하여 안전하게 저장)
            td_errors_np = td_errors.numpy().astype(np.float32)
            self.replay_buffer.update_priorities(idxs, td_errors_np)

            # 명시적 메모리 해제 (GPU OOM 방지)
            loss_value = float(loss.numpy())
            del batch, next_states, states, next_q_main, next_q_target
            del q_values, masked_q_values, masked_next_q_main
            del vehicle_tensor, request_tensor, relation_tensor
            del next_vehicle_tensor, next_request_tensor, next_relation_tensor
            del td_errors, is_weights_tensor
            
            return loss_value

        except Exception as e:
            print(f"[Error] Training failed: {e}")
            return 0.0