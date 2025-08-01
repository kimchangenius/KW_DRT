import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from app.pending_buffer import PendingBuffer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, RepeatVector


class PPOAgent:
    def __init__(self, hidden_dim, batch_size, learning_rate):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # PPO specific hyperparameters (더 적극적으로 조정)
        self.clip_ratio = 0.2  # 0.1 → 0.2 (더 큰 정책 변화 허용)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01  # 0.05 → 0.01 (엔트로피 보너스 감소)
        self.value_coef = 0.5
        self.max_grad_norm = 1.0  # 0.3 → 1.0 (더 큰 그래디언트 허용)
        self.target_kl = 0.02  # 0.005 → 0.02 (더 관대한 KL divergence 제한)
        
        # Learning rate scheduling
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        
        # PPO는 episode 기반 학습이 더 효과적
        self.episode_buffer = []  # 현재 에피소드의 transitions
        self.update_frequency = 10  # 3 → 10 (더 많은 데이터로 학습)
        
        # Build actor and critic networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.current_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.current_learning_rate)
        
        # Experience buffer
        self.pending_buffer = PendingBuffer()
        self.trajectory_buffer = []
        
    def save_model(self, file_path):
        actor_path = file_path.replace('.h5', '_actor.h5')
        critic_path = file_path.replace('.h5', '_critic.h5')
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Model weights saved at {actor_path} and {critic_path}")
        
    def load_model(self, file_path):
        actor_path = file_path.replace('.h5', '_actor.h5')
        critic_path = file_path.replace('.h5', '_critic.h5')
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print(f"Model weights loaded from {actor_path} and {critic_path}")
        else:
            print(f"No model weights found at {actor_path} or {critic_path}")
            
    def build_shared_network(self):
        """공통 feature extraction network 구축"""
        vehicle_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM), name="vehicle_input")
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name="request_input")
        relation_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM), name="relation_input")
        
        # Embed vehicles and requests
        v_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(vehicle_input)
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(request_input)
        
        # Create vehicle-request pairs
        v_expand = tf.expand_dims(v_embed, axis=2)  # (B, V, 1, H)
        r_expand = tf.expand_dims(r_embed, axis=1)  # (B, 1, R, H)
        
        v_tiled = tf.tile(v_expand, [1, 1, cfg.MAX_NUM_REQUEST, 1])  # (B, V, R, H)
        r_tiled = tf.tile(r_expand, [1, cfg.MAX_NUM_VEHICLES, 1, 1])  # (B, V, R, H)
        
        # Concatenate all features
        pair_embed = Concatenate(axis=-1)([v_tiled, r_tiled, relation_input])  # (B, V, R, 2H + Drel)
        
        return [vehicle_input, request_input, relation_input], pair_embed, v_embed, r_embed
        
    def build_actor(self):
        """Policy network (Actor) 구축"""
        inputs, pair_embed, v_embed, r_embed = self.build_shared_network()
        
        # Process vehicle-request pairs for matching actions
        match_logits = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation='relu')))(pair_embed)
        match_logits = TimeDistributed(TimeDistributed(Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')))(match_logits)
        match_logits = Lambda(lambda x: tf.squeeze(x, axis=-1))(match_logits)  # (B, V, R)
        
        # Process reject actions
        r_summary = tf.reduce_mean(r_embed, axis=1)  # (B, H)
        r_summary = RepeatVector(cfg.MAX_NUM_VEHICLES)(r_summary)  # (B, V, H)
        reject_context = Concatenate(axis=-1)([v_embed, r_summary])  # (B, V, 2H)
        
        reject_logits = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(reject_context)
        reject_logits = TimeDistributed(Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros'))(reject_logits)  # (B, V, 1)
        
        # Combine all action logits
        action_logits = Concatenate(axis=-1)([match_logits, reject_logits])  # (B, V, R+1)
        
        return Model(inputs=inputs, outputs=action_logits)
        
    def build_critic(self):
        """Value network (Critic) 구축"""
        inputs, pair_embed, v_embed, r_embed = self.build_shared_network()
        
        # Global state representation for value estimation
        global_v = tf.reduce_mean(v_embed, axis=1)  # (B, H)
        global_r = tf.reduce_mean(r_embed, axis=1)  # (B, H)
        global_state = Concatenate(axis=-1)([global_v, global_r])  # (B, 2H)
        
        # Value estimation
        value = Dense(self.hidden_dim, activation='relu')(global_state)
        value = Dense(self.hidden_dim, activation='relu')(value)
        value = Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')(value)  # (B, 1)
        
        # 출력 값 클리핑으로 극값 방지
        value = tf.clip_by_value(value, -10.0, 10.0)
        
        return Model(inputs=inputs, outputs=value)
        
    def get_action_probs(self, state, action_mask):
        """상태에서 액션 확률 분포 계산"""
        logits = self.actor(state, training=False)  # (B, V, A)
        
        # Actor 출력 직접 디버깅
        # print(f"[Debug] Raw logits: min={tf.reduce_min(logits):.6f}, max={tf.reduce_max(logits):.6f}")
        # print(f"[Debug] Raw logits has_inf={tf.reduce_any(tf.math.is_inf(logits))}, has_nan={tf.reduce_any(tf.math.is_nan(logits))}")
        
        # NaN/무한대 직접 처리 (가장 강력한 방법)
        logits = tf.where(tf.math.is_finite(logits), logits, tf.zeros_like(logits))
        logits = tf.clip_by_value(logits, -5.0, 5.0)
        # print(f"[Debug] Cleaned logits: min={tf.reduce_min(logits):.6f}, max={tf.reduce_max(logits):.6f}")
        
        # Apply action mask (더 안전한 값 사용)
        masked_logits = tf.where(action_mask == 1, logits, tf.constant(-5.0, dtype=tf.float32))
        
        # Convert to probabilities (대폭 강화된 수치적 안정성)
        # 더 강한 로그잇 클리핑
        clipped_logits = tf.clip_by_value(masked_logits, -5.0, 5.0)
        
        # 온도 조정 소프트맥스 (더 안정적)
        temperature = 2.0
        scaled_logits = clipped_logits / temperature
        action_probs = tf.nn.softmax(scaled_logits, axis=-1)
        
        # 더 높은 확률 하한선
        action_probs = tf.maximum(action_probs, 1e-6)
        
        # 재정규화
        action_probs = action_probs / tf.reduce_sum(action_probs, axis=-1, keepdims=True)
        
        # 최종 확률 체크
        action_probs = tf.where(tf.math.is_finite(action_probs), action_probs, tf.constant(1e-6, dtype=tf.float32))
        # print(f"[Debug] Final action_probs: min={tf.reduce_min(action_probs):.6f}, max={tf.reduce_max(action_probs):.6f}")
        
        return action_probs, logits
        
    def act(self, state, action_mask):
        """액션 선택 - DQN과 유사한 안전한 방식"""
        action_probs, logits = self.get_action_probs(state, action_mask)
        
        # 유효한 액션들의 인덱스 찾기 
        valid_actions = tf.where(action_mask == 1)
        
        if tf.shape(valid_actions)[0] == 0:
            # 유효한 액션이 없는 경우 (예외 상황)
            # print("[Warning] 유효한 Action이 없음")
            vehicle_idx = 0
            action_idx = cfg.POSSIBLE_ACTION - 1  # REJECT
            action_prob_value = 1.0
        else:
            # 마스킹된 확률 분포에서 샘플링
            masked_probs = tf.where(action_mask == 1, action_probs, tf.constant(0.0, dtype=tf.float32))
            masked_probs_flat = tf.reshape(masked_probs, (-1,))
            
            # 확률이 0이 아닌 액션들만 고려
            non_zero_indices = tf.where(masked_probs_flat > 0)[:, 0]
            
            if tf.shape(non_zero_indices)[0] == 0:
                # 모든 확률이 0인 경우 - 유효한 액션 중 균등하게 선택
                # print("[Warning] 모든 action 확률이 0임")
                valid_indices_flat = tf.reshape(tf.where(action_mask == 1), (-1,))
                # 2D 인덱스를 1D로 변환
                vehicle_indices = valid_indices_flat[::2]  # 짝수 인덱스 (vehicle)
                action_indices = valid_indices_flat[1::2]  # 홀수 인덱스 (action)
                flat_valid_indices = vehicle_indices * cfg.POSSIBLE_ACTION + action_indices
                
                # 균등 확률로 샘플링
                random_idx = tf.random.uniform([], 0, tf.shape(flat_valid_indices)[0], dtype=tf.int32)
                flat_action_idx = flat_valid_indices[random_idx]
                
                vehicle_idx = int(flat_action_idx // cfg.POSSIBLE_ACTION)
                action_idx = int(flat_action_idx % cfg.POSSIBLE_ACTION)
                action_prob_value = 1.0 / float(tf.shape(flat_valid_indices)[0])
            else:
                non_zero_probs = tf.gather(masked_probs_flat, non_zero_indices)
                
                # 정규화
                non_zero_probs = non_zero_probs / tf.reduce_sum(non_zero_probs)
                
                # 샘플링
                sampled_idx = tf.random.categorical(tf.math.log(non_zero_probs[tf.newaxis, :]), 1)[0, 0]
                flat_action_idx = non_zero_indices[sampled_idx]
                
                # 차량과 액션 인덱스 계산
                vehicle_idx = int(flat_action_idx // cfg.POSSIBLE_ACTION)
                action_idx = int(flat_action_idx % cfg.POSSIBLE_ACTION)
                
                # 해당 액션의 확률
                action_prob_value = float(action_probs[0][vehicle_idx][action_idx].numpy())
        
        info = {
            'mode': 'sample',
            'action_prob': action_prob_value,
            'logits': logits.numpy()
        }
        
        return [vehicle_idx, action_idx, info]
        
    def get_value(self, state):
        """상태 가치 추정"""
        return self.critic(state, training=False)
        
    def remember(self, transition):
        """에피소드 기반으로 경험 저장"""
        self.episode_buffer.append(transition)
        
    def pending(self, transition):
        """Pending action 처리"""
        action = transition[1]
        action_id = action[2]['id']
        self.pending_buffer.add(action_id, transition)
        
    def confirm_and_remember(self, action_id, reward):
        """Pending action 확정 및 경험 저장"""
        transition = self.pending_buffer.confirm(action_id, reward)
        if transition is not None:
            self.remember(transition)
            
    def finish_episode(self):
        """에피소드가 끝날 때 trajectory buffer에 추가"""
        if len(self.episode_buffer) > 0:
            # print(f"[Debug] finish_episode: episode_buffer {len(self.episode_buffer)}개 → trajectory_buffer")
            self.trajectory_buffer.extend(self.episode_buffer)
            # print(f"[Debug] finish_episode 후: trajectory_buffer 크기 = {len(self.trajectory_buffer)}")
            self.episode_buffer = []
            
    def should_train(self):
        """학습을 수행할 조건 확인"""
        should = len(self.trajectory_buffer) >= self.update_frequency
        # if should:
        #     print(f"[Debug] should_train() = True, buffer 크기: {len(self.trajectory_buffer)}")
        return should
        
    def update_learning_rate(self, kl_div):
        """KL divergence 기반 적응적 learning rate 조정"""
        if kl_div > self.target_kl * 1.5:
            self.current_learning_rate *= 0.5  # 감소
            print(f"Learning rate decreased to {self.current_learning_rate:.6f}")
        elif kl_div < self.target_kl / 1.5:
            self.current_learning_rate *= 1.2  # 증가
            print(f"Learning rate increased to {self.current_learning_rate:.6f}")
        
        # 옵티마이저 learning rate 업데이트
        self.actor_optimizer.learning_rate.assign(self.current_learning_rate)
        self.critic_optimizer.learning_rate.assign(self.current_learning_rate)
        
    def compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation 계산 (NaN 안전장치 포함)"""
        # 입력 값 NaN 체크 및 대체
        if np.any(np.isnan(values)):
            # print("[Warning] Values contain NaN! Replacing with zeros.")
            values = np.nan_to_num(values, nan=0.0)
        
        if np.any(np.isnan(rewards)):
            # print("[Warning] Rewards contain NaN! Replacing with zeros.")
            rewards = np.nan_to_num(rewards, nan=0.0)
            
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[i])
            
            # NaN 체크
            if np.isnan(gae):
                # print(f"[Warning] GAE is NaN at step {i}! Setting to 0.")
                gae = 0.0
                
            advantages.insert(0, gae)
            
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values
        
        # NaN 최종 체크
        if np.any(np.isnan(advantages)):
            # print("[Warning] Advantages contain NaN! Replacing with zeros.")
            advantages = np.nan_to_num(advantages, nan=0.0)
            
        if np.any(np.isnan(returns)):
            # print("[Warning] Returns contain NaN! Replacing with zeros.")
            returns = np.nan_to_num(returns, nan=0.0)
        
        # Normalize advantages
        if len(advantages) > 1:
            std_val = np.std(advantages)
            if std_val > 1e-8:
                advantages = (advantages - np.mean(advantages)) / std_val
            else:
                advantages = advantages - np.mean(advantages)
        
        return advantages, returns
        
    def train(self):
        """PPO 학습"""
        # 학습 조건 강화
        if len(self.trajectory_buffer) < 5:  # 2 → 5 (더 많은 데이터 요구)
            # print(f"[Debug] 학습 스킵: trajectory_buffer 크기 = {len(self.trajectory_buffer)}")
            return None
        
        # print(f"[Debug] 학습 시작: trajectory_buffer 크기 = {len(self.trajectory_buffer)}")
            
        batch = self.trajectory_buffer[:]  # 모든 데이터 사용
        
        # 상태, 액션, 보상 등 추출
        states = [
            np.array([b[0][0][0] for b in batch]),  # vehicle states
            np.array([b[0][1][0] for b in batch]),  # request states  
            np.array([b[0][2][0] for b in batch])   # relation states
        ]
        
        actions = np.array([[b[1][0], b[1][1]] for b in batch])  # (B, 2)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = [
            np.array([b[3][0][0] for b in batch]),
            np.array([b[3][1][0] for b in batch]),
            np.array([b[3][2][0] for b in batch])
        ]
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        action_masks = np.array([b[5]['m'] for b in batch])
        old_action_probs = np.array([b[1][2]['action_prob'] for b in batch], dtype=np.float32)
        
        # 가치 추정
        values = tf.reshape(self.get_value(states), [-1]).numpy()
        next_values = tf.reshape(self.get_value(next_states), [-1]).numpy()
        
        # print(f"[Debug] values: min={np.min(values):.6f}, max={np.max(values):.6f}, mean={np.mean(values):.6f}")
        # print(f"[Debug] rewards: min={np.min(rewards):.6f}, max={np.max(rewards):.6f}, mean={np.mean(rewards):.6f}")
        # print(f"[Debug] old_action_probs: min={np.min(old_action_probs):.6f}, max={np.max(old_action_probs):.6f}")
        
        # GAE 계산
        advantages, returns = self.compute_gae(rewards, values, dones)
        # print(f"[Debug] advantages: min={np.min(advantages):.6f}, max={np.max(advantages):.6f}, has_nan={np.any(np.isnan(advantages))}")
        # print(f"[Debug] returns: min={np.min(returns):.6f}, max={np.max(returns):.6f}, has_nan={np.any(np.isnan(returns))}")
        
        # PPO 업데이트
        total_actor_loss = 0
        total_critic_loss = 0
        
        for epoch in range(4):  # PPO는 보통 여러 epoch으로로 학습
            # Actor 업데이트
            with tf.GradientTape() as tape:
                action_probs, _ = self.get_action_probs(states, action_masks)
                
                # 선택된 액션의 확률 추출
                batch_indices = tf.range(tf.shape(actions)[0])
                action_indices = tf.stack([batch_indices, actions[:, 0], actions[:, 1]], axis=1)
                new_action_probs = tf.gather_nd(action_probs, action_indices)
                
                # Ratio 계산
                ratio = new_action_probs / (old_action_probs + 1e-8)
                
                # KL divergence 계산 (수치적 안정성 강화)
                old_probs_safe = tf.clip_by_value(old_action_probs, 1e-8, 1.0)
                new_probs_safe = tf.clip_by_value(new_action_probs, 1e-8, 1.0)
                
                # print(f"[Debug] Epoch {epoch}: old_probs range={tf.reduce_min(old_probs_safe):.6f}~{tf.reduce_max(old_probs_safe):.6f}")
                # print(f"[Debug] Epoch {epoch}: new_probs range={tf.reduce_min(new_probs_safe):.6f}~{tf.reduce_max(new_probs_safe):.6f}")
                
                log_ratio = tf.math.log(old_probs_safe / new_probs_safe)
                # print(f"[Debug] Epoch {epoch}: log_ratio has_inf={tf.reduce_any(tf.math.is_inf(log_ratio))}, has_nan={tf.reduce_any(tf.math.is_nan(log_ratio))}")
                
                kl_div = tf.reduce_mean(old_probs_safe * log_ratio)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                
                # Entropy bonus
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=-1)
                entropy = tf.reduce_mean(entropy)
                
                # Actor loss
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - self.entropy_coef * entropy
                
                # Actor loss NaN 체크
                if tf.math.is_nan(actor_loss):
                    # print(f"[Warning] Actor loss is NaN! Setting to 0.")
                    actor_loss = tf.constant(0.0)
                # print(f"[Debug] Epoch {epoch}: Actor loss = {actor_loss:.6f}")
                
            # Actor gradients
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            if actor_grads is not None:
                actor_grads = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in actor_grads if grad is not None]
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            # Critic 업데이트
            with tf.GradientTape() as tape:
                current_values = tf.reshape(self.get_value(states), [-1])
                critic_loss = tf.reduce_mean(tf.square(returns - current_values))
                
                # Critic loss NaN 체크
                if tf.math.is_nan(critic_loss):
                    # print(f"[Warning] Critic loss is NaN! Setting to 0.")
                    critic_loss = tf.constant(0.0)
                # print(f"[Debug] Epoch {epoch}: Critic loss = {critic_loss:.6f}")
                
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            if critic_grads is not None:
                critic_grads = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in critic_grads if grad is not None]
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
            # Loss 축적 (NaN 안전하게)
            actor_loss_val = actor_loss.numpy()
            critic_loss_val = critic_loss.numpy()
            
            if not np.isnan(actor_loss_val):
                total_actor_loss += actor_loss_val
            # else:
            #     print(f"[Warning] Skipping NaN actor loss at epoch {epoch}")
                
            if not np.isnan(critic_loss_val):
                total_critic_loss += critic_loss_val
            # else:
            #     print(f"[Warning] Skipping NaN critic loss at epoch {epoch}")
            
            # KL divergence가 너무 크면 중단
            if kl_div.numpy() > self.target_kl:
                # print(f"Early stopping at epoch {epoch+1}, KL div: {kl_div.numpy():.6f}")
                break
            
        # 적응적 learning rate 조정 
        final_kl_div = kl_div.numpy()
        self.update_learning_rate(final_kl_div)  # 적응적 학습률 활성화
        
        # 버퍼 정리
        self.trajectory_buffer = []
        
        return (total_actor_loss / 4, total_critic_loss / 4, final_kl_div)
        
    def reset_learning_rate(self):
        """Learning rate를 초기값으로 리셋"""
        self.current_learning_rate = self.initial_learning_rate
        self.actor_optimizer.learning_rate.assign(self.current_learning_rate)
        self.critic_optimizer.learning_rate.assign(self.current_learning_rate)
        print(f"Learning rate reset to {self.current_learning_rate:.6f}")
        