import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from app.pending_buffer import PendingBuffer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, RepeatVector


class PPOAgent:
    def __init__(self, hidden_dim, batch_size, actor_learning_rate, critic_learning_rate):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Learning rates 분리 + Linear decay 설정
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_initial_lr = actor_learning_rate
        self.critic_initial_lr = critic_learning_rate
        # 선형 스케줄 최종 학습률
        self.lr_final = 1e-4
        self.lr_decay_steps = 30000  # 학습 진행 시 선형으로 감소
        self.train_step_count = 0

        # PPO hyperparameters
        self.clip_ratio = 0.13
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.022
        self.value_coef = 0.65
        self.max_grad_norm = 0.3
        self.target_kl = 0.01

        # Learning rate scheduling (KL-based) - critic에만 적용
        self.initial_learning_rate = self.critic_learning_rate
        self.current_learning_rate = self.critic_learning_rate

        # On-policy buffers
        self.episode_buffer = []
        # 업데이트 주기: 배치 크기 기반으로 너무 커지지 않도록 제한
        self.update_frequency = 256

        # Build actor and critic networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # Optimizers (actor/critic 분리 학습률)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.current_learning_rate)

        # Experience buffer
        self.pending_buffer = PendingBuffer()
        self.trajectory_buffer = []

        # Monitoring
        self.last_kl = 0.0
        self.last_train_stats = {}

        # Reward normalization (Welford)
        self.reward_mean = 0.0
        self.reward_M2 = 0.0
        self.reward_count = 0

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save_model(self, file_path):
        actor_path = file_path.replace(".h5", "_actor.h5")
        critic_path = file_path.replace(".h5", "_critic.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Model weights saved at {actor_path} and {critic_path}")

    def load_model(self, file_path):
        actor_path = file_path.replace(".h5", "_actor.h5")
        critic_path = file_path.replace(".h5", "_critic.h5")
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print(f"Model weights loaded from {actor_path} and {critic_path}")
        else:
            print(f"No model weights found at {actor_path} or {critic_path}")

    # ---------------------------
    # Networks
    # ---------------------------
    def build_shared_network(self):
        """공통 feature extraction network 구축"""
        vehicle_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM), name="vehicle_input")
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name="request_input")
        relation_input = Input(
            shape=(cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM),
            name="relation_input",
        )

        v_embed = TimeDistributed(Dense(self.hidden_dim, activation="relu"))(vehicle_input)
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation="relu"))(request_input)

        # KerasTensor 안전 처리를 위해 Lambda 사용
        v_expand = Lambda(lambda x: tf.expand_dims(x, axis=2))(v_embed)  # (B, V, 1, H)
        r_expand = Lambda(lambda x: tf.expand_dims(x, axis=1))(r_embed)  # (B, 1, R, H)

        v_tiled = Lambda(lambda x: tf.tile(x, [1, 1, cfg.MAX_NUM_REQUEST, 1]))(v_expand)  # (B, V, R, H)
        r_tiled = Lambda(lambda x: tf.tile(x, [1, cfg.MAX_NUM_VEHICLES, 1, 1]))(r_expand)  # (B, V, R, H)

        pair_embed = Concatenate(axis=-1)([v_tiled, r_tiled, relation_input])  # (B, V, R, 2H + Drel)
        return [vehicle_input, request_input, relation_input], pair_embed, v_embed, r_embed

    def build_actor(self):
        """Policy network (Actor) 구축"""
        inputs, pair_embed, v_embed, r_embed = self.build_shared_network()

        match_logits = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation="relu")))(pair_embed)
        match_logits = TimeDistributed(TimeDistributed(Dense(1, kernel_initializer="glorot_uniform", bias_initializer="zeros")))(
            match_logits
        )
        match_logits = Lambda(lambda x: tf.squeeze(x, axis=-1))(match_logits)  # (B, V, R)

        r_summary = Lambda(lambda x: tf.reduce_mean(x, axis=1))(r_embed)  # (B, H)
        r_summary = RepeatVector(cfg.MAX_NUM_VEHICLES)(r_summary)  # (B, V, H)
        reject_context = Concatenate(axis=-1)([v_embed, r_summary])  # (B, V, 2H)

        reject_logits = TimeDistributed(Dense(self.hidden_dim, activation="relu"))(reject_context)
        reject_logits = TimeDistributed(Dense(1, kernel_initializer="glorot_uniform", bias_initializer="zeros"))(reject_logits)  # (B, V, 1)

        action_logits = Concatenate(axis=-1)([match_logits, reject_logits])  # (B, V, R+1)
        return Model(inputs=inputs, outputs=action_logits)

    def build_critic(self):
        """Value network (Critic) 구축"""
        inputs, pair_embed, v_embed, r_embed = self.build_shared_network()

        global_v = Lambda(lambda x: tf.reduce_mean(x, axis=1))(v_embed)  # (B, H)
        global_r = Lambda(lambda x: tf.reduce_mean(x, axis=1))(r_embed)  # (B, H)
        global_state = Concatenate(axis=-1)([global_v, global_r])  # (B, 2H)

        value = Dense(self.hidden_dim, activation="relu")(global_state)
        value = Dense(self.hidden_dim, activation="relu")(value)
        value = Dense(1, kernel_initializer="glorot_uniform", bias_initializer="zeros")(value)  # (B, 1)

        # (선택) 크리틱 폭주 방지 - 너무 강하면 학습 신호 죽을 수 있어 범위를 완화
        value = Lambda(lambda x: tf.clip_by_value(x, -50.0, 50.0))(value)
        return Model(inputs=inputs, outputs=value)

    # ---------------------------
    # Action / Value
    # ---------------------------
    def get_action_probs(self, state, action_mask):
        # 1. 입력 차원 확인 (Vehicle State 기준)
        v_state = state[0]
        is_single_input = (v_state.ndim == 2)

        # 2. 단일 입력이면 배치 차원 추가
        if is_single_input:
            state_input = [np.expand_dims(s, axis=0) for s in state]
            mask_input = np.expand_dims(action_mask, axis=0)
        else:
            state_input = state
            mask_input = action_mask

        # 3. 모델 예측 (Logits 계산)
        # 모델 출력이 float16일 수 있으므로, 안전하게 float32로 캐스팅
        logits = self.actor(state_input, training=False) 
        logits = tf.cast(logits, dtype=tf.float32)

        # 4. 마스킹 처리 (float32로 통일)
        mask_tensor = tf.cast(mask_input, dtype=tf.float32)
        
        # 마스크가 0인(불가능한) 행동에 -1e9를 더함 (Softmax 결과 0 유도)
        inf_mask = (1.0 - mask_tensor) * -1e9
        
        # 이제 둘 다 float32이므로 안전하게 더하기 가능
        masked_logits = logits + inf_mask

        # 5. Softmax 계산 (float32 권장)
        action_probs = tf.nn.softmax(masked_logits, axis=-1)

        # # 6. 단일 입력이었으면 결과의 배치 차원 제거
        # if is_single_input:
        #     action_probs = tf.squeeze(action_probs, axis=0)
        #     masked_logits = tf.squeeze(masked_logits, axis=0)

        return action_probs, masked_logits

    def act(self, state, action_mask):
        action_probs, logits = self.get_action_probs(state, action_mask)

        # 유효 액션이 없으면 예외 처리
        valid_actions = tf.where(action_mask == 1)
        if tf.shape(valid_actions)[0] == 0:
            vehicle_idx = 0
            action_idx = cfg.POSSIBLE_ACTION - 1  # REJECT
            action_prob_value = 1.0
            logp_value = 0.0
        else:
            masked_probs = tf.where(action_mask == 1, action_probs, tf.constant(0.0, dtype=action_probs.dtype))
            masked_probs_flat = tf.reshape(masked_probs, (-1,))

            non_zero_indices = tf.where(masked_probs_flat > 0)[:, 0]
            if tf.shape(non_zero_indices)[0] == 0:
                # 유효 액션 중 균등
                valid_indices_flat = tf.reshape(tf.where(action_mask == 1), (-1,))
                vehicle_indices = valid_indices_flat[::2]
                action_indices = valid_indices_flat[1::2]
                flat_valid_indices = vehicle_indices * cfg.POSSIBLE_ACTION + action_indices

                random_idx = tf.random.uniform([], 0, tf.shape(flat_valid_indices)[0], dtype=tf.int32)
                flat_action_idx = flat_valid_indices[random_idx]

                vehicle_idx = int(flat_action_idx // cfg.POSSIBLE_ACTION)
                action_idx = int(flat_action_idx % cfg.POSSIBLE_ACTION)
                action_prob_value = 1.0 / float(tf.shape(flat_valid_indices)[0])
                logp_value = float(np.log(max(action_prob_value, 1e-8)))
            else:
                non_zero_probs = tf.gather(masked_probs_flat, non_zero_indices)
                non_zero_probs = non_zero_probs / tf.reduce_sum(non_zero_probs)

                sampled_idx = tf.random.categorical(tf.math.log(non_zero_probs[tf.newaxis, :]), 1)[0, 0]
                flat_action_idx = non_zero_indices[sampled_idx]

                flat_action_idx_int64 = tf.cast(flat_action_idx, dtype=tf.int64)
                possible_action_int64 = tf.constant(cfg.POSSIBLE_ACTION, dtype=tf.int64)
                vehicle_idx_int64 = tf.math.floordiv(flat_action_idx_int64, possible_action_int64)
                action_idx_int64 = flat_action_idx_int64 - vehicle_idx_int64 * possible_action_int64

                vehicle_idx = int(vehicle_idx_int64.numpy())
                action_idx = int(action_idx_int64.numpy())

                action_prob_value = float(action_probs[0][vehicle_idx][action_idx].numpy())
                logp_value = float(np.log(max(action_prob_value, 1e-8)))

        info = {
            "mode": "sample",
            "action_prob": action_prob_value,
            "logp": logp_value,          # PPO 학습에서 ratio 계산 안정성을 위해 저장
            "logits": logits.numpy(),
        }
        return [vehicle_idx, action_idx, info]

    def get_value(self, state):
        return self.critic(state, training=False)

    # ---------------------------
    # Buffering
    # ---------------------------
    def remember(self, transition):
        self.episode_buffer.append(transition)

    def pending(self, transition):
        action = transition[1]
        action_id = action[2]["id"]
        self.pending_buffer.add(action_id, transition)

    def confirm_and_remember(self, action_id, reward):
        transition = self.pending_buffer.confirm(action_id, reward)
        if transition is not None:
            self.remember(transition)

    def finish_episode(self):
        if len(self.episode_buffer) > 0:
            self.trajectory_buffer.extend(self.episode_buffer)
            self.episode_buffer = []

    def should_train(self):
        # 에피소드 버퍼 + 대기 중 trajectory 합산하여 학습 여부 판단
        return (len(self.episode_buffer) + len(self.trajectory_buffer)) >= self.update_frequency

    # ---------------------------
    # Reward normalization (Welford)
    # ---------------------------
    def _update_reward_stats(self, rewards: np.ndarray):
        for r in rewards:
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            self.reward_M2 += delta * delta2

    def _reward_std(self):
        if self.reward_count < 2:
            return 1.0
        var = self.reward_M2 / (self.reward_count - 1)
        return float(np.sqrt(max(var, 1e-8)))

    # ---------------------------
    # KL-based LR adaptation
    # ---------------------------
    def update_learning_rate(self, kl_div):
        min_lr = 1e-8
        max_lr = 1e-3

        if self.current_learning_rate > max_lr or self.current_learning_rate < min_lr:
            print(f"[Warning] Learning rate {self.current_learning_rate:.2e} abnormal. Reset -> {self.initial_learning_rate:.2e}")
            self.current_learning_rate = self.initial_learning_rate

        if kl_div < 1e-8:
            return

        if kl_div > self.target_kl * 1.5:
            self.current_learning_rate *= 0.8
        elif kl_div < self.target_kl / 1.5:
            self.current_learning_rate *= 1.05

        self.current_learning_rate = max(min_lr, min(max_lr, self.current_learning_rate))
        # Actor lr는 고정(3e-6) 유지
        self.critic_optimizer.learning_rate.assign(self.current_learning_rate)

    def _linear_lr(self, step, initial_lr):
        """선형 스케줄: initial -> lr_final까지 lr_decay_steps 동안 감소"""
        if self.lr_decay_steps <= 0:
            return initial_lr
        progress = min(1.0, step / float(self.lr_decay_steps))
        return initial_lr + (self.lr_final - initial_lr) * progress

    def reset_learning_rate(self):
        self.current_learning_rate = self.initial_learning_rate
        # Actor lr 고정 유지
        self.critic_optimizer.learning_rate.assign(self.current_learning_rate)
        print(f"Learning rate reset to {self.current_learning_rate:.6f}")

    # ---------------------------
    # GAE (FIXED: use next_values bootstrap properly)
    # ---------------------------
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Standard GAE:
          delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
          A_t = delta_t + gamma*lambda*(1-done)*A_{t+1}
        """
        rewards = np.nan_to_num(np.asarray(rewards, dtype=np.float32), nan=0.0)
        values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
        next_values = np.nan_to_num(np.asarray(next_values, dtype=np.float32), nan=0.0)
        dones = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values

        # Advantage normalization (standard)
        if len(advantages) > 1:
            adv_mean = float(np.mean(advantages))
            adv_std = float(np.std(advantages))
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std
            else:
                advantages = advantages - adv_mean

        return advantages.astype(np.float32), returns.astype(np.float32)

    # ---------------------------
    # Training (FIXED: minibatch PPO, no extra ratio pre-clip, target_kl early stop, LR adapt)
    # ---------------------------
    def train(self):
        # 에피소드 버퍼에 쌓인 데이터를 우선 trajectory_buffer로 이동
        if len(self.episode_buffer) > 0:
            self.trajectory_buffer.extend(self.episode_buffer)
            self.episode_buffer = []

        if len(self.trajectory_buffer) < self.update_frequency:
            # 학습 스킵 시에도 상태 기록 (버퍼 크기 확인용)
            current_size = len(self.trajectory_buffer)
            total_buffer = len(self.episode_buffer) + current_size
            self.last_train_stats = {
                "buffer_size": total_buffer,
                "trajectory_buffer_size": current_size,
                "train_updates": 0,
                "avg_kl": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "skipped_actor": 0,
                "skipped_critic": 0,
            }
            return None

        batch = self.trajectory_buffer[:]

        # # Extract arrays
        # states = [
        #     np.array([b[0][0] for b in batch], dtype=np.float32),  # vehicle states
        #     np.array([b[0][1] for b in batch], dtype=np.float32),  # request states
        #     np.array([b[0][2] for b in batch], dtype=np.float32),  # relation states
        # ]
        # next_states = [
        #     np.array([b[3][0] for b in batch], dtype=np.float32),
        #     np.array([b[3][1] for b in batch], dtype=np.float32),
        #     np.array([b[3][2] for b in batch], dtype=np.float32),
        # ]

        try:
            def safe_stack_and_squeeze(data_list):
                """리스트를 적재하고 (Batch, 1, ...) 형태인 경우 1차원을 제거"""
                arr = np.array(data_list, dtype=np.float32)
                # 차원이 2 이상이고 두 번째 차원(axis=1)이 1이라면 제거
                if arr.ndim > 1 and arr.shape[1] == 1:
                    arr = np.squeeze(arr, axis=1)
                return arr

            # Current States
            s_v = safe_stack_and_squeeze([b[0][0] for b in batch])   # Vehicle
            s_r = safe_stack_and_squeeze([b[0][1] for b in batch])   # Request
            s_rel = safe_stack_and_squeeze([b[0][2] for b in batch]) # Relation
            states = [s_v, s_r, s_rel]

            # Next States
            ns_v = safe_stack_and_squeeze([b[3][0] for b in batch])
            ns_r = safe_stack_and_squeeze([b[3][1] for b in batch])
            ns_rel = safe_stack_and_squeeze([b[3][2] for b in batch])
            next_states = [ns_v, ns_r, ns_rel]

        except Exception as e:
            print(f"\n[Critical Error] 데이터 전처리 중 오류 발생! 배치를 건너뜁니다.")
            print(f"Error Details: {e}")
            # 디버깅용 Shape 출력
            try:
                tmp_v = np.array([b[0][0] for b in batch])
                print(f"Debug Info - Raw Vehicle Shape: {tmp_v.shape}")
            except:
                pass
            
            # 오류가 발생한 데이터는 학습에 사용할 수 없으므로 버퍼를 비워 무한 루프 방지
            self.trajectory_buffer = []
            return None

        self.trajectory_buffer = []
        buffer_size_used = len(batch)

        actions = np.array([[b[1][0], b[1][1]] for b in batch], dtype=np.int32)  # (N,2)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        action_masks = np.array([b[5]["m"] for b in batch], dtype=np.int32)

        # Prefer stored old logp if present, else fallback to log(prob)
        old_logps = []
        for b in batch:
            info = b[1][2]
            if "logp" in info and np.isfinite(info["logp"]):
                old_logps.append(info["logp"])
            else:
                p = float(info.get("action_prob", 1e-8))
                old_logps.append(np.log(max(p, 1e-8)))
        old_logps = np.array(old_logps, dtype=np.float32)

        # Reward normalization (apply for real)
        self._update_reward_stats(rewards)
        r_std = self._reward_std()
        rewards_norm = (rewards - float(self.reward_mean)) / (r_std + 1e-8)

        # Values
        values = tf.reshape(self.get_value(states), [-1]).numpy().astype(np.float32)
        next_values = tf.reshape(self.get_value(next_states), [-1]).numpy().astype(np.float32)

        # GAE
        advantages, returns = self.compute_gae(rewards_norm, values, next_values, dones)

        # PPO updates
        K = 6  # epochs
        N = len(rewards)
        B = min(self.batch_size, N)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_batches = 0
        n_updates = 0
        skipped_actor = 0
        skipped_critic = 0

        early_stop = False

        # Explained variance (사전 계산: 업데이트 전 value 예측 기준)
        var_ret = np.var(returns)
        explained_variance = 0.0
        if var_ret > 1e-8:
            explained_variance = 1.0 - np.var(returns - values) / (var_ret + 1e-8)

        for epoch in range(K):
            idx = np.random.permutation(N)

            for start in range(0, N, B):
                mb_idx = idx[start : start + B]
                if len(mb_idx) == 0:
                    continue

                mb_states = [s[mb_idx] for s in states]
                mb_actions = actions[mb_idx]
                mb_masks = action_masks[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]
                mb_old_logp = old_logps[mb_idx]

                mb_adv_tf = tf.convert_to_tensor(mb_adv, dtype=tf.float32)
                mb_ret_tf = tf.convert_to_tensor(mb_ret, dtype=tf.float32)
                mb_old_logp_tf = tf.convert_to_tensor(mb_old_logp, dtype=tf.float32)

                # ---------------- Actor update ----------------
                with tf.GradientTape() as tape:
                    action_probs, _ = self.get_action_probs(mb_states, mb_masks)
                    action_probs_fp32 = tf.cast(action_probs, tf.float32)

                    batch_indices = tf.range(tf.shape(mb_actions)[0], dtype=tf.int32)
                    action_indices = tf.stack([batch_indices, mb_actions[:, 0], mb_actions[:, 1]], axis=1)
                    new_probs = tf.gather_nd(action_probs_fp32, action_indices)
                    new_probs = tf.clip_by_value(new_probs, 1e-8, 1.0)

                    logp_new = tf.math.log(new_probs)
                    logp_old = tf.cast(mb_old_logp_tf, dtype=logp_new.dtype)

                    ratio = tf.exp(logp_new - logp_old)  # NO extra pre-clip

                    surr1 = ratio * mb_adv_tf
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv_tf
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                    # Entropy (invalid probs are ~0 due to -1e9 masking)
                    entropy = -tf.reduce_sum(action_probs_fp32 * tf.math.log(action_probs_fp32 + 1e-8), axis=-1)
                    entropy = tf.reduce_mean(entropy)

                    actor_loss = policy_loss - self.entropy_coef * entropy

                    # Approx KL for monitoring / early stop
                    approx_kl = tf.reduce_mean(logp_old - logp_new)
                    # Clip fraction: 비율이 클리핑 범위를 벗어난 비율
                    clip_fraction = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_ratio, tf.float32))

                # NaN/Inf 방지: 비정상 손실이면 스킵
                if not tf.reduce_all(tf.math.is_finite(actor_loss)):
                    # print("[Warn] skip actor update due to non-finite loss")
                    skipped_actor += 1
                    continue

                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                actor_grads = [g for g in actor_grads if g is not None]
                actor_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in actor_grads]
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                # ---------------- Critic update ----------------
                with tf.GradientTape() as tape:
                    v_pred = tf.cast(tf.reshape(self.get_value(mb_states), [-1]), tf.float32)
                    diff = mb_ret_tf - v_pred
                    huber_delta = 1.0
                    huber = tf.where(
                        tf.abs(diff) < huber_delta,
                        0.5 * tf.square(diff),
                        huber_delta * (tf.abs(diff) - 0.5 * huber_delta),
                    )
                    critic_loss = tf.reduce_mean(huber) * self.value_coef
                if not tf.reduce_all(tf.math.is_finite(critic_loss)):
                    # print("[Warn] skip critic update due to non-finite loss")
                    skipped_critic += 1
                    continue

                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                critic_grads = [g for g in critic_grads if g is not None]
                critic_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in critic_grads]
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                # Accumulate
                total_actor_loss += float(actor_loss.numpy())
                total_critic_loss += float(critic_loss.numpy())
                kl_val = float(approx_kl.numpy())
                total_kl += kl_val
                total_entropy += float(entropy.numpy())
                total_clip_frac += float(clip_fraction.numpy())
                total_batches += 1
                n_updates += 1

                # Early stopping on KL (standard PPO safety)
                if kl_val > self.target_kl * 1.5:
                    early_stop = True
                    break

            if early_stop:
                break

        # Stats
        if n_updates == 0:
            self.last_kl = 0.0
            self.last_train_stats = {
                "buffer_size": buffer_size_used,
                "train_updates": 0,
                "avg_kl": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": float(explained_variance),
                "skipped_actor": skipped_actor,
                "skipped_critic": skipped_critic,
            }
            return None

        avg_actor_loss = total_actor_loss / n_updates
        avg_critic_loss = total_critic_loss / n_updates
        avg_kl = total_kl / n_updates
        avg_entropy = total_entropy / max(total_batches, 1)
        avg_clip_frac = total_clip_frac / max(total_batches, 1)
        self.last_kl = avg_kl

        # Apply KL-based LR adaptation (now actually used)
        self.update_learning_rate(avg_kl)

        # Save last train stats for logging
        self.last_train_stats = {
            "buffer_size": buffer_size_used,
            "train_updates": n_updates,
            "avg_kl": float(avg_kl),
            "actor_loss": float(avg_actor_loss),
            "critic_loss": float(avg_critic_loss),
            "entropy": float(avg_entropy),
            "clip_fraction": float(avg_clip_frac),
            "explained_variance": float(explained_variance),
            "skipped_actor": skipped_actor,
            "skipped_critic": skipped_critic,
        }

        # 선형 스케줄로 actor/critic lr 감소
        self.train_step_count += 1
        new_actor_lr = self._linear_lr(self.train_step_count, self.actor_initial_lr)
        new_critic_lr = self._linear_lr(self.train_step_count, self.critic_initial_lr)
        self.actor_optimizer.learning_rate.assign(new_actor_lr)
        self.current_learning_rate = new_critic_lr
        self.critic_optimizer.learning_rate.assign(new_critic_lr)

        return (avg_actor_loss, avg_critic_loss)
