import os
import numpy as np
import tensorflow as tf
import app.config as cfg
from app.action_type import ActionType
from tensorflow.keras import mixed_precision

from app.pending_buffer import PendingBuffer
from app.replay_buffer import ReplayBuffer
# from app.prioritized_replay_buffer import PrioritizedReplayBuffer
# from app.llm_agent import LLMAssistant
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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # 더 천천히 감소

        # Prioritized Experience Replay 사용
        # self.replay_buffer = PrioritizedReplayBuffer(
        #     capacity=800,  # 1500 -> 800 (OOM 완화)
        #     alpha=0.6,  # 우선순위 강도
        #     beta=0.4,   # IS weights 보정 강도 (초기값)
        #     beta_increment=0.00015  # 5000 ep 기준: 4000 ep에서 1.0 도달
        # )
        self.replay_buffer = ReplayBuffer(capacity=800)
        self.pending_buffer = PendingBuffer()
        
        # 학습률 스케줄링
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        
        # 성능 추적을 위한 변수들
        self.recent_rewards = []
        self.performance_window = 10
        # PER beta 스케줄 (에피소드 기반)
        # self.beta_start = 0.4
        # self.beta_end = 1.0
        # self.beta_target_episode = 4000  # 5000 ep의 80%
        self.train_step_cnt = 0
        # LLM 보조기 (config 기반)
        # self.llm_assistant = LLMAssistant()

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

    def get_action_q_values(self, state, action_mask):
        # 1. 입력 차원 확인 (Vehicle State 기준)
        v_state = state[0]
        is_single_input = (v_state.ndim == 2)

        # 2. 단일 입력이면 배치 차원 추가 (Batch=1)
        if is_single_input:
            state_input = [np.expand_dims(s, axis=0) for s in state]
            mask_input = np.expand_dims(action_mask, axis=0)
        else:
            state_input = state
            mask_input = action_mask

        # 3. 모델 예측 (Q-Value 계산)
        # mixed_precision 호환성을 위해 float32로 변환
        q_values = self.model(state_input, training=False) 
        q_values = tf.cast(q_values, dtype=tf.float32)

        # 4. 마스킹 처리 (불가능한 행동 제외)
        # mask가 0인 위치에 -1e9를 더해 Q값을 떨어뜨림 (Argmax에서 선택 안 되게)
        mask_tensor = tf.cast(mask_input, dtype=tf.float32)
        inf_mask = (1.0 - mask_tensor) * -1e9
        
        masked_q_values = q_values + inf_mask

        # 5. 단일 입력이었으면 결과의 배치 차원 제거
        if is_single_input:
            masked_q_values = tf.squeeze(masked_q_values, axis=0)

        return masked_q_values

    def get_action_info(self, vehicle_idx, action_idx):
        info = {}
        
        # 기본 인덱스 정보
        info['v_idx'] = int(vehicle_idx)
        info['action_idx'] = int(action_idx)

        # Action 해석
        # cfg.MAX_NUM_REQUEST (보통 12) 미만이면 승객 요청 처리
        if action_idx < cfg.MAX_NUM_REQUEST:
            # 1. 승객 처리 (일단 SERVE로 표시하거나 None으로 둠)
            # 환경(env)에서 차량 상태(승객 탑승 여부)를 보고 PICKUP/DROPOFF로 바꿔줍니다.
            info['type'] = "SERVE" 
            info['r_id'] = int(action_idx) # 요청 슬롯 인덱스
            
        else:
            # 2. 대기/거절 (마지막 인덱스)
            info['type'] = ActionType.REJECT
            info['r_id'] = None

        return info
    

    def act(self, state, action_mask):
        # 1. 탐험 (Exploration)
        if np.random.rand() <= self.epsilon:
            # 마스크가 1인(가능한) 행동 중에서 무작위 선택
            # action_mask shape: (Vehicle, Action) -> Flatten
            flat_mask = action_mask.flatten()
            available_indices = np.where(flat_mask == 1)[0]
            
            if len(available_indices) == 0:
                # 비상시 (이론상 없어야 함)
                return [0, 0, self.get_action_info(0, 0)]
            
            flat_action_idx = np.random.choice(available_indices)
        
        # 2. 활용 (Exploitation)
        else:
            # Q-Value 계산 (위에서 수정한 함수 사용)
            q_values = self.get_action_q_values(state, action_mask)
            
            # (Vehicle, Action) -> (V*A) Flatten
            flat_q_values = tf.reshape(q_values, [-1])
            
            # Argmax로 최대 Q값을 가진 인덱스 찾기 (Tensor)
            flat_action_tensor = tf.argmax(flat_q_values)

            # [핵심] Tensor -> CPU Int 변환 (JIT 오류 회피)
            try:
                flat_action_idx = int(flat_action_tensor.numpy())
            except AttributeError:
                flat_action_idx = int(flat_action_tensor)

        # 3. 순수 파이썬 연산으로 인덱스 분해
        num_actions = int(cfg.POSSIBLE_ACTION)
        
        action_idx = flat_action_idx % num_actions
        vehicle_idx = flat_action_idx // num_actions

        return [vehicle_idx, action_idx, self.get_action_info(vehicle_idx, action_idx)]

    # def act_with_llm(self, env, state, action_mask, priority="대기시간 최소화 > 시간창 준수 > 승차율", constraints="", task="recommend", dqn_action=""):
    #     """
    #     LLM 어시스트를 통한 액션 선택. 실패 시 기본 act()로 폴백.
    #     """
    #     return self.llm_assistant.act_with_llm(
    #         env=env,
    #         state=state,
    #         action_mask=action_mask,
    #         get_action_info_fn=self.get_action_info,
    #         fallback_fn=lambda: self.act(state, action_mask),
    #         priority=priority,
    #         constraints=constraints,
    #         task=task,
    #         dqn_action=dqn_action,
    #     )

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
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
        # progress = min(1.0, max(0.0, episode / float(self.beta_target_episode)))
        # beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        # if hasattr(self, 'replay_buffer') and hasattr(self.replay_buffer, 'set_beta'):
        #     self.replay_buffer.set_beta(beta)

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

    @tf.function(reduce_retracing=True)
    def _optimize_on_batch(
        self,
        s_v,
        s_r,
        s_rel,
        ns_v,
        ns_r,
        ns_rel,
        action_masks,
        actions,
        rewards,
        dones,
    ):
        """
        고정된 입력 signature로 학습 스텝을 컴파일해 tf.function retracing 경고를 완화.
        """
        # 입력 shape 고정
        s_v = tf.ensure_shape(s_v, (self.batch_size, cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM))
        s_r = tf.ensure_shape(s_r, (self.batch_size, cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM))
        s_rel = tf.ensure_shape(s_rel, (self.batch_size, cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM))
        ns_v = tf.ensure_shape(ns_v, (self.batch_size, cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM))
        ns_r = tf.ensure_shape(ns_r, (self.batch_size, cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM))
        ns_rel = tf.ensure_shape(ns_rel, (self.batch_size, cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM))

        action_masks = tf.ensure_shape(
            tf.cast(action_masks, tf.float32),
            (self.batch_size, cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST + 1),
        )
        actions = tf.ensure_shape(tf.cast(actions, tf.int32), (self.batch_size, 2))
        rewards = tf.ensure_shape(tf.cast(rewards, tf.float32), (self.batch_size,))
        dones = tf.ensure_shape(tf.cast(dones, tf.float32), (self.batch_size,))

        states = [s_v, s_r, s_rel]
        next_states = [ns_v, ns_r, ns_rel]

        # Target 계산
        next_q_main = self.model(next_states, training=False)
        inf_mask = (1.0 - action_masks) * -1e9
        masked_next_q = next_q_main + inf_mask

        bsz = tf.shape(next_q_main)[0]
        flat_next_q = tf.reshape(masked_next_q, [bsz, -1])
        max_action_indices = tf.argmax(flat_next_q, axis=1, output_type=tf.int32)

        next_q_target = self.target_model(next_states, training=False)
        flat_target_q = tf.reshape(next_q_target, [bsz, -1])

        batch_indices = tf.range(bsz, dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, max_action_indices], axis=1)
        max_next_q_val = tf.gather_nd(flat_target_q, gather_indices)

        target_q = rewards + (1.0 - dones) * self.gamma * max_next_q_val
        target_q = tf.stop_gradient(target_q)

        # Gradient Descent
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)

            v_indices = actions[:, 0]
            a_indices = actions[:, 1]

            shape_q = tf.shape(q_values)
            num_acts = shape_q[2]

            flat_act_indices = v_indices * num_acts + a_indices
            flat_q_values = tf.reshape(q_values, [bsz, -1])
            gather_indices_main = tf.stack([batch_indices, flat_act_indices], axis=1)

            main_q_val = tf.gather_nd(flat_q_values, gather_indices_main)

            huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)(target_q, main_q_val)
            loss = tf.reduce_mean(huber_loss)

            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss

        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)

        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        try:
            batch = self.replay_buffer.sample(self.batch_size)
            if batch is None:
                return None
        except Exception as e:
            print(f"[DDQN Error] 샘플링 실패: {e}")
            return None

        try:
            def safe_squeeze(data_list):
                arr = np.array(data_list, dtype=np.float32)
                if arr.ndim > 1 and arr.shape[1] == 1:
                    arr = np.squeeze(arr, axis=1)
                return arr

            if any(b[0] is None for b in batch):
                return 0.0

            s_v = safe_squeeze([b[0][0] for b in batch])
            s_r = safe_squeeze([b[0][1] for b in batch])
            s_rel = safe_squeeze([b[0][2] for b in batch])
            states = [s_v, s_r, s_rel]

            ns_v = safe_squeeze([b[3][0] for b in batch])
            ns_r = safe_squeeze([b[3][1] for b in batch])
            ns_rel = safe_squeeze([b[3][2] for b in batch])
            next_states = [ns_v, ns_r, ns_rel]

            action_masks = np.array([b[5]["m"] for b in batch], dtype=np.float32)

            action_list = []
            for item in batch:
                raw_action = item[1]
                if isinstance(raw_action, (list, tuple, np.ndarray)) and len(raw_action) >= 2:
                    action_list.append([int(raw_action[0]), int(raw_action[1])])
                else:
                    action_list.append([0, 0])
            actions = np.array(action_list, dtype=np.int32)

            rewards = np.array([b[2] for b in batch], dtype=np.float32)
            dones = np.array([b[4] for b in batch], dtype=np.float32)

            loss = self._optimize_on_batch(
                tf.convert_to_tensor(s_v, dtype=tf.float32),
                tf.convert_to_tensor(s_r, dtype=tf.float32),
                tf.convert_to_tensor(s_rel, dtype=tf.float32),
                tf.convert_to_tensor(ns_v, dtype=tf.float32),
                tf.convert_to_tensor(ns_r, dtype=tf.float32),
                tf.convert_to_tensor(ns_rel, dtype=tf.float32),
                tf.convert_to_tensor(action_masks, dtype=tf.float32),
                tf.convert_to_tensor(actions, dtype=tf.int32),
                tf.convert_to_tensor(rewards, dtype=tf.float32),
                tf.convert_to_tensor(dones, dtype=tf.float32),
            )

            # 후처리
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.train_step_cnt += 1
            if self.train_step_cnt % self.update_target_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

            return float(loss.numpy())

        except Exception as e:
            print(f"[DDQN Error] 학습 중 오류 발생: {e}")
            return 0.0
    


    # def train(self):
    #     if len(self.replay_buffer) < self.batch_size:
    #         return None

    #     try:
    #         sample_result = self.replay_buffer.sample(self.batch_size)
    #         if sample_result is None:
    #             return None
    #         PrioritizedReplayBuffer.sample returns (batch, idxs, is_weights)
    #         batch, idxs, is_weights = sample_result
    #     except Exception as e:
    #         print(f"[DDQN Error] 샘플링 실패: {e}")
    #         return None

    #     try:
    #         # --- [Step 1] 데이터 전처리 (방어적 코딩) ---
    #         def safe_squeeze(data_list):
    #             arr = np.array(data_list, dtype=np.float32)
    #             if arr.ndim > 1 and arr.shape[1] == 1:
    #                 arr = np.squeeze(arr, axis=1)
    #             return arr

    #         # State가 None인 경우 방지
    #         if any(b[0] is None for b in batch):
    #             # print(f"[DDQN Error] None State detected inside batch. Skipping.")
    #             return 0.0

    #         s_v = safe_squeeze([b[0][0] for b in batch])
    #         s_r = safe_squeeze([b[0][1] for b in batch])
    #         s_rel = safe_squeeze([b[0][2] for b in batch])
    #         states = [s_v, s_r, s_rel]

    #         ns_v = safe_squeeze([b[3][0] for b in batch])
    #         ns_r = safe_squeeze([b[3][1] for b in batch])
    #         ns_rel = safe_squeeze([b[3][2] for b in batch])
    #         next_states = [ns_v, ns_r, ns_rel]
            
    #         action_masks = np.array([b[5]["m"] for b in batch], dtype=np.float32)
            
    #         # =========================================================
    #         # [핵심 수정] Action이 int로 오염되었을 경우 처리
    #         # =========================================================
    #         action_list = []
    #         for i, item in enumerate(batch):
    #             raw_action = item[1] # b[1]
                
    #             # 정상 케이스: 리스트/튜플/배열이고 길이가 2 이상
    #             if isinstance(raw_action, (list, tuple, np.ndarray)) and len(raw_action) >= 2:
    #                 action_list.append([int(raw_action[0]), int(raw_action[1])])
    #             else:
    #                 # 오염된 케이스: int형이거나 잘못된 포맷
    #                 # print(f"[DDQN Error] Batch[{i}] 잘못된 Action 포맷: {raw_action} (Type: {type(raw_action)}) -> [0, 0]으로 대체")
    #                 action_list.append([0, 0]) # 더미 액션으로 대체하여 학습 계속 진행

    #         actions = np.array(action_list, dtype=np.int32)
    #         # =========================================================

    #         rewards = np.array([b[2] for b in batch], dtype=np.float32)
    #         dones = np.array([b[4] for b in batch], dtype=np.float32)
    #         is_weights_tensor = tf.convert_to_tensor(is_weights, dtype=tf.float32)

    #         # --- [Step 2] 학습 진행 ---
    #         # 2-1. Target 계산
    #         next_q_main = self.model(next_states, training=False)
    #         inf_mask = (1.0 - action_masks) * -1e9
    #         masked_next_q = next_q_main + inf_mask
            
    #         batch_size = tf.shape(next_q_main)[0]
    #         flat_next_q = tf.reshape(masked_next_q, [batch_size, -1])
    #         max_action_indices = tf.argmax(flat_next_q, axis=1, output_type=tf.int32)

    #         next_q_target = self.target_model(next_states, training=False)
    #         flat_target_q = tf.reshape(next_q_target, [batch_size, -1])
            
    #         batch_indices = tf.range(batch_size, dtype=tf.int32)
    #         gather_indices = tf.stack([batch_indices, max_action_indices], axis=1)
    #         max_next_q_val = tf.gather_nd(flat_target_q, gather_indices)

    #         target_q = rewards + (1.0 - dones) * self.gamma * max_next_q_val
    #         target_q = tf.stop_gradient(target_q)

    #         # 2-2. Gradient Descent
    #         with tf.GradientTape() as tape:
    #             q_values = self.model(states, training=True)
                
    #             v_indices = actions[:, 0]
    #             a_indices = actions[:, 1]
                
    #             shape_q = tf.shape(q_values)
    #             num_acts = shape_q[2]
                
    #             flat_act_indices = v_indices * num_acts + a_indices
    #             flat_q_values = tf.reshape(q_values, [batch_size, -1])
    #             gather_indices_main = tf.stack([batch_indices, flat_act_indices], axis=1)
                
    #             main_q_val = tf.gather_nd(flat_q_values, gather_indices_main)

    #             td_errors = target_q - main_q_val
    #             huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)(target_q, main_q_val)
    #             weighted_loss = huber_loss * is_weights_tensor
    #             loss = tf.reduce_mean(weighted_loss)
                
    #             if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
    #                 scaled_loss = self.optimizer.get_scaled_loss(loss)
    #             else:
    #                 scaled_loss = loss

    #         # 2-3. Apply Gradients
    #         if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
    #             grads = tape.gradient(scaled_loss, self.model.trainable_variables)
    #             grads = self.optimizer.get_unscaled_gradients(grads)
    #         else:
    #             grads = tape.gradient(loss, self.model.trainable_variables)

    #         grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    #         self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    #         # --- [Step 3] 후처리 ---
    #         td_errors_np = tf.abs(td_errors).numpy().astype(np.float32)
    #         self.replay_buffer.update_priorities(idxs, td_errors_np)
            
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= self.epsilon_decay
                
    #         self.train_step_cnt += 1
    #         if self.train_step_cnt % self.update_target_freq == 0:
    #             self.target_model.set_weights(self.model.get_weights())
                
    #         return float(loss.numpy())

    #     except Exception as e:
    #         # 에러 발생 시 건너뜀 (프로그램 종료 방지)
    #         print(f"[DDQN Error] 학습 중 오류 발생: {e}")
    #         return 0.0