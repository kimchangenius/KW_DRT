import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from app.pending_buffer import PendingBuffer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, RepeatVector


class MAPPOAgent:
    def __init__(self, hidden_dim, batch_size, learning_rate):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_agents = cfg.MAX_NUM_VEHICLES  # 차량 수 = 에이전트 수
        
        # MAPPO specific hyperparameters (Loss 안정성 향상)
        self.clip_ratio = 0.1  # 0.2 -> 0.1 (더 보수적인 클리핑)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.005  # 0.01 -> 0.005 (더 작은 엔트로피)
        self.value_coef = 0.5
        self.max_grad_norm = 0.3  # 0.5 -> 0.3 (더 작은 그래디언트 클리핑)
        self.target_kl = 0.1  # 0.05 -> 0.1 (더 완화)
        
        # Learning rate with scheduling
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.lr_decay_rate = 0.999  # 0.998 -> 0.999 (더 느린 감소)
        self.min_lr = learning_rate * 0.5  # 0.8 -> 0.5 (더 낮은 최소값)
        
        # Episode buffer for multi-agent
        self.episode_buffer = []
        self.update_frequency = 15  # 10 -> 15 (더 큰 배치로 안정성 향상)
        
        # Build decentralized actor networks (각 에이전트마다 독립적인 actor)
        self.actors = [self.build_actor(agent_id) for agent_id in range(self.num_agents)]
        
        # Build centralized critic network (모든 에이전트가 공유하는 critic)
        self.critic = self.build_centralized_critic()
        
        # Optimizers
        self.actor_optimizers = [
            tf.keras.optimizers.Adam(learning_rate=self.current_learning_rate) 
            for _ in range(self.num_agents)
        ]
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.current_learning_rate)
        
        # Experience buffer
        self.pending_buffer = PendingBuffer()
        self.trajectory_buffer = []
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
    def save_model(self, file_path):
        # 각 에이전트의 actor 저장
        for i in range(self.num_agents):
            actor_path = file_path.replace('.h5', f'_actor_{i}.h5')
            self.actors[i].save_weights(actor_path)
            
        # Centralized critic 저장
        critic_path = file_path.replace('.h5', '_critic.h5')
        self.critic.save_weights(critic_path)
        print(f"MAPPO model weights saved")
        
    def load_model(self, file_path):
        # 각 에이전트의 actor 로드
        all_exist = True
        for i in range(self.num_agents):
            actor_path = file_path.replace('.h5', f'_actor_{i}.h5')
            if os.path.exists(actor_path):
                self.actors[i].load_weights(actor_path)
            else:
                all_exist = False
                
        # Centralized critic 로드
        critic_path = file_path.replace('.h5', '_critic.h5')
        if os.path.exists(critic_path):
            self.critic.load_weights(critic_path)
        else:
            all_exist = False
            
        if all_exist:
            print(f"MAPPO model weights loaded")
        else:
            print(f"Some MAPPO model weights not found")
            
    def build_actor(self, agent_id):
        """각 에이전트의 독립적인 Actor network 구축"""
        # 단일 차량의 상태 + 요청 정보 + 관계 정보
        vehicle_input = Input(shape=(cfg.VEHICLE_INPUT_DIM,), name=f"vehicle_input_{agent_id}")
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name=f"request_input_{agent_id}")
        relation_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM), name=f"relation_input_{agent_id}")
        
        # 차량 임베딩
        v_embed = Dense(self.hidden_dim, activation='relu')(vehicle_input)
        v_embed_expanded = RepeatVector(cfg.MAX_NUM_REQUEST)(v_embed)  # (B, R, H)
        
        # 요청 임베딩
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(request_input)  # (B, R, H)
        
        # 차량-요청 쌍 특징
        pair_features = Concatenate(axis=-1)([v_embed_expanded, r_embed, relation_input])  # (B, R, 2H+Drel)
        
        # 매칭 결정을 위한 처리
        match_logits = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(pair_features)
        match_logits = TimeDistributed(Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros'))(match_logits)
        match_logits = Lambda(lambda x: tf.squeeze(x, axis=-1))(match_logits)  # (B, R)
        
        # Reject 액션 처리
        r_summary = tf.reduce_mean(r_embed, axis=1)  # (B, H)
        reject_context = Concatenate(axis=-1)([v_embed, r_summary])  # (B, 2H)
        
        reject_logits = Dense(self.hidden_dim, activation='relu')(reject_context)
        reject_logits = Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')(reject_logits)  # (B, 1)
        
        # 모든 액션 로짓 결합
        action_logits = Concatenate(axis=-1)([match_logits, reject_logits])  # (B, R+1)
        
        return Model(inputs=[vehicle_input, request_input, relation_input], outputs=action_logits)
        
    def build_centralized_critic(self):
        """Centralized critic network - 전체 상태를 관찰"""
        # 전체 시스템 상태 입력
        vehicle_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM), name="all_vehicles_input")
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name="all_requests_input")
        relation_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM), name="all_relations_input")
        
        # 차량 및 요청 임베딩
        v_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(vehicle_input)  # (B, V, H)
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(request_input)  # (B, R, H)
        
        # 전체 상태 요약
        global_v = tf.reduce_mean(v_embed, axis=1)  # (B, H)
        global_r = tf.reduce_mean(r_embed, axis=1)  # (B, H)
        
        # 관계 정보 요약
        relation_flat = tf.reshape(relation_input, (-1, cfg.MAX_NUM_VEHICLES * cfg.MAX_NUM_REQUEST * cfg.RELATION_INPUT_DIM))
        relation_summary = Dense(self.hidden_dim, activation='relu')(relation_flat)  # (B, H)
        
        # 전체 상태 표현
        global_state = Concatenate(axis=-1)([global_v, global_r, relation_summary])  # (B, 3H)
        
        # 가치 추정
        value = Dense(self.hidden_dim, activation='relu')(global_state)
        value = Dense(self.hidden_dim, activation='relu')(value)
        value = Dense(1, kernel_initializer='glorot_uniform', bias_initializer='zeros')(value)
        
        # 출력 값 클리핑
        value = tf.clip_by_value(value, -10.0, 10.0)
        
        return Model(inputs=[vehicle_input, request_input, relation_input], outputs=value)
        
    def get_agent_state(self, global_state, agent_id):
        """전체 상태에서 특정 에이전트의 로컬 상태 추출"""
        vehicle_state = global_state[0][0, agent_id, :]  # 해당 차량 상태
        request_state = global_state[1][0, :, :]  # 모든 요청 상태 (공유)
        relation_state = global_state[2][0, agent_id, :, :]  # 해당 차량과 요청들의 관계
        
        return [
            vehicle_state[np.newaxis, :],  # (1, Dv)
            request_state[np.newaxis, :, :],  # (1, R, Dr)
            relation_state[np.newaxis, :, :]  # (1, R, Drel)
        ]
        
    def get_action_probs(self, global_state, action_masks, agent_id):
        """특정 에이전트의 액션 확률 분포 계산"""
        # 에이전트의 로컬 상태 추출
        agent_state = self.get_agent_state(global_state, agent_id)
        
        # 해당 에이전트의 actor 사용
        logits = self.actors[agent_id](agent_state, training=False)  # (1, A)
        
        # NaN/무한대 처리
        logits = tf.where(tf.math.is_finite(logits), logits, tf.zeros_like(logits))
        logits = tf.clip_by_value(logits, -5.0, 5.0)
        
        # 액션 마스크 적용 (해당 차량의 마스크만)
        agent_mask = action_masks[agent_id:agent_id+1, :]  # (1, A)
        masked_logits = tf.where(agent_mask == 1, logits, tf.constant(-5.0, dtype=tf.float32))
        
        # 확률 변환
        clipped_logits = tf.clip_by_value(masked_logits, -5.0, 5.0)
        temperature = 2.0
        scaled_logits = clipped_logits / temperature
        action_probs = tf.nn.softmax(scaled_logits, axis=-1)
        
        # 확률 안정화
        action_probs = tf.maximum(action_probs, 1e-6)
        action_probs = action_probs / tf.reduce_sum(action_probs, axis=-1, keepdims=True)
        action_probs = tf.where(tf.math.is_finite(action_probs), action_probs, tf.constant(1e-6, dtype=tf.float32))
        
        return action_probs, logits
        
    def act(self, state, action_mask, env=None):
        """MAPPO 액션 선택 (개선된 경쟁 규칙 적용)"""
        # 모든 IDLE 차량 찾기
        idle_vehicles = []
        for v_idx in range(self.num_agents):
            if np.sum(action_mask[v_idx]) > 0:  # 이 차량이 IDLE (유효한 액션이 있음)
                idle_vehicles.append(v_idx)
                
        if not idle_vehicles:
            # IDLE 차량이 없음
            return [0, cfg.POSSIBLE_ACTION - 1, {'mode': 'no_idle', 'action_prob': 0.0, 'logits': None}]
        
        # 모든 IDLE 차량에 대해 동시에 액션 선택
        all_actions = []
        all_probs = []
        all_logits = []
        request_conflicts = {}  # 요청별 충돌 차량들
        
        # 1단계: 모든 차량의 액션 선택
        for vehicle_idx in idle_vehicles:
            # 해당 차량의 액션 확률 계산
            action_probs, logits = self.get_action_probs(state, action_mask, vehicle_idx)
            
            # 유효한 액션 찾기
            valid_actions = tf.where(action_mask[vehicle_idx] == 1)
            
            if tf.shape(valid_actions)[0] == 0:
                # 유효한 액션이 없는 경우
                action_idx = cfg.POSSIBLE_ACTION - 1  # REJECT
                action_prob_value = 1.0
            else:
                # 마스킹된 확률에서 샘플링
                masked_probs = tf.where(
                    action_mask[vehicle_idx] == 1, 
                    action_probs[0], 
                    tf.constant(0.0, dtype=tf.float32)
                )
                
                # 확률이 0이 아닌 액션들만 고려
                non_zero_indices = tf.where(masked_probs > 0)[:, 0]
                
                if tf.shape(non_zero_indices)[0] == 0:
                    # 모든 확률이 0인 경우 - 균등 선택
                    valid_indices = tf.where(action_mask[vehicle_idx] == 1)[:, 0]
                    random_idx = tf.random.uniform([], 0, tf.shape(valid_indices)[0], dtype=tf.int32)
                    action_idx = int(valid_indices[random_idx])
                    action_prob_value = 1.0 / float(tf.shape(valid_indices)[0])
                else:
                    non_zero_probs = tf.gather(masked_probs, non_zero_indices)
                    non_zero_probs = non_zero_probs / tf.reduce_sum(non_zero_probs)
                    
                    # 샘플링
                    sampled_idx = tf.random.categorical(tf.math.log(non_zero_probs[tf.newaxis, :]), 1)[0, 0]
                    action_idx = int(non_zero_indices[sampled_idx])
                    action_prob_value = float(action_probs[0][action_idx].numpy())
            
            # 충돌 감지 (REJECT가 아닌 경우)
            if action_idx != cfg.POSSIBLE_ACTION - 1:
                if action_idx not in request_conflicts:
                    request_conflicts[action_idx] = []
                request_conflicts[action_idx].append({
                    'vehicle_idx': vehicle_idx,
                    'action_prob': action_prob_value,
                    'logits': logits.numpy()
                })
            
            all_actions.append([vehicle_idx, action_idx])
            all_probs.append(action_prob_value)
            all_logits.append(logits.numpy())
        
        # 2단계: 충돌 해결
        resolved_actions = []
        resolved_probs = []
        resolved_logits = []
        selected_requests = set()
        
        for i, (vehicle_idx, action_idx) in enumerate(all_actions):
            if action_idx == cfg.POSSIBLE_ACTION - 1:
                # REJECT 액션은 그대로 유지
                resolved_actions.append([vehicle_idx, action_idx])
                resolved_probs.append(all_probs[i])
                resolved_logits.append(all_logits[i])
            elif action_idx in request_conflicts:
                # 충돌이 있는 요청
                conflicting_vehicles = request_conflicts[action_idx]
                if len(conflicting_vehicles) == 1:
                    # 충돌 없음 - 승리
                    resolved_actions.append([vehicle_idx, action_idx])
                    resolved_probs.append(all_probs[i])
                    resolved_logits.append(all_logits[i])
                    selected_requests.add(action_idx)
                else:
                    # 충돌 있음 - 경쟁 규칙 적용
                    winner_idx = self.resolve_conflict(conflicting_vehicles, action_idx, state, env)
                    if winner_idx == vehicle_idx:
                        # 승리
                        resolved_actions.append([vehicle_idx, action_idx])
                        resolved_probs.append(all_probs[i])
                        resolved_logits.append(all_logits[i])
                        selected_requests.add(action_idx)
                    else:
                        # 패배 - REJECT로 변경
                        resolved_actions.append([vehicle_idx, cfg.POSSIBLE_ACTION - 1])
                        resolved_probs.append(1.0)
                        resolved_logits.append(all_logits[i])
            else:
                # 이미 처리된 요청
                resolved_actions.append([vehicle_idx, action_idx])
                resolved_probs.append(all_probs[i])
                resolved_logits.append(all_logits[i])
        
        # 모든 액션을 반환 (진정한 멀티 에이전트)
        info = {
            'mode': 'multi_agent_simultaneous',
            'all_actions': resolved_actions,  # 모든 에이전트의 액션 정보
            'all_probs': resolved_probs,
            'all_logits': resolved_logits,
            'idle_vehicles': idle_vehicles,
            'selected_requests': list(selected_requests),  # 선택된 요청들
            'num_agents': len(idle_vehicles),  # 동시 행동한 에이전트 수
            'conflicts_resolved': len(request_conflicts)  # 해결된 충돌 수
        }
        
        # 첫 번째 차량의 액션을 반환 (기존 인터페이스 호환성)
        vehicle_idx, action_idx = resolved_actions[0]
        action_prob_value = resolved_probs[0]
        logits = resolved_logits[0]
        
        info.update({
            'action_prob': action_prob_value,
            'logits': logits,
            'agent_id': vehicle_idx
        })
        
        return [vehicle_idx, action_idx, info]
        
    def resolve_conflict(self, conflicting_vehicles, request_id, state, env=None):
        """충돌 해결 규칙 - 여러 옵션 중 선택 가능"""
        # 옵션 1: 거리 기반 경쟁 (Distance-based Competition)
        return self.resolve_by_distance(conflicting_vehicles, request_id, state, env)
        
        # 옵션 2: 확률 기반 경쟁 (Probability-based Competition)
        # return self.resolve_by_probability(conflicting_vehicles)
        
        # 옵션 3: 혼합 경쟁 (Hybrid Competition)
        # return self.resolve_by_hybrid(conflicting_vehicles, request_id, state, env)
        
    def resolve_by_distance(self, conflicting_vehicles, request_id, state, env=None):
        """거리 기반 충돌 해결 - 가장 가까운 차량이 승리"""
        if env is None:
            # 환경 정보가 없으면 확률 기반으로 대체
            return self.resolve_by_probability(conflicting_vehicles)
            
        # 요청 정보 가져오기
        request = env.active_request_list[request_id]
        request_location = request.from_node_id
        
        # 각 차량의 거리 계산
        distances = []
        for vehicle_info in conflicting_vehicles:
            vehicle_idx = vehicle_info['vehicle_idx']
            vehicle = env.vehicle_list[vehicle_idx]
            distance = env.network.get_duration(vehicle.curr_node, request_location)
            distances.append((vehicle_idx, distance, vehicle_info))
        
        # 거리가 가장 가까운 차량 선택
        winner = min(distances, key=lambda x: x[1])
        return winner[0]
        
    def resolve_by_probability(self, conflicting_vehicles):
        """확률 기반 충돌 해결 - 액션 확률이 높은 차량이 승리"""
        best_vehicle = max(conflicting_vehicles, key=lambda x: x['action_prob'])
        return best_vehicle['vehicle_idx']
        
    def resolve_by_hybrid(self, conflicting_vehicles, request_id, state, env=None):
        """혼합 충돌 해결 - 거리와 확률을 결합한 점수로 경쟁"""
        if env is None:
            # 환경 정보가 없으면 확률 기반으로 대체
            return self.resolve_by_probability(conflicting_vehicles)
            
        # 요청 정보 가져오기
        request = env.active_request_list[request_id]
        request_location = request.from_node_id
        
        # 각 차량의 점수 계산
        scores = []
        for vehicle_info in conflicting_vehicles:
            vehicle_idx = vehicle_info['vehicle_idx']
            vehicle = env.vehicle_list[vehicle_idx]
            
            # 거리 점수 (가까울수록 높음)
            distance = env.network.get_duration(vehicle.curr_node, request_location)
            distance_score = 1.0 / (1.0 + distance / env.network.max_duration)
            
            # 확률 점수
            prob_score = vehicle_info['action_prob']
            
            # 혼합 점수 (가중치 조정 가능)
            total_score = 0.6 * distance_score + 0.4 * prob_score
            scores.append((vehicle_idx, total_score, vehicle_info))
        
        # 점수가 가장 높은 차량 선택
        winner = max(scores, key=lambda x: x[1])
        return winner[0]
        
    def get_value(self, state):
        """전체 상태의 가치 추정 (centralized)"""
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
            self.trajectory_buffer.extend(self.episode_buffer)
            self.episode_buffer = []
            
    def should_train(self):
        """학습을 수행할 조건 확인"""
        return len(self.trajectory_buffer) >= self.update_frequency
        
    def update_learning_rate(self, episode):
        """Learning rate 스케줄링"""
        if episode > 0:
            self.current_learning_rate = max(
                self.initial_learning_rate * (self.lr_decay_rate ** episode),
                self.min_lr
            )
            
            # Optimizer learning rate 업데이트
            for optimizer in self.actor_optimizers:
                optimizer.learning_rate = self.current_learning_rate
            self.critic_optimizer.learning_rate = self.current_learning_rate
            
    def normalize_rewards(self, rewards):
        """Reward 정규화"""
        if len(rewards) == 0:
            return rewards
            
        # 온라인 정규화 업데이트
        for reward in rewards:
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = reward - self.reward_mean
            self.reward_std += (delta * delta2 - self.reward_std) / self.reward_count
            
        # 안전한 표준편차
        safe_std = max(self.reward_std, 1e-8)
        
        # 정규화된 보상 반환
        normalized_rewards = (rewards - self.reward_mean) / safe_std
        return normalized_rewards
        
    def compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation 계산"""
        # 입력 값 NaN 체크 및 대체
        if np.any(np.isnan(values)):
            values = np.nan_to_num(values, nan=0.0)
        
        if np.any(np.isnan(rewards)):
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
            
            if np.isnan(gae):
                gae = 0.0
                
            advantages.insert(0, gae)
            
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values
        
        # NaN 최종 체크
        if np.any(np.isnan(advantages)):
            advantages = np.nan_to_num(advantages, nan=0.0)
            
        if np.any(np.isnan(returns)):
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
        """MAPPO 학습"""
        if len(self.trajectory_buffer) < 5:
            return None
            
        batch = self.trajectory_buffer[:]
        
        # 상태, 액션, 보상 등 추출
        states = [
            np.array([b[0][0][0] for b in batch]),  # vehicle states
            np.array([b[0][1][0] for b in batch]),  # request states  
            np.array([b[0][2][0] for b in batch])   # relation states
        ]
        
        actions = np.array([[b[1][0], b[1][1]] for b in batch])  # (B, 2) - [vehicle_idx, action_idx]
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = [
            np.array([b[3][0][0] for b in batch]),
            np.array([b[3][1][0] for b in batch]),
            np.array([b[3][2][0] for b in batch])
        ]
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        action_masks = np.array([b[5]['m'] for b in batch])
        old_action_probs = np.array([b[1][2]['action_prob'] for b in batch], dtype=np.float32)
        agent_ids = np.array([b[1][2].get('agent_id', b[1][0]) for b in batch])  # 에이전트 ID
        
        # Centralized 가치 추정
        values = tf.reshape(self.get_value(states), [-1]).numpy()
        next_values = tf.reshape(self.get_value(next_states), [-1]).numpy()
        
        # Reward 정규화
        normalized_rewards = self.normalize_rewards(rewards)
        
        # GAE 계산
        advantages, returns = self.compute_gae(normalized_rewards, values, dones)
        
        # MAPPO 업데이트
        total_actor_losses = [0.0] * self.num_agents
        total_critic_loss = 0.0
        
        for epoch in range(4):
            # 각 에이전트의 Actor 업데이트
            for agent_id in range(self.num_agents):
                # 해당 에이전트가 행동한 경험만 필터링
                agent_mask = (agent_ids == agent_id)
                if np.sum(agent_mask) == 0:
                    continue  # 이 에이전트는 이번 배치에서 행동하지 않음
                    
                agent_indices = np.where(agent_mask)[0]
                
                with tf.GradientTape() as tape:
                    # 해당 에이전트의 상태만 추출
                    agent_states = [
                        states[0][agent_indices],
                        states[1][agent_indices],
                        states[2][agent_indices]
                    ]
                    agent_action_masks = action_masks[agent_indices]
                    agent_actions = actions[agent_indices]
                    agent_advantages = advantages[agent_indices]
                    agent_old_probs = old_action_probs[agent_indices]
                    
                    # 선택된 액션의 확률 추출
                    action_indices = agent_actions[:, 1]  # action_idx만 사용
                    
                    # 에이전트별 액션 확률 계산 (개별 처리)
                    new_action_probs = []
                    for i, idx in enumerate(agent_indices):
                        # 각 에이전트의 상태를 개별적으로 처리
                        state_i = [
                            agent_states[0][i:i+1],  # (1, vehicle_dim)
                            agent_states[1][i:i+1],  # (1, request_dim, request_features)
                            agent_states[2][i:i+1]   # (1, request_dim, relation_features)
                        ]
                        probs, _ = self.get_action_probs(state_i, agent_action_masks[i:i+1], agent_id)
                        
                        # 선택된 액션의 확률 추출
                        action_idx = agent_actions[i, 1]  # action_idx
                        
                        # probs는 (1, 2, 9) 형태이므로 [0, 0, action_idx]로 접근
                        probs_shape = tf.shape(probs)
                        if tf.size(probs) > 0 and probs_shape[0] > 0 and action_idx < 9:
                            prob_value = probs[0, 0, action_idx]
                        else:
                            prob_value = tf.constant(0.0, dtype=tf.float32)
                            
                        new_action_probs.append(prob_value)
                    
                    new_action_probs = tf.stack(new_action_probs)
                    
                    # 에이전트별 전체 액션 확률 계산 (entropy 계산용)
                    agent_action_probs = []
                    for i, idx in enumerate(agent_indices):
                        state_i = [
                            agent_states[0][i:i+1],
                            agent_states[1][i:i+1],
                            agent_states[2][i:i+1]
                        ]
                        probs, _ = self.get_action_probs(state_i, agent_action_masks[i:i+1], agent_id)
                        probs_shape = tf.shape(probs)
                        if tf.size(probs) > 0 and probs_shape[0] > 0:
                            agent_action_probs.append(probs[0, 0, :])  # (9,)
                        else:
                            # 빈 텐서인 경우 0으로 채운 텐서 생성
                            agent_action_probs.append(tf.zeros(9, dtype=tf.float32))
                    
                    agent_action_probs = tf.stack(agent_action_probs)  # (batch_size, 9)
                    
                    # Ratio 계산 - 안전한 방식
                    ratio = new_action_probs / tf.maximum(agent_old_probs, 1e-8)
                    ratio = tf.clip_by_value(ratio, 0.1, 10.0)  # 극값 클리핑
                    
                    # Clipped surrogate objective
                    surr1 = ratio * agent_advantages
                    surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * agent_advantages
                    
                    # Entropy bonus
                    entropy = -tf.reduce_sum(agent_action_probs * tf.math.log(agent_action_probs + 1e-8), axis=-1)
                    entropy = tf.reduce_mean(entropy)
                    
                    # Actor loss (KL Divergence 제거)
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - self.entropy_coef * entropy
                    
                    # NaN/Inf 체크 및 대체
                    actor_loss = tf.where(tf.math.is_nan(actor_loss), tf.constant(0.0), actor_loss)
                    actor_loss = tf.where(tf.math.is_inf(actor_loss), tf.constant(0.0), actor_loss)
                        
                # Actor gradients
                actor_grads = tape.gradient(actor_loss, self.actors[agent_id].trainable_variables)
                if actor_grads is not None:
                    actor_grads = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in actor_grads if grad is not None]
                    self.actor_optimizers[agent_id].apply_gradients(
                        zip(actor_grads, self.actors[agent_id].trainable_variables)
                    )
                    
                actor_loss_val = actor_loss.numpy()
                if not np.isnan(actor_loss_val):
                    total_actor_losses[agent_id] += actor_loss_val
                    
            # Centralized Critic 업데이트
            with tf.GradientTape() as tape:
                current_values = tf.reshape(self.get_value(states), [-1])
                
                # Huber Loss 적용 (Loss 안정성 향상)
                huber_delta = 1.0
                diff = returns - current_values
                huber_loss = tf.where(
                    tf.abs(diff) < huber_delta,
                    0.5 * tf.square(diff),
                    huber_delta * (tf.abs(diff) - 0.5 * huber_delta)
                )
                critic_loss = tf.reduce_mean(huber_loss)
                
                # NaN/Inf 체크 및 대체
                critic_loss = tf.where(tf.math.is_nan(critic_loss), tf.constant(0.0), critic_loss)
                critic_loss = tf.where(tf.math.is_inf(critic_loss), tf.constant(0.0), critic_loss)
                    
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            if critic_grads is not None:
                critic_grads = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in critic_grads if grad is not None]
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                
            critic_loss_val = critic_loss.numpy()
            if not np.isnan(critic_loss_val):
                total_critic_loss += critic_loss_val
                
        # 버퍼 정리
        self.trajectory_buffer = []
        
        # 평균 actor loss 계산
        avg_actor_loss = np.mean([loss / 4 for loss in total_actor_losses if loss > 0])
        
        return (avg_actor_loss, total_critic_loss / 4)
