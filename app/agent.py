import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from scipy.optimize import linear_sum_assignment

from app.pending_buffer import PendingBuffer
from app.replay_buffer import ReplayBuffer
from app.request_status import RequestStatus
from app.vehicle_status import VehicleStatus
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

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

    def _predict(self, state):
        return self.model(state, training=False).numpy()

    def act(self, state, action_mask):
        info = {
            'mode': None
        }
        if np.random.rand() < self.epsilon:
            info['mode'] = 'explore'
            valid_actions = tf.where(action_mask == 1)
            rand_idx = tf.random.uniform(shape=(), maxval=tf.shape(valid_actions)[0], dtype=tf.int32)
            rand_action = valid_actions[rand_idx]
            rand_action = rand_action.numpy()
            vehicle_idx = int(rand_action[0])
            action_idx = int(rand_action[1])
        else:
            info['mode'] = 'exploit'
            q_values = self._predict(state)
            masked_q = tf.where(action_mask == 1, q_values, tf.constant(-1e2, dtype=tf.float32))
            flat_idx = tf.argmax(tf.reshape(masked_q, (-1,))).numpy()
            vehicle_idx = int(flat_idx // cfg.POSSIBLE_ACTION)
            action_idx = int(flat_idx % cfg.POSSIBLE_ACTION)
        return [vehicle_idx, action_idx, info]

    def act_pickup_assignments(self, state, action_mask, env):
        """
        IDLE 차량들에 대해 PICKUP/DROPOFF/REJECT 액션을 Hungarian Method로 동시 결정.

        Cost matrix 컬럼 구성:
            - 0..MAX_NUM_REQUEST-1: 각 active request 슬롯
                * PENDING + capacity 충분 + waiting_time + dur < MAX_WAIT_TIME → PICKUP 후보
                * PICKEDUP & 차량과 매칭됨 → DROPOFF 후보
                * 그 외 → +INF (선택 불가)
            - MAX_NUM_REQUEST..MAX_NUM_REQUEST+n_v-1: 차량 i 전용 REJECT 컬럼
                * 다른 차량은 +INF, 자기 자신만 -Q[v, REJECT_slot]

        반환:
            List[[veh_idx(int), action_idx(int), info(dict)]]
            action_idx는 active_request 슬롯 또는 cfg.POSSIBLE_ACTION-1 (REJECT).
        """
        INF = 1e9
        REJECT_IDX = cfg.POSSIBLE_ACTION - 1  # action 공간에서의 REJECT 슬롯

        idle_v_indices = []
        for i, v in enumerate(env.vehicle_list):
            if v.status != VehicleStatus.IDLE:
                continue
            idle_v_indices.append(i)

        if len(idle_v_indices) == 0:
            return []

        is_explore = np.random.rand() < self.epsilon
        if is_explore:
            mode = 'explore'
            score = -np.random.rand(cfg.MAX_NUM_VEHICLES, cfg.POSSIBLE_ACTION).astype(np.float32)
        else:
            mode = 'exploit'
            q_values = self._predict(state)[0]
            score = q_values

        n_v = len(idle_v_indices)
        n_r = cfg.MAX_NUM_REQUEST
        n_cols = n_r + n_v  # request 슬롯 + 차량별 REJECT 슬롯

        cost = np.full((n_v, n_cols), INF, dtype=np.float32)

        # request 컬럼 (PICKUP 또는 DROPOFF)
        for ii, vi in enumerate(idle_v_indices):
            v = env.vehicle_list[vi]
            for rj in range(n_r):
                if rj >= len(env.active_request_list):
                    break
                if action_mask[vi, rj] != 1:
                    continue
                r = env.active_request_list[rj]
                if r.status == RequestStatus.PENDING:
                    # PICKUP 후보: 누적 대기 + 이동시간이 timeout 이내일 때만
                    dur = env.network.get_duration(v.curr_node, r.from_node_id)
                    if r.waiting_time + dur >= cfg.MAX_WAIT_TIME:
                        continue
                    cost[ii, rj] = -float(score[vi, rj])
                elif r.status == RequestStatus.PICKEDUP:
                    # DROPOFF 후보: action_mask에서 이미 r.assigned_v_id == v.id 검사 통과
                    cost[ii, rj] = -float(score[vi, rj])
                # 그 외 status는 +INF 유지

            # 차량 ii 전용 REJECT 컬럼
            if action_mask[vi, REJECT_IDX] == 1:
                cost[ii, n_r + ii] = -float(score[vi, REJECT_IDX])

        row_idx, col_idx = linear_sum_assignment(cost)

        actions = []
        for ii, jj in zip(row_idx, col_idx):
            if cost[ii, jj] >= INF:
                continue
            vi = idle_v_indices[ii]
            if jj < n_r:
                action_idx = int(jj)
                r_obj = env.active_request_list[jj]
            else:
                action_idx = REJECT_IDX
                r_obj = None
            actions.append([int(vi), action_idx, {'mode': mode}, r_obj])
        return actions

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

    @tf.function(reduce_retracing=True)
    def _train_step(self, vehicle_t, request_t, relation_t, action_mask, full_indices, targets):
        with tf.GradientTape() as tape:
            q_values = self.model([vehicle_t, request_t, relation_t], training=True)
            masked_q_values = tf.where(action_mask == 1, q_values, tf.constant(-1e1, dtype=tf.float32))
            q_sa = tf.gather_nd(masked_q_values, full_indices)
            # loss = tf.reduce_mean(tf.keras.losses.MSE(targets, q_sa))
            loss = tf.reduce_mean(tf.keras.losses.huber(targets, q_sa, delta=1.0))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # print("\n\n================= Train : {} =================".format(self.train_step))
        batch = self.replay_buffer.sample(self.batch_size)

        # === 입력 텐서 일괄 구성 (numpy → tf.constant, dtype/shape 고정으로 retracing 최소화) ===
        next_vehicle_t = tf.constant(np.array([b[3][0][0] for b in batch], dtype=np.float32))
        next_request_t = tf.constant(np.array([b[3][1][0] for b in batch], dtype=np.float32))
        next_relation_t = tf.constant(np.array([b[3][2][0] for b in batch], dtype=np.float32))
        next_action_mask = tf.constant(np.array([b[5]['nm'] for b in batch], dtype=np.float32))

        vehicle_t = tf.constant(np.array([b[0][0][0] for b in batch], dtype=np.float32))
        request_t = tf.constant(np.array([b[0][1][0] for b in batch], dtype=np.float32))
        relation_t = tf.constant(np.array([b[0][2][0] for b in batch], dtype=np.float32))
        action_mask = tf.constant(np.array([b[5]['m'] for b in batch], dtype=np.float32))

        rewards = tf.constant(np.array([b[2] for b in batch], dtype=np.float32))
        dones = tf.constant(np.array([b[4] for b in batch], dtype=np.float32))

        actions_arr = np.array([[b[1][0], b[1][1]] for b in batch], dtype=np.int32)
        indices = tf.constant(actions_arr, dtype=tf.int32)
        batch_indices = tf.range(tf.shape(indices)[0], dtype=tf.int32)[:, tf.newaxis]
        full_indices = tf.concat([batch_indices, indices], axis=1)

        # === Q(s', a') ===
        next_q_main = self.model([next_vehicle_t, next_request_t, next_relation_t], training=False)
        masked_main = tf.where(next_action_mask == 1, next_q_main, tf.constant(-1e9, dtype=tf.float32))
        B = tf.shape(masked_main)[0]
        flat_idx = tf.argmax(tf.reshape(masked_main, (B, -1)), axis=1, output_type=tf.int32)
        v_idx = flat_idx // cfg.POSSIBLE_ACTION
        a_idx = flat_idx % cfg.POSSIBLE_ACTION
        b_idx = tf.range(B, dtype=tf.int32)
        gather_idx = tf.stack([b_idx, v_idx, a_idx], axis=1)

        next_q_target = self.target_model([next_vehicle_t, next_request_t, next_relation_t], training=False)
        max_next_q = tf.gather_nd(next_q_target, gather_idx)

        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        # === Q(s, a) + 학습 ===
        loss = self._train_step(vehicle_t, request_t, relation_t, action_mask, full_indices, targets)

        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        return loss.numpy()
