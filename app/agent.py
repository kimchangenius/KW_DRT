"""
Pair-wise Q scorer (단순 MLP).

노드 임베딩 기반 차량/요청 투영 → 평균 풀 글로벌 + 스칼라 time_norm + global_stats →
페어별 MLP. 페어 입력에는 snapshot 의 pair_agg(Option A: 큐·시간·부하 집계 스칼라)를 concat.

공간 배치는 vehicle_nodes · request_nodes · rel_feat(및 네트워크 기반 지속시간)으로
여전히 학습 가능하다. 집계 스칼라는 “시스템이 얼마나 바쁜가”를 알려 줄 뿐
지리를 직접 인코딩하지 않는다.
"""
import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from scipy.optimize import linear_sum_assignment

from app.action_type import ActionType
from app.pending_buffer import PendingBuffer
from app.replay_buffer import ReplayBuffer
from app.vehicle_status import VehicleStatus
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras import mixed_precision


# ===========================================================================
class MLPPairScorer(tf.keras.Model):
    def __init__(self, hidden_dim, edge_weight_np=None, **kwargs):
        super().__init__(**kwargs)
        _ = edge_weight_np
        self.hidden_dim = hidden_dim

        self.node_emb = Embedding(
            cfg.NUM_NODES + 1, cfg.NODE_EMB_DIM, name='node_embedding'
        )
        self.v_proj = Dense(hidden_dim, activation='relu', name='v_proj')
        self.r_proj = Dense(hidden_dim, activation='relu', name='r_proj')

        self.global_proj = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu'),
        ], name='global_proj')

        self.r_null = self.add_weight(
            name='r_null_token',
            shape=(hidden_dim,),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.pair_head = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu'),
            Dense(1, dtype='float32', name='q_out'),
        ], name='pair_head')

    def call(self, inputs, training=False):
        (
            vehicle_static, vehicle_nodes,
            request_static, request_nodes, request_mask,
            time_norm, global_stats,
            pair_batch_idx, pair_v_idx, pair_r_idx, pair_is_reject, pair_rel,
            pair_agg_batch,
        ) = inputs
        v_ctx, r_ctx, g_ctx = self.encode_context(
            vehicle_static, vehicle_nodes,
            request_static, request_nodes, request_mask,
            time_norm, global_stats,
            training=training,
        )
        return self.score_pairs(
            v_ctx, r_ctx, g_ctx,
            pair_batch_idx, pair_v_idx, pair_r_idx, pair_is_reject, pair_rel,
            pair_agg_batch,
        )

    def encode_context(
        self,
        vehicle_static, vehicle_nodes,
        request_static, request_nodes, request_mask,
        time_norm, global_stats,
        training=False,
    ):
        _ = training
        all_nodes = tf.range(cfg.NUM_NODES + 1)
        node_tbl = self.node_emb(all_nodes)
        fd = node_tbl.dtype

        v_node_emb = tf.gather(node_tbl, vehicle_nodes)
        v_shape = tf.shape(vehicle_static)
        v_node_emb = tf.reshape(
            v_node_emb, (v_shape[0], v_shape[1], 2 * cfg.NODE_EMB_DIM)
        )
        v_input = tf.concat([tf.cast(vehicle_static, fd), v_node_emb], axis=-1)
        v_ctx = self.v_proj(v_input)

        r_node_emb = tf.gather(node_tbl, request_nodes)
        r_shape = tf.shape(request_static)
        r_node_emb = tf.reshape(
            r_node_emb, (r_shape[0], r_shape[1], 2 * cfg.NODE_EMB_DIM)
        )
        r_input = tf.concat([tf.cast(request_static, fd), r_node_emb], axis=-1)
        r_ctx = self.r_proj(r_input)

        v_pool = tf.reduce_mean(v_ctx, axis=1)
        rmask_f = tf.cast(request_mask, r_ctx.dtype)
        r_sum = tf.reduce_sum(r_ctx * tf.expand_dims(rmask_f, -1), axis=1)
        r_cnt = tf.reduce_sum(rmask_f, axis=1, keepdims=True)
        r_pool = r_sum / tf.maximum(r_cnt, 1.0)

        tn = tf.cast(tf.expand_dims(time_norm, -1), fd)
        gs = tf.cast(global_stats, fd)
        global_in = tf.concat([v_pool, r_pool, tn, gs], axis=-1)
        global_ctx = self.global_proj(global_in)
        return v_ctx, r_ctx, global_ctx

    def score_pairs(
        self,
        v_ctx, r_ctx, global_ctx,
        pair_batch_idx, pair_v_idx, pair_r_idx, pair_is_reject, pair_rel,
        pair_agg_batch,
    ):
        """pair_agg_batch: (B, PAIR_AGG_DIM) — 배치 b의 페어는 pair_agg_batch[b]."""
        v_gather = tf.stack([pair_batch_idx, pair_v_idx], axis=1)
        v_emb = tf.gather_nd(v_ctx, v_gather)

        r_len = tf.shape(r_ctx)[1]
        Bsz = tf.shape(r_ctx)[0]
        Hsz = tf.shape(r_ctx)[2]
        r_ctx_safe = tf.cond(
            tf.greater(r_len, 0),
            lambda: r_ctx,
            lambda: tf.zeros((Bsz, 1, Hsz), dtype=r_ctx.dtype),
        )

        r_gather = tf.stack([pair_batch_idx, pair_r_idx], axis=1)
        r_emb_real = tf.gather_nd(r_ctx_safe, r_gather)
        is_rej_b = tf.cast(pair_is_reject, tf.bool)[:, tf.newaxis]
        r_null_hw = tf.cast(self.r_null, r_emb_real.dtype)
        r_null_b = tf.broadcast_to(r_null_hw[tf.newaxis, :], tf.shape(r_emb_real))
        r_emb = tf.where(is_rej_b, r_null_b, r_emb_real)

        g_emb = tf.gather(global_ctx, pair_batch_idx)
        pair_rel_c = tf.cast(pair_rel, v_emb.dtype)
        agg = tf.gather(pair_agg_batch, pair_batch_idx)
        agg = tf.cast(agg, v_emb.dtype)
        pair_in = tf.concat([v_emb, r_emb, pair_rel_c, g_emb, agg], axis=-1)
        q = self.pair_head(pair_in)
        return tf.squeeze(q, axis=-1)


# ===========================================================================
# DQNAgent (Double DQN with MLP pair scorer)
class DQNAgent:
    def __init__(self, hidden_dim, batch_size, learning_rate, edge_weight_np):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_micro_batch_size = (
            min(getattr(cfg, 'TRAIN_MICRO_BATCH_SIZE', 8), batch_size)
            if batch_size > 0 else 1
        )

        self.model = MLPPairScorer(hidden_dim, edge_weight_np)
        self.target_model = MLPPairScorer(hidden_dim, edge_weight_np)

        # 빌드: 더미 forward로 weight 초기화 (양쪽 동일 가중치)
        self._dry_forward(self.model)
        self._dry_forward(self.target_model)
        self.target_model.set_weights(self.model.get_weights())

        self.train_step = 0
        self.update_target_freq = 500
        base_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer = mixed_precision.LossScaleOptimizer(base_opt)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.replay_buffer = ReplayBuffer()
        self.pending_buffer = PendingBuffer()

    @staticmethod
    def _dense_grad_if_indexed_slices(grad, variable):
        """Embedding 등에서 IndexedSlices 로 오는 미분은 tensor_scatter_nd_add 로 dense 로 합산."""
        if grad is None:
            return None
        if isinstance(grad, tf.IndexedSlices):
            base = tf.zeros(variable.shape, dtype=grad.values.dtype)
            idx = tf.expand_dims(grad.indices, 1)
            return tf.tensor_scatter_nd_add(base, idx, grad.values)
        return grad

    @staticmethod
    def _dry_forward(model):
        """모델 weight 초기화를 위해 더미 입력 forward."""
        V = cfg.MAX_NUM_VEHICLES
        R = 1
        v_static = tf.zeros((1, V, cfg.VEHICLE_RAW_DIM), dtype=tf.float32)
        v_nodes = tf.zeros((1, V, 2), dtype=tf.int32)
        r_static = tf.zeros((1, R, cfg.REQUEST_RAW_DIM), dtype=tf.float32)
        r_nodes = tf.zeros((1, R, 2), dtype=tf.int32)
        r_mask = tf.ones((1, R), dtype=tf.bool)
        time_n = tf.zeros((1,), dtype=tf.float32)
        gstats = tf.zeros((1, cfg.GLOBAL_STATS_DIM), dtype=tf.float32)
        pair_agg_bf = tf.zeros((1, cfg.PAIR_AGG_DIM), dtype=tf.float32)
        inputs = (
            v_static, v_nodes, r_static, r_nodes, r_mask, time_n, gstats,
            tf.zeros((1,), dtype=tf.int32),
            tf.zeros((1,), dtype=tf.int32),
            tf.zeros((1,), dtype=tf.int32),
            tf.ones((1,), dtype=tf.float32),
            tf.zeros((1, cfg.RELATION_INPUT_DIM), dtype=tf.float32),
            pair_agg_bf,
        )
        model(inputs, training=False)

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

    # -----------------------------------------------------------------------
    # Snapshot → tensor 변환 헬퍼 (단일 batch)
    # -----------------------------------------------------------------------
    @staticmethod
    def _pair_agg_numpy(snapshot_or_none):
        if snapshot_or_none is None:
            return np.zeros(cfg.PAIR_AGG_DIM, dtype=np.float32)
        pa = snapshot_or_none.get('pair_agg')
        if pa is None:
            return np.zeros(cfg.PAIR_AGG_DIM, dtype=np.float32)
        pa = np.asarray(pa, dtype=np.float32).reshape(-1)
        if pa.shape[0] != cfg.PAIR_AGG_DIM:
            z = np.zeros(cfg.PAIR_AGG_DIM, dtype=np.float32)
            n = min(pa.shape[0], cfg.PAIR_AGG_DIM)
            z[:n] = pa[:n]
            pa = z
        return pa

    @staticmethod
    def _snapshot_to_batched_tensors(snapshot):
        """단일 snapshot dict → batch 차원 추가된 tf 텐서들."""
        v_static = tf.constant(snapshot['vehicle_static'][None, ...], dtype=tf.float32)
        v_nodes = tf.constant(snapshot['vehicle_nodes'][None, ...], dtype=tf.int32)
        R = snapshot['request_static'].shape[0]
        if R > 0:
            r_static = tf.constant(snapshot['request_static'][None, ...], dtype=tf.float32)
            r_nodes = tf.constant(snapshot['request_nodes'][None, ...], dtype=tf.int32)
            r_mask = tf.ones((1, R), dtype=tf.bool)
        else:
            r_static = tf.zeros((1, 0, cfg.REQUEST_RAW_DIM), dtype=tf.float32)
            r_nodes = tf.zeros((1, 0, 2), dtype=tf.int32)
            r_mask = tf.zeros((1, 0), dtype=tf.bool)
        time_n = tf.constant([snapshot['time_norm']], dtype=tf.float32)
        gstats = tf.constant(snapshot['global_stats'][None, ...], dtype=tf.float32)
        pair_agg = tf.constant(
            DQNAgent._pair_agg_numpy(snapshot)[None, ...], dtype=tf.float32
        )
        return v_static, v_nodes, r_static, r_nodes, r_mask, time_n, gstats, pair_agg

    # -----------------------------------------------------------------------
    # 의사결정 (Hungarian + per-pair Q)
    # -----------------------------------------------------------------------
    def act_pickup_assignments(self, env, snapshot=None):
        """
        IDLE 차량들에 대해 페어 후보를 enumerate해 (full-context) Q로 점수화한 뒤,
        Hungarian으로 PICKUP/DROPOFF/REJECT를 동시 결정.

        반환: List[dict]  — env.step에 그대로 넣을 수 있는 액션 + transition 메타
            'vehicle_idx', 'action_type', 'request', 'pair_info', 'action_id', 'mode'
        """
        INF = 1e9

        idle_v = [v for v in env.vehicle_list if v.status == VehicleStatus.IDLE]
        if not idle_v:
            return []
        if not env.has_dispatch_candidate():
            return []

        # 1) snapshot이 미리 안 들어왔으면 그 자리에서 캡처
        if snapshot is None:
            snapshot = env.get_snapshot()

        # 2) 페어 후보 enumerate
        candidates_by_v = env.enumerate_pair_candidates(idle_v, include_wait=True)

        flat_pairs = []      # (v_idx, r_slot_idx, is_reject, rel_feat)
        flat_meta = []       # (idle_row_idx, candidate dict)
        for ii, v in enumerate(idle_v):
            for c in candidates_by_v[v.id]:
                flat_pairs.append((c['v_idx'], c['r_slot_idx'], c['is_reject'], c['rel_feat']))
                flat_meta.append((ii, c))

        if not flat_pairs:
            return []

        # 3) Q 점수 (혹은 explore 시 랜덤)
        is_explore = np.random.rand() < self.epsilon
        if is_explore:
            mode = 'explore'
            scores = np.random.rand(len(flat_pairs)).astype(np.float32)
        else:
            mode = 'exploit'
            v_static, v_nodes, r_static, r_nodes, r_mask, time_n, gstats, pair_agg_bf = \
                self._snapshot_to_batched_tensors(snapshot)
            v_ctx, r_ctx, g_ctx = self.model.encode_context(
                v_static, v_nodes, r_static, r_nodes, r_mask, time_n, gstats, training=False,
            )
            P = len(flat_pairs)
            pair_batch_idx = tf.zeros((P,), dtype=tf.int32)
            pair_v_idx = tf.constant([p[0] for p in flat_pairs], dtype=tf.int32)
            pair_r_idx = tf.constant([p[1] for p in flat_pairs], dtype=tf.int32)
            pair_is_rej = tf.constant([float(p[2]) for p in flat_pairs], dtype=tf.float32)
            pair_rel = tf.constant(np.stack([p[3] for p in flat_pairs]).astype(np.float32))
            scores = self.model.score_pairs(
                v_ctx, r_ctx, g_ctx,
                pair_batch_idx, pair_v_idx, pair_r_idx, pair_is_rej, pair_rel,
                pair_agg_batch=pair_agg_bf,
            ).numpy()

        # 4) Cost matrix 구성 — (n_idle_v, n_active_r + n_idle_v)
        n_v = len(idle_v)
        n_r = len(env.active_request_list)
        n_cols = n_r + n_v

        cost = np.full((n_v, n_cols), INF, dtype=np.float32)
        for (ii, c), score in zip(flat_meta, scores):
            if c['is_reject']:
                col = n_r + ii
            else:
                col = c['r_slot_idx']
            cost[ii, col] = -float(score)

        row_idx, col_idx = linear_sum_assignment(cost)

        actions = []
        for ii, jj in zip(row_idx, col_idx):
            if cost[ii, jj] >= INF:
                continue
            v = idle_v[ii]
            if jj < n_r:
                cand = self._find_candidate(candidates_by_v[v.id], jj)
                if cand is None:
                    continue
                r_obj = cand['r']
                atype = cand['action_type']
                action_id = self._make_action_id(r_obj, atype)
            else:
                cand = next((c for c in candidates_by_v[v.id] if c['is_reject']), None)
                if cand is None:
                    continue
                r_obj = None
                atype = ActionType.REJECT
                action_id = None

            actions.append({
                'vehicle_idx': v.id,
                'action_type': atype,
                'request': r_obj,
                'pair_info': {
                    'v_idx': cand['v_idx'],
                    'r_slot_idx': cand['r_slot_idx'],
                    'is_reject': cand['is_reject'],
                    'rel_feat': cand['rel_feat'],
                },
                'action_id': action_id,
                'mode': mode,
                'score': -float(cost[ii, jj]),
            })
        return actions

    @staticmethod
    def _find_candidate(cand_list, r_slot_idx):
        for c in cand_list:
            if not c['is_reject'] and c['r_slot_idx'] == r_slot_idx:
                return c
        return None

    @staticmethod
    def _make_action_id(request, action_type):
        if request is None:
            return None
        return f"{request.id}_{action_type.value}"

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    # -----------------------------------------------------------------------
    # Replay / Pending 인터페이스
    # -----------------------------------------------------------------------
    def remember(self, transition):
        self.replay_buffer.append(transition)

    def pending(self, transition):
        action_id = transition['meta']['action_id']
        if action_id is None:
            self.replay_buffer.append(transition)
            return
        self.pending_buffer.add(action_id, transition)

    def confirm_and_remember(self, action_id, reward):
        transition = self.pending_buffer.confirm(action_id, reward)
        if transition is not None:
            self.remember(transition)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    @staticmethod
    def _pad_snapshots(snapshots):
        """B개 snapshot을 batch 차원으로 묶고 R 축 패딩 + mask 생성.

        Returns dict of tensors ready for model.encode_context().
        """
        B = len(snapshots)
        V = snapshots[0]['vehicle_static'].shape[0]  # 모두 동일 (=MAX_NUM_VEHICLES)
        R_max = max(s['request_static'].shape[0] for s in snapshots)

        v_static = np.zeros((B, V, cfg.VEHICLE_RAW_DIM), dtype=np.float32)
        v_nodes = np.zeros((B, V, 2), dtype=np.int32)
        r_static = np.zeros((B, max(R_max, 1), cfg.REQUEST_RAW_DIM), dtype=np.float32)
        r_nodes = np.zeros((B, max(R_max, 1), 2), dtype=np.int32)
        r_mask = np.zeros((B, max(R_max, 1)), dtype=bool)
        time_n = np.zeros((B,), dtype=np.float32)
        gstats = np.zeros((B, cfg.GLOBAL_STATS_DIM), dtype=np.float32)
        pair_agg = np.zeros((B, cfg.PAIR_AGG_DIM), dtype=np.float32)

        for i, s in enumerate(snapshots):
            v_static[i] = s['vehicle_static']
            v_nodes[i] = s['vehicle_nodes']
            R = s['request_static'].shape[0]
            if R > 0:
                r_static[i, :R] = s['request_static']
                r_nodes[i, :R] = s['request_nodes']
                r_mask[i, :R] = True
            time_n[i] = s['time_norm']
            gstats[i] = s['global_stats']
            pair_agg[i] = DQNAgent._pair_agg_numpy(s)

        return {
            'v_static': tf.constant(v_static),
            'v_nodes': tf.constant(v_nodes),
            'r_static': tf.constant(r_static),
            'r_nodes': tf.constant(r_nodes),
            'r_mask': tf.constant(r_mask),
            'time_n': tf.constant(time_n),
            'gstats': tf.constant(gstats),
            'pair_agg': tf.constant(pair_agg),
        }

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None

        B = len(batch)

        # === Snapshots: current + next ===
        cur_snap = [b['snapshot'] for b in batch]
        next_snap = [b['next_snapshot'] for b in batch]

        nxt_t = self._pad_snapshots(next_snap)

        # === Current action pair info (B,) ===
        cur_pair_v = np.array([b['action_pair']['v_idx'] for b in batch], dtype=np.int32)
        cur_pair_r = np.array([b['action_pair']['r_slot_idx'] for b in batch], dtype=np.int32)
        cur_pair_isrej = np.array([float(b['action_pair']['is_reject']) for b in batch], dtype=np.float32)
        cur_pair_rel = np.stack([b['action_pair']['rel_feat'] for b in batch]).astype(np.float32)

        # === Next pairs flat concat (sum_N pairs) + segment_ids ===
        n_v_list, n_r_list, n_isrej_list, n_rel_list, n_seg = [], [], [], [], []
        for i, bitem in enumerate(batch):
            vehicle_idx = bitem.get('vehicle_idx')
            for p in bitem['next_pair_indices']:
                if vehicle_idx is not None and p['v_idx'] != vehicle_idx:
                    continue
                n_v_list.append(p['v_idx'])
                n_r_list.append(p['r_slot_idx'])
                n_isrej_list.append(float(p['is_reject']))
                n_rel_list.append(p['rel_feat'])
                n_seg.append(i)

        if n_v_list:
            n_v_arr = np.array(n_v_list, dtype=np.int32)
            n_r_arr = np.array(n_r_list, dtype=np.int32)
            n_isrej_arr = np.array(n_isrej_list, dtype=np.float32)
            n_rel_arr = np.stack(n_rel_list).astype(np.float32)
            n_seg_arr = np.array(n_seg, dtype=np.int32)
        else:
            n_v_arr = np.zeros((0,), dtype=np.int32)
            n_r_arr = np.zeros((0,), dtype=np.int32)
            n_isrej_arr = np.zeros((0,), dtype=np.float32)
            n_rel_arr = np.zeros((0, cfg.RELATION_INPUT_DIM), dtype=np.float32)
            n_seg_arr = np.zeros((0,), dtype=np.int32)

        rewards = np.array([b['reward'] for b in batch], dtype=np.float32)
        dones = np.array([float(b['done']) for b in batch], dtype=np.float32)

        # 전체 배치 타깃 (역전파 없음; full batch next 상태 한 번 평가)
        targets = self._compute_td_targets(
            nxt_t,
            tf.constant(n_v_arr), tf.constant(n_r_arr), tf.constant(n_isrej_arr),
            tf.constant(n_rel_arr), tf.constant(n_seg_arr),
            tf.constant(B, dtype=tf.int32),
            tf.constant(rewards, dtype=tf.float32),
            tf.constant(dones, dtype=tf.float32),
        )

        # 마이크로 배치별 그래디언트 누적 — VRAM 피크는 micro 크기만 사용
        # loss_k_mean = 마이크로 내 평균 Huber 일 때 Σ_k loss_k_mean * (m_k/B) = 전체 배치 평균 Huber 의 그래디언트
        micro = self.train_micro_batch_size
        huber_none = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)

        grads_acc = None
        reporting_loss_accum = 0.0

        bf = tf.cast(B, tf.float32)
        vars_ = self.model.trainable_variables

        for s in range(0, B, micro):
            e = min(s + micro, B)
            msize = e - s
            cur_t_m = self._pad_snapshots(cur_snap[s:e])

            cp_batch_m = tf.range(msize, dtype=tf.int32)
            cp_v_m = tf.constant(cur_pair_v[s:e], dtype=tf.int32)
            cp_r_m = tf.constant(cur_pair_r[s:e], dtype=tf.int32)
            cp_isrej_m = tf.constant(cur_pair_isrej[s:e], dtype=tf.float32)
            cp_rel_m = tf.constant(cur_pair_rel[s:e], dtype=tf.float32)
            targets_m = tf.gather(targets, tf.range(s, e))

            mf = tf.cast(msize, tf.float32)

            with tf.GradientTape() as tape:
                v_ctx, r_ctx, g_ctx = self.model.encode_context(
                    cur_t_m['v_static'], cur_t_m['v_nodes'],
                    cur_t_m['r_static'], cur_t_m['r_nodes'], cur_t_m['r_mask'],
                    cur_t_m['time_n'], cur_t_m['gstats'], training=True,
                )
                q_sa = self.model.score_pairs(
                    v_ctx, r_ctx, g_ctx,
                    cp_batch_m, cp_v_m, cp_r_m, cp_isrej_m, cp_rel_m,
                    pair_agg_batch=cur_t_m['pair_agg'],
                )
                per_h = huber_none(targets_m, q_sa)  # (msize,)
                loss_weighted = tf.reduce_mean(per_h) * (mf / bf)
                # Keras LossScaleOptimizer 내부 플래그: get_scaled_loss 가 그래프를 끊어
                # tape.gradient 가 None 이 되는 버전 이슈가 있어, 역전파에는 loss*scale 을 쓰되
                # 동적 스케일 상태를 위해 no-grad 경로로 한 번 호출한다.
                self.optimizer.get_scaled_loss(tf.stop_gradient(loss_weighted))
                ls = tf.cast(self.optimizer.loss_scale, tf.float32)
                scaled = loss_weighted * ls

            grads_k = tape.gradient(scaled, vars_)
            grads_k = self.optimizer.get_unscaled_gradients(grads_k)

            if grads_acc is None:
                grads_acc = [
                    self._dense_grad_if_indexed_slices(gk, vk)
                    for gk, vk in zip(grads_k, vars_)
                ]
            else:
                next_acc = []
                for ga, gk, vk in zip(grads_acc, grads_k, vars_):
                    dk = self._dense_grad_if_indexed_slices(gk, vk)
                    if ga is None and dk is None:
                        next_acc.append(None)
                    elif ga is None:
                        next_acc.append(dk)
                    elif dk is None:
                        next_acc.append(ga)
                    else:
                        next_acc.append(ga + dk)
                grads_acc = next_acc

            reporting_loss_accum += float(tf.reduce_mean(per_h).numpy()) * msize

        grads_clip = []
        vars_clip = []
        for ga, v in zip(grads_acc, vars_):
            if ga is not None:
                grads_clip.append(ga)
                vars_clip.append(v)
        grads_clip, _ = tf.clip_by_global_norm(grads_clip, 5.0)
        self.optimizer.apply_gradients(zip(grads_clip, vars_clip))

        mean_loss_report = reporting_loss_accum / float(B)

        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        return np.float32(mean_loss_report)

    def _compute_td_targets(
        self,
        nxt_t,
        np_v, np_r, np_isrej, np_rel, np_seg,
        B, rewards, dones,
    ):
        """Tape 밖에서 Double-DQN 타깃만 계산 (전체 next 배치 1회)."""
        n_total = tf.shape(np_v)[0]

        def _compute_target_max():
            v_ctx_m, r_ctx_m, g_ctx_m = self.model.encode_context(
                nxt_t['v_static'], nxt_t['v_nodes'],
                nxt_t['r_static'], nxt_t['r_nodes'], nxt_t['r_mask'],
                nxt_t['time_n'], nxt_t['gstats'], training=False,
            )
            next_q_main = self.model.score_pairs(
                v_ctx_m, r_ctx_m, g_ctx_m, np_seg, np_v, np_r, np_isrej, np_rel,
                pair_agg_batch=nxt_t['pair_agg'],
            )
            v_ctx_t, r_ctx_t, g_ctx_t = self.target_model.encode_context(
                nxt_t['v_static'], nxt_t['v_nodes'],
                nxt_t['r_static'], nxt_t['r_nodes'], nxt_t['r_mask'],
                nxt_t['time_n'], nxt_t['gstats'], training=False,
            )
            next_q_target = self.target_model.score_pairs(
                v_ctx_t, r_ctx_t, g_ctx_t, np_seg, np_v, np_r, np_isrej, np_rel,
                pair_agg_batch=nxt_t['pair_agg'],
            )

            seg_max_main = tf.math.unsorted_segment_max(next_q_main, np_seg, B)
            is_argmax = tf.equal(next_q_main, tf.gather(seg_max_main, np_seg))
            sentinel = tf.fill(tf.shape(next_q_target), tf.float32.min)
            masked_target = tf.where(is_argmax, next_q_target, sentinel)
            seg_max_target = tf.math.unsorted_segment_max(masked_target, np_seg, B)
            return tf.where(seg_max_target > -1e30, seg_max_target, tf.zeros_like(seg_max_target))

        target_max = tf.cond(
            n_total > 0,
            _compute_target_max,
            lambda: tf.zeros([B], dtype=tf.float32),
        )
        return rewards + self.gamma * target_max * (1.0 - dones)

    # _train_step_impl 제거됨 → train() + _compute_td_targets 로 분리
