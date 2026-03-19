import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, TimeDistributed, Lambda,
    Concatenate, RepeatVector, Flatten
)


class PPOAgent:
    def __init__(self, hidden_dim, pi_lr, vf_lr, mini_batch_size=64):
        self.hidden_dim = hidden_dim
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.mini_batch_size = mini_batch_size

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.ppo_epochs = 4
        self.max_grad_norm = 0.5

        self.trajectory = []
        self.pending_actions = {}

        self.train_step = 0
        self._flat_action_size = cfg.MAX_NUM_VEHICLES * cfg.POSSIBLE_ACTION

    def save_model(self, file_path):
        actor_path = file_path + "_actor.h5"
        critic_path = file_path + "_critic.h5"
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Model saved: {file_path}")

    def load_model(self, file_path):
        actor_path = file_path + "_actor.h5"
        critic_path = file_path + "_critic.h5"
        if os.path.exists(actor_path):
            self.actor.load_weights(actor_path)
            print(f"Actor loaded: {actor_path}")
        else:
            print(f"No actor weights found: {actor_path}")
        if os.path.exists(critic_path):
            self.critic.load_weights(critic_path)
            print(f"Critic loaded: {critic_path}")
        else:
            print(f"No critic weights found: {critic_path}")

    def _build_actor(self):
        vehicle_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM), name="vehicle_input")
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name="request_input")
        relation_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM), name="relation_input")

        v_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(vehicle_input)
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(request_input)

        v_expand = tf.expand_dims(v_embed, axis=2)
        r_expand = tf.expand_dims(r_embed, axis=1)

        v_tiled = tf.tile(v_expand, [1, 1, cfg.MAX_NUM_REQUEST, 1])
        r_tiled = tf.tile(r_expand, [1, cfg.MAX_NUM_VEHICLES, 1, 1])

        pair_embed = Concatenate(axis=-1)([v_tiled, r_tiled, relation_input])

        logits_match = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation='relu')))(pair_embed)
        logits_match = TimeDistributed(TimeDistributed(Dense(1)))(logits_match)
        logits_match = Lambda(lambda x: tf.squeeze(x, axis=-1))(logits_match)

        r_summary = tf.reduce_mean(r_embed, axis=1)
        r_summary = RepeatVector(cfg.MAX_NUM_VEHICLES)(r_summary)
        reject_context = Concatenate(axis=-1)([v_embed, r_summary])

        logits_reject = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(reject_context)
        logits_reject = TimeDistributed(Dense(1))(logits_reject)

        policy_logits = Concatenate(axis=-1)([logits_match, logits_reject])  # (B, V, R+1)

        return Model(inputs=[vehicle_input, request_input, relation_input], outputs=policy_logits)

    def _build_critic(self):
        vehicle_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM), name="vehicle_input")
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name="request_input")
        relation_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM), name="relation_input")

        v_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(vehicle_input)
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(request_input)

        v_flat = Flatten()(v_embed)
        r_flat = Flatten()(r_embed)
        rel_flat = Flatten()(relation_input)

        combined = Concatenate()([v_flat, r_flat, rel_flat])
        hidden = Dense(self.hidden_dim, activation='relu')(combined)
        hidden = Dense(self.hidden_dim, activation='relu')(hidden)
        value = Dense(1)(hidden)

        return Model(inputs=[vehicle_input, request_input, relation_input], outputs=value)

    def get_value(self, state):
        value = self.critic.predict(state, verbose=0)
        return float(value[0, 0])

    def act(self, state, action_mask):
        info = {'mode': 'policy'}

        logits = self.actor.predict(state, verbose=0)  # (1, V, A)
        masked_logits = tf.where(
            action_mask == 1, logits, tf.constant(-1e9, dtype=tf.float32)
        )
        flat_logits = tf.reshape(masked_logits, (1, -1))  # (1, V*A)

        flat_action = tf.random.categorical(flat_logits, num_samples=1, dtype=tf.int32)
        flat_idx = int(flat_action[0, 0].numpy())

        vehicle_idx = flat_idx // cfg.POSSIBLE_ACTION
        action_idx = flat_idx % cfg.POSSIBLE_ACTION

        flat_log_probs = tf.nn.log_softmax(flat_logits, axis=-1)
        info['log_prob'] = float(flat_log_probs[0, flat_idx].numpy())

        return [vehicle_idx, action_idx, info]

    def act_greedy(self, state, action_mask):
        info = {'mode': 'greedy'}

        logits = self.actor.predict(state, verbose=0)
        masked_logits = tf.where(
            action_mask == 1, logits, tf.constant(-1e9, dtype=tf.float32)
        )
        flat_logits = tf.reshape(masked_logits, (-1,))
        flat_idx = int(tf.argmax(flat_logits).numpy())

        vehicle_idx = flat_idx // cfg.POSSIBLE_ACTION
        action_idx = flat_idx % cfg.POSSIBLE_ACTION

        flat_log_probs = tf.nn.log_softmax(tf.reshape(masked_logits, (1, -1)), axis=-1)
        info['log_prob'] = float(flat_log_probs[0, flat_idx].numpy())

        return [vehicle_idx, action_idx, info]

    # ── Trajectory Management ──

    def store_transition(self, state, action, reward, action_mask, log_prob, value, done=False):
        self.trajectory.append({
            'state': state,
            'action': [action[0], action[1]],
            'reward': reward,
            'action_mask': action_mask,
            'log_prob': log_prob,
            'value': value,
            'done': done,
        })

    def add_pending(self, action_id, traj_idx):
        self.pending_actions[action_id] = traj_idx

    def confirm_pending(self, action_id, reward):
        traj_idx = self.pending_actions.pop(action_id, None)
        if traj_idx is not None and traj_idx < len(self.trajectory):
            self.trajectory[traj_idx]['reward'] += reward
            return True
        return False

    def clear_trajectory(self):
        self.trajectory = []
        self.pending_actions.clear()

    # ── GAE ──

    def _compute_gae(self, last_value):
        T = len(self.trajectory)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            reward = self.trajectory[t]['reward']
            value = self.trajectory[t]['value']
            done = 1.0 if self.trajectory[t]['done'] else 0.0

            delta = reward + self.gamma * next_value * (1.0 - done) - value
            gae = delta + self.gamma * self.lam * (1.0 - done) * gae

            advantages[t] = gae
            returns[t] = gae + value
            next_value = value

        return advantages, returns

    # ── PPO Training ──

    def train(self, last_value=0.0):
        T = len(self.trajectory)
        if T == 0:
            return None

        advantages, returns = self._compute_gae(last_value)

        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        states_v = np.array([t['state'][0][0] for t in self.trajectory])
        states_r = np.array([t['state'][1][0] for t in self.trajectory])
        states_rel = np.array([t['state'][2][0] for t in self.trajectory])
        old_log_probs = np.array([t['log_prob'] for t in self.trajectory], dtype=np.float32)
        actions = np.array([t['action'] for t in self.trajectory], dtype=np.int32)
        action_masks = np.array([t['action_mask'] for t in self.trajectory], dtype=np.float32)

        total_loss = 0.0
        num_updates = 0

        # Drop Last Batch: TRL/CleanRL 방식 - 마지막 불완전 배치 스킵으로 고정 shape 보장 (retracing 방지)
        # 참고: Hugging Face TRL drop_last=True, CleanRL은 고정 batch_size로 항상 균등 분할
        num_full_batches = T // self.mini_batch_size
        if num_full_batches == 0:
            return None  # trajectory가 mini_batch_size 미만이면 학습 스킵

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(T)

            for batch_i in range(num_full_batches):
                start = batch_i * self.mini_batch_size
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = [states_v[mb_idx], states_r[mb_idx], states_rel[mb_idx]]
                mb_actions = tf.constant(actions[mb_idx], dtype=tf.int32)
                mb_old_lp = tf.constant(old_log_probs[mb_idx], dtype=tf.float32)
                mb_adv = tf.constant(advantages[mb_idx], dtype=tf.float32)
                mb_ret = tf.constant(returns[mb_idx], dtype=tf.float32)
                mb_masks = tf.constant(action_masks[mb_idx], dtype=tf.float32)

                # ── Actor ──
                with tf.GradientTape() as actor_tape:
                    logits = self.actor(mb_states, training=True)  # (mb, V, A)
                    masked_logits = tf.where(
                        mb_masks == 1, logits,
                        tf.constant(-1e9, dtype=tf.float32)
                    )
                    flat_logits = tf.reshape(masked_logits, [-1, self._flat_action_size])

                    flat_log_probs = tf.nn.log_softmax(flat_logits, axis=-1)
                    flat_probs = tf.nn.softmax(flat_logits, axis=-1)

                    flat_act_idx = mb_actions[:, 0] * cfg.POSSIBLE_ACTION + mb_actions[:, 1]
                    batch_idx = tf.range(self.mini_batch_size, dtype=tf.int32)
                    gather_idx = tf.stack([batch_idx, flat_act_idx], axis=1)
                    new_log_probs = tf.gather_nd(flat_log_probs, gather_idx)

                    ratio = tf.exp(new_log_probs - mb_old_lp)
                    clipped = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    policy_loss_per_sample = -tf.minimum(ratio * mb_adv, clipped * mb_adv)
                    policy_loss = tf.reduce_mean(policy_loss_per_sample)

                    safe_log = tf.where(flat_probs > 0, flat_log_probs, tf.zeros_like(flat_log_probs))
                    entropy_per_sample = -tf.reduce_sum(flat_probs * safe_log, axis=-1)
                    entropy_loss = -self.entropy_coef * tf.reduce_mean(entropy_per_sample)

                    actor_loss = policy_loss + entropy_loss

                actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                actor_grads = [
                    g if g is not None else tf.zeros_like(v)
                    for g, v in zip(actor_grads, self.actor.trainable_variables)
                ]
                actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                # ── Critic ──
                with tf.GradientTape() as critic_tape:
                    values = self.critic(mb_states, training=True)  # (mb, 1)
                    values = tf.squeeze(values, axis=-1)
                    critic_loss_per_sample = tf.square(mb_ret - values)
                    critic_loss = tf.reduce_mean(critic_loss_per_sample)

                critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
                critic_grads = [
                    g if g is not None else tf.zeros_like(v)
                    for g, v in zip(critic_grads, self.critic.trainable_variables)
                ]
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                total_loss += (actor_loss.numpy() + critic_loss.numpy())
                num_updates += 1

        self.train_step += 1
        return total_loss / max(num_updates, 1)
