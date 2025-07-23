import os
import numpy as np
import tensorflow as tf
import app.config as cfg

from app.replay_buffer import ReplayBuffer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, RepeatVector, Reshape

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self, scale=1.0):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x * scale  

    def reset(self):
        """Noise state를 초기값으로 리셋"""
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)
        
    def decay_std(self, decay_factor=0.995, min_std=0.01):
        """
        Noise 표준편차를 점진적으로 감소
        Args:
            decay_factor: 감소 비율 (0.995 = 0.5% 감소)
            min_std: 최소 표준편차
        """
        self.std_dev = np.maximum(self.std_dev * decay_factor, min_std)

class DDPGAgent:
    def __init__(self, hidden_dim, batch_size, learning_rate):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.replay_buffer = ReplayBuffer()
        # OU Noise 초기화 (더 적극적인 탐험을 위해 std_deviation 증가)
        self.noise = OUActionNoise(
            mean=np.zeros(cfg.MAX_NUM_VEHICLES), 
            std_deviation=0.3 * np.ones(cfg.MAX_NUM_VEHICLES),
            theta=0.15,
            dt=1e-2
        )

        self.tau = 0.005
        self.gamma = 0.99
        
    def reset_noise(self):
        """에피소드 시작 시 noise 상태 리셋"""
        self.noise.reset()
        
    def decay_noise(self):
        """에피소드마다 noise 강도 감소"""
        self.noise.decay_std(decay_factor=0.995, min_std=0.05)

    def build_actor(self):
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
        pair_embed = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation='relu')))(pair_embed)
        pair_embed = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation='relu')))(pair_embed)
        pair_flat = Reshape((cfg.MAX_NUM_VEHICLES, -1))(pair_embed)

        v_flat = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(v_embed)
        concat = Concatenate(axis=-1)([v_flat, pair_flat])
        
        # 더 부드러운 action 생성을 위한 추가 레이어
        out = TimeDistributed(Dense(self.hidden_dim // 2, activation='relu'))(concat)
        out = TimeDistributed(Dense(1, activation='tanh'))(out)
        
        # 추가: action의 범위를 조절하여 더 세밀한 제어 가능
        out = Lambda(lambda x: x * 2.0)(out)  # [-2, 2] 범위로 확장
        out = Reshape((cfg.MAX_NUM_VEHICLES,))(out)
        return Model([vehicle_input, request_input, relation_input], out)

    def build_critic(self):
        vehicle_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.VEHICLE_INPUT_DIM), name="vehicle_input")
        request_input = Input(shape=(cfg.MAX_NUM_REQUEST, cfg.REQUEST_INPUT_DIM), name="request_input")
        relation_input = Input(shape=(cfg.MAX_NUM_VEHICLES, cfg.MAX_NUM_REQUEST, cfg.RELATION_INPUT_DIM), name="relation_input")
        action_input = Input(shape=(cfg.MAX_NUM_VEHICLES,), name="action_input")

        v_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(vehicle_input)
        r_embed = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(request_input)

        v_expand = tf.expand_dims(v_embed, axis=2)
        r_expand = tf.expand_dims(r_embed, axis=1)
        v_tiled = tf.tile(v_expand, [1, 1, cfg.MAX_NUM_REQUEST, 1])
        r_tiled = tf.tile(r_expand, [1, cfg.MAX_NUM_VEHICLES, 1, 1])

        pair_embed = Concatenate(axis=-1)([v_tiled, r_tiled, relation_input])
        pair_embed = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation='relu')))(pair_embed)
        pair_embed = TimeDistributed(TimeDistributed(Dense(self.hidden_dim, activation='relu')))(pair_embed)
        pair_flat = Reshape((cfg.MAX_NUM_VEHICLES, -1))(pair_embed)

        v_flat = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(v_embed)
        concat = Concatenate(axis=-1)([v_flat, pair_flat, action_input[..., None]])
        
        # critic도 더 깊게 만들어 복잡한 state-action 관계 학습
        out = TimeDistributed(Dense(self.hidden_dim, activation='relu'))(concat)
        out = TimeDistributed(Dense(self.hidden_dim // 2, activation='relu'))(out)
        out = TimeDistributed(Dense(1))(out)
        out = Lambda(lambda x: tf.reduce_mean(x, axis=1))(out)  
        return Model([vehicle_input, request_input, relation_input, action_input], out)

    def act(self, state, add_noise=True, noise_scale=1.0):
        # state: [vehicle_state, request_state, relation_state]
        state = [s.astype(np.float32) for s in state]
        action = self.actor.predict(state, verbose=0)[0]
        if add_noise:
            # OUActionNoise의 scale 파라미터 활용
            noise = self.noise(scale=noise_scale).astype(np.float32)
            action += noise
        # 확장된 범위에 맞춰 clipping
        return np.clip(action, -2.0, 2.0)  # [-2, 2] 범위로 확장

    def preprocess_action(self, action):
        """
        연속 action을 환경에 더 적합하게 전처리
        Args:
            action: 원시 연속 action [-2, 2]
        Returns:
            processed_action: 환경에 최적화된 action
        """
        # action을 [0, 1] 범위로 정규화
        normalized = (action + 2.0) / 4.0  # [-2,2] -> [0,1]
        
        # 더 부드러운 action 분포를 위해 beta 분포 적용
        # 중간값(0.5) 근처에 더 많은 확률 질량 할당
        alpha, beta = 2.0, 2.0
        processed = np.random.beta(alpha + normalized * 2, beta + (1-normalized) * 2)
        
        # 다시 [-2, 2] 범위로 변환
        return (processed * 4.0) - 2.0

    def remember(self, transition):
        self.replay_buffer.append(transition)


    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = [np.array([b[0][i][0] for b in batch]).astype(np.float32) for i in range(3)]  # vehicle, request, relation
        actions = np.array([b[1] for b in batch]).astype(np.float32)  # 연속 action 그대로 사용
        rewards = np.array([b[2] for b in batch]).reshape(-1, 1).astype(np.float32)
        next_states = [np.array([b[3][i][0] for b in batch]).astype(np.float32) for i in range(3)]
        dones = np.array([b[4] for b in batch]).reshape(-1, 1).astype(np.float32)

        # Critic update
        next_actions = self.target_actor.predict(next_states, verbose=0)
        target_q = self.target_critic.predict([*next_states, next_actions], verbose=0)
        y = rewards + self.gamma * target_q * (1 - dones)
        with tf.GradientTape() as tape:
            q = self.critic([*states, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            actor_loss = -tf.reduce_mean(self.critic([*states, actions_pred]))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update
        self.update_target(self.target_actor, self.actor)
        self.update_target(self.target_critic, self.critic)
        return actor_loss.numpy(), critic_loss.numpy()

    def update_target(self, target, source):
        for t, s in zip(target.weights, source.weights):
            t.assign(self.tau * s + (1 - self.tau) * t)

    def save_model(self, file_path):
        self.actor.save_weights(file_path + '_actor.h5')
        self.critic.save_weights(file_path + '_critic.h5')
        print(f"DDPG model weights saved at {file_path}_actor.h5, {file_path}_critic.h5")

    def load_model(self, file_path):
        if os.path.exists(file_path + '_actor.h5') and os.path.exists(file_path + '_critic.h5'):
            self.actor.load_weights(file_path + '_actor.h5')
            self.critic.load_weights(file_path + '_critic.h5')
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
            print(f"DDPG model weights loaded from {file_path}_actor.h5, {file_path}_critic.h5")
        else:
            print(f"No DDPG model weights found at {file_path}_actor.h5, {file_path}_critic.h5") 