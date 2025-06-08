import os
import app.config as cfg
from pprint import pprint
from app.env_builder import EnvBuilder
from app.agent import DQNAgent

CURR_PATH = os.getcwd()
DATA_PATH = os.path.join(CURR_PATH, 'data')
RESULT_PATH = os.path.join(CURR_PATH, 'result')

episodes = 500
update_freq = 10
final_train_steps = 5

def train_ddqn(env_builder, config):
    env = env_builder.build()
    hidden_dim = config["hidden_dim"]

    agent = DQNAgent(hidden_dim=hidden_dim)

    total_rewards = []
    total_losses = []
    transition_id = 0
    for ep in range(episodes):
        # print('\n============ Ep : {} ============'.format(ep))
        total_loss = 0.0
        total_reward = 0.0
        state = env.reset()

        delayed_reward_confirm = 0
        while True:
            # print('\n============ Time : {} ============'.format(env.curr_time))
            # env.print_vehicles()
            # env.print_active_requests()

            while env.has_idle_vehicle():
                # print('\n------------ Step : {} (Time : {}) ------------'.format(env.curr_step, env.curr_time))

                action_mask = env.get_action_mask()
                action = agent.act(state, action_mask)
                env.enrich_action(action)

                next_state, reward, info = env.step(action)
                next_action_mask = env.get_action_mask()

                # print('Curr Reward: {}'.format(reward))
                # env.print_vehicles()
                # env.print_active_requests()

                t_info = {
                    'id': transition_id,
                    'm': action_mask,
                    'nm': next_action_mask,
                }

                transition = [state, action, reward, next_state, False, t_info]
                transition_id += 1
                if info['is_pending'] is True:
                    agent.pending(transition)
                else:
                    agent.remember(transition)

                if info['has_delayed_reward'] is True:
                    for action_id in info['action_id_list']:
                        d_reward = info['reward']
                        agent.confirm_and_remember(action_id, d_reward)
                        delayed_reward_confirm += 1

                if env.curr_step % update_freq == 0:
                    curr_loss = agent.train()
                    if curr_loss is not None:
                        total_loss += curr_loss

                total_reward += reward
                state = next_state

            env.curr_time += 1
            d_reward_list = env.handle_time_update()

            for pair in d_reward_list:
                agent.confirm_and_remember(pair[0], pair[1])
                delayed_reward_confirm += 1

            if env.is_done():
                # 마지막 transition의 done을 True로 변경
                last_transition = agent.replay_buffer.get_last()
                if last_transition is not None:
                    last_transition[4] = True

                for _ in range(final_train_steps):
                    curr_loss = agent.train()
                    if curr_loss is not None:
                        total_loss += curr_loss

                if len(agent.pending_buffer) != 0:
                    for k, v in agent.pending_buffer.pending.items():
                        print(k)
                        pprint(v)
                assert len(agent.pending_buffer) == 0, "Pending buffer is not empty"

                num_accepted = 0
                num_served = 0
                for v in env.vehicle_list:
                    num_accepted += v.num_accept
                    num_served += v.num_serve
                    v.on_service_driving_time = env.curr_time - v.idle_time
                print('====== Ep: {} / Reward: {} / Loss: {} / eps: {} ======'.format(ep, total_reward, total_loss, agent.epsilon))
                print(num_accepted, num_served)
                for v in env.vehicle_list:
                    print(v.idle_time, v.on_service_driving_time, env.curr_time)
                for r in env.done_request_list:
                    print(r.waiting_time, r.in_vehicle_time)
                total_rewards.append(total_reward)
                total_losses.append(total_loss)
                # print('Num Transitions: {}'.format(len(agent.replay_buffer)))
                # print('Num Delayed Reward: {}'.format(delayed_reward_confirm))
                # env.print_vehicles()
                break

            env.sync_state()
            state = env.state

        agent.decay_epsilon()

    # TODO: Write CSV
    print(total_rewards)
    print(total_losses)
    # agent.model.save_weights("dqn_weights.h5")


def main():
    env_builder = EnvBuilder(data_dir=DATA_PATH, result_dir=RESULT_PATH)

    for params in cfg.config_list:
        train_ddqn(env_builder, params)

if __name__ == "__main__":
    main()
