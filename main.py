import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import app.config as cfg

from pprint import pprint
from app.env import RideSharingEnvironment
from app.agent import DQNAgent
from app.request import Request
from app.network import DRTNetwork

CURR_PATH = os.getcwd()
DATA_PATH = os.path.join(CURR_PATH, 'data')
RESULT_PATH = os.path.join(CURR_PATH, 'result')
EPISODES_PATH = os.path.join(RESULT_PATH, 'episodes')
PASSENGER_PATH = os.path.join(DATA_PATH, 'passengers.csv')
VEH_POS_PATH = os.path.join(DATA_PATH, 'vehicle_positions.csv')
OD_MATRIX_PATH = os.path.join(DATA_PATH, 'od_matrix.csv')
DQN_WEIGHT_PATH = os.path.join(RESULT_PATH, "dqn_model_weights_final.weight.h5")


def load_requests(path, network):
    requests = []
    with open(path, newline='', encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            req = Request(
                request_id=int(row["User_ID"]),
                from_node_id=int(row["Start_node"]),
                to_node_id=int(row["End_node"]),
                request_time=int(row["Request_time"]),
                network=network
            )
            requests.append(req)
    return requests


def main():
    network = DRTNetwork()
    network.set_od_matrix(OD_MATRIX_PATH)

    request_list = load_requests(PASSENGER_PATH, network)
    request_list.sort(key=lambda r: r.request_time)
    for r in request_list:
        tt = network.get_duration(r.from_node_id, r.to_node_id)
        r.set_travel_time(tt)

    vehicle_positions = pd.read_csv(VEH_POS_PATH)['initial_position'].tolist()

    print("Data Load & Network Setup Complete")

    agent = DQNAgent(hidden_dim=128)
    # agent.load_model(DQN_WEIGHT_PATH)

    env = RideSharingEnvironment(
        network=network,
        original_request_list=request_list,
        vehicle_init_pos=vehicle_positions
    )

    total_rewards = []
    total_losses = []

    episodes = 500
    update_freq = 10
    final_train_steps = 5
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

                assert len(agent.pending_buffer) == 0, "Pending buffer is not empty"

                # env.print_statistics()
                # print('Total Reward: {}'.format(total_reward))
                # print('Total Loss: {}'.format(total_loss))
                print('====== Ep: {} / Reward: {} / Loss: {} / eps: {} ======'.format(ep, total_reward, total_loss, agent.epsilon))
                total_rewards.append(total_reward)
                total_losses.append(total_loss)
                # print('Num Transitions: {}'.format(len(agent.replay_buffer)))
                # print('Num Delayed Reward: {}'.format(delayed_reward_confirm))
                # env.print_vehicles()
                break

            env.sync_state()
            state = env.state

        agent.decay_epsilon()

    print(total_rewards)
    print(total_losses)


if __name__ == "__main__":
    main()
