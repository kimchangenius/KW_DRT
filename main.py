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

    episodes = 1
    update_freq = 10
    final_train_steps = 5
    for ep in range(episodes):
        total_loss = 0.0
        total_reward = 0.0
        state = env.reset()

        delayed_reward_confirm = 0
        while True:
            print('\n============ Time : {} ============'.format(env.curr_time))
            env.print_vehicles()
            env.print_active_requests()

            while env.has_idle_vehicle():
                print('\n------------ Step : {} (Time : {}) ------------'.format(env.curr_step, env.curr_time))
                action_mask = env.get_action_mask()
                # print(action_mask)
                action = agent.act(state, action_mask)
                env.enrich_action(action)
                next_state, reward, info = env.step(action)

                print('Curr Reward: {}'.format(reward))
                env.print_vehicles()
                env.print_active_requests()

                transition = [state, action, reward, next_state, False]
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
                    agent.train()

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
                    agent.train()

                assert len(agent.pending_buffer) == 0, "Pending buffer is not empty"

                env.print_statistics()
                print('Total Reward: {}'.format(total_reward))
                print('Num Transitions: {}'.format(len(agent.replay_buffer)))
                print('Num Delayed Reward: {}'.format(delayed_reward_confirm))
                # env.print_vehicles()
                break

            env.sync_state()
            state = env.state

            # while any(v.status == 'idle' for v in env.vehicle_list):
            #     batch_states = np.tile(st, (NUM_VEHICLES, 1))
            #     veh_i, act = agent.act(batch_states, env.vehicle_list)
            #
            #     pprint(act)
            #     reward, next_state = env.step(veh_i, act)
            #     total_reward += reward
            #
            #     pprint(next_state)
            #     s = np.concatenate([batch_states[veh_i]])
            #     s_next = env.flatten_state(next_state)
            #     agent.remember(s.reshape(1, -1), [act], reward, s_next.reshape(1, -1), done)
            #     loss = agent.replay()
            #     if loss is None:
            #         loss = 0.0
            #     total_loss += loss
            #
            #     st = s_next
            #
            # env.curr_time += 1
            #
            # for v in env.vehicle_list:
            #     env.status_map.get(v.status, 0)
            #
            # done = (env.curr_time >= 60)
            # if done:
            #     break


if __name__ == "__main__":
    main()
