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
    agent.load_model(DQN_WEIGHT_PATH)

    episode_logs = []
    episodes = 1

    env = RideSharingEnvironment(
        network=network,
        original_request_list=request_list,
        vehicle_init_pos=vehicle_positions
    )

    for ep in range(episodes):
        # step_logs = []
        # total_reward = 0.0
        # total_loss = 0.0
        # prev_dropped = 0

        done = False
        time_updated = False

        state = env.reset()
        # print(state[0].shape)
        # print(state[1].shape)
        # print(state[2].shape)

        while not done:
            print('\n============ Time : {} ============'.format(env.curr_time))
            env.print_vehicles()
            env.print_requests()

            while env.has_idle_vehicle():
                print('\n------------ Step : {} (Time : {}) ------------'.format(env.curr_step, env.curr_time))
                action_mask = env.get_action_mask()
                print(action_mask)
                action = agent.act(state, action_mask)
                next_state, reward, done = env.step(action)

                # print(next_state.shape)
                print(reward)
                env.print_vehicles()
                env.print_requests()

                state = next_state

            env.curr_time += 1

            # idle vehicle이 없는 상황
            env.handle_time_update()
            env.sync_state()
            state = env.state

            if env.curr_time == 70:
                env.print_logs()
                return

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

        agent.decay_epsilon(ep, episodes)

            # dropped_this_step = env.dropped_passengers - prev_dropped
            # prev_dropped = env.dropped_passengers
            #
            # waiting = sum(1 for p in env.passengers if p.pickup_time is None)
            #
            # matched = len(env.matched_ids)
            # canceled = len(env.canceled_ids)
            # total_accounted = matched + canceled + waiting
            #
            # step_logs.append({
            #     "Step": t + 1,
            #     "Total Reward": total_reward,
            #     "Loss": loss,
            #     "Matched Passengers": matched,
            #     "Canceled": canceled,
            #     "Dropped Passengers": dropped_this_step,
            #     "Waiting Passengers": waiting,
            #     "Total Accounted Passengers": total_accounted,
            #     "Rebalancing Count": env.rebalancing_count
            # })


        # if ep < 100:
        #     agent.epsilon = 1.0
        # else:
        #     agent.epsilon = max(agent.epsilon_min,
        #                         1.0 - (ep - 100) * (1.0 - agent.epsilon_min) / (episodes - 100))

        waiting_end = sum(1 for p in env.passengers if p.pickup_at is None)
        episode_logs.append({
            "Episode": ep + 1,
            "Total Reward": total_reward,
            "Loss": total_loss,
            "Matched Passengers": len(env.matched_ids),
            "Canceled": len(env.canceled_ids),
            "Dropped Passengers": env.dropped_passengers,
            "Waiting Passengers": waiting_end,
            "Total Accounted Passengers": len(env.matched_ids) + len(env.canceled_ids) + waiting_end,
            "Rebalancing Count": env.rebalancing_count
        })

        print(f"Episode: {ep + 1}  Total Reward: {total_reward:.2f}  Total Loss: {total_loss:.4f}")
        step_df = pd.DataFrame(step_logs)
        step_df.to_csv(os.path.join(EPISODES_PATH, f"{ep + 1} episode result.csv"), index=False)
        if ep == episodes - 1:
            final_model_path = os.path.join(RESULT_PATH, f"dqn_model_weights_final.weight.h5")
            agent.save_model(final_model_path)

    df = pd.DataFrame(episode_logs)
    df.to_csv(os.path.join(RESULT_PATH, "results.csv"), index=False)
    print("DQN training completed.")


if __name__ == "__main__":
    main()
