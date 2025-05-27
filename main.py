import os
import numpy as np
import pandas as pd

from pprint import pprint
from app.network import SiouxFallsNetwork
from app.env import RideSharingEnvironment
from app.agent import DQNAgent


def main():
    # data load & save
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    result_dir = os.path.join(current_dir, 'result')
    episode_dir = os.path.join(result_dir, 'episodes')
    passenger_data = os.path.join(data_dir, 'passengers.csv')
    vehicle_positions = os.path.join(data_dir, 'vehicle_positions.csv')
    net_data = os.path.join(data_dir, 'SiouxFalls_net.tntp')
    flow_data = os.path.join(data_dir, 'SiouxFalls_flow.tntp')
    node_coord_data = os.path.join(data_dir, 'SiouxFalls_node.tntp')
    node_xy_data = os.path.join(data_dir, 'SiouxFalls_node_xy.tntp')

    network = SiouxFallsNetwork(net_data, flow_data, node_coord_data, node_xy_data) # create network

    travel_time_output = os.path.join(data_dir, 'travel_time.csv')
    network.save_travel_time(travel_time_output)

    od_matrix_output = os.path.join(data_dir, 'od_matrix.csv')
    network.generate_od_matrix(od_matrix_output)

    print("Simulation setup complete")

    vehicle_positions = pd.read_csv(vehicle_positions)['initial_position'].tolist()
    passenger_data = pd.read_csv(passenger_data)

    state_size = len(vehicle_positions) * 5 + 20 * 3


    episode_logs = []
    episodes = 1

    agent = DQNAgent(state_size)
    weight_path = os.path.join(result_dir, "dqn_model_weights_final.weight.h5")
    agent.load_model(weight_path)

    for ep in range(episodes):
        step_logs = []
        env = RideSharingEnvironment(
            network=network,
            capacity=5,
            passenger_data=passenger_data,
            vehicle_positions=vehicle_positions
        )
        n_v = len(env.vehicles)
        total_reward = 0.0
        total_loss = 0.0
        prev_dropped = 0
        done = False

        while not done:
            st = env.get_state()
            pprint(st)
            st = env.flatten_state(st)
            env.update_current_requirement()

            # print(env.time)
            while any(v.status == 'idle' for v in env.vehicles):
                batch_states = np.tile(st, (n_v, 1))
                veh_i, act = agent.act(batch_states, env.vehicles)

                pprint(act)
                reward, next_state = env.step(veh_i, act)
                total_reward += reward

                pprint(next_state)
                s = np.concatenate([batch_states[veh_i]])
                s_next = env.flatten_state(next_state)
                agent.remember(s.reshape(1, -1), [act], reward, s_next.reshape(1, -1), done)
                loss = agent.replay()
                if loss is None:
                    loss = 0.0
                total_loss += loss

                st = s_next

            env.time += 1

            for v in env.vehicles:
                env.status_map.get(v.status, 0)

            done = (env.time >= 60)
            if done:
                break

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

        waiting_end = sum(1 for p in env.passengers if p.pickup_time is None)
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
        step_df.to_csv(os.path.join(episode_dir, f"{ep + 1} episode result.csv"), index=False)
        if ep == episodes - 1:
            final_model_path = os.path.join(result_dir, f"dqn_model_weights_final.weight.h5")
            agent.save_model(final_model_path)

    df = pd.DataFrame(episode_logs)
    df.to_csv(os.path.join(result_dir, "results.csv"), index=False)
    print("DQN training completed.")


if __name__ == "__main__":
    main()
