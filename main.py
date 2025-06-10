import os
import csv
import app.config as cfg
from pprint import pprint
from app.env_builder import EnvBuilder
from app.agent import DQNAgent
from app.request_status import RequestStatus

CURR_PATH = os.getcwd()
DATA_PATH = os.path.join(CURR_PATH, 'data')
RESULT_PATH = os.path.join(CURR_PATH, 'result')

episodes = 500
update_freq = 10
final_train_steps = 5


def log_episode(path, info):
    ep = info['episode']

    drt_info_list = info['drt_info']
    filename = f'episode_{ep:03}_vehicle.csv'
    filepath = os.path.join(path, filename)
    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Vehicle ID', 'Num. Accept', 'Num. Serve', 'On-Service Driving Time', 'Idle Time'])
        for v in drt_info_list:
            curr_row = [v['id'], v['num_accept'], v['num_serve'], v['on_service_driving_time'], v['idle_time']]
            writer.writerow(curr_row)

    req_info_list = info['request_info']
    filename = f'episode_{ep:03}_request.csv'
    filepath = os.path.join(path, filename)
    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Request ID', 'Status', 'Waiting Time', 'In-Vehicle Time', 'Detour Time'])
        for r in req_info_list:
            curr_row = [r['id'], r['status'], r['waiting_time'], r['in_vehicle_time'], r['detour_time']]
            writer.writerow(curr_row)


def log_all_episodes(path, info_list):
    filename = 'episodes.csv'
    filepath = os.path.join(path, filename)
    with open(filepath, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Total Reward', 'Total Loss', 'Total Num. Accept', 'Total Num. Serve',
                         'Mean Waiting Time', 'Mean In-Vehicle Time', 'Mean Detour Time'])
        for e in info_list:
            curr_row = [
                e['episode'],
                f"{e['total_reward']:.2f}",
                f"{e['total_loss']:.2f}",
                e['total_num_accept'],
                e['total_num_serve'],
                f"{e['mean_waiting_time']:.2f}",
                f"{e['mean_in_vehicle_time']:.2f}",
                f"{e['mean_detour_time']:.2f}"
            ]
            writer.writerow(curr_row)


def get_run_folder_name(config):
    hd = config.get("hidden_dim", "x")
    bs = config.get("batch_size", "x")
    lr = config.get("learning_rate", "x")
    return f"hd{hd}_bs{bs}_lr{lr}"


def train_ddqn(env_builder, config):
    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< Training Session: {config_str} >>>>")

    # Create Result Directory
    run_name = get_run_folder_name(config)
    run_path = os.path.join(RESULT_PATH, run_name)
    os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config["learning_rate"]

    env = env_builder.build()
    agent = DQNAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')

    for ep in range(1, episodes + 1):
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
                    print("[Warning] Pending Buffer is not empty!")
                    # for k, v in agent.pending_buffer.pending.items():
                    #     print(k)
                    #     pprint(v)
                    agent.pending_buffer.clear()
                # assert len(agent.pending_buffer) == 0, "Pending buffer is not empty"

                drt_info_list = []
                req_info_list = []
                total_num_accept = 0
                total_num_serve = 0
                for v in env.vehicle_list:
                    total_num_accept += v.num_accept
                    total_num_serve += v.num_serve
                    v.on_service_driving_time = env.curr_time - v.idle_time
                    v_info = {
                        'id': v.id,
                        'num_accept': v.num_accept,
                        'num_serve': v.num_serve,
                        'idle_time': v.idle_time,
                        'on_service_driving_time': v.on_service_driving_time
                    }
                    drt_info_list.append(v_info)

                total_waiting_time = 0
                total_in_vehicle_time = 0
                total_detour_time = 0
                served_count = 0
                for r in env.done_request_list:
                    r.detour_time = r.in_vehicle_time - r.travel_time
                    if r.status == RequestStatus.SERVED:
                        r_status = 'Served'
                        served_count += 1
                        total_waiting_time += r.waiting_time
                        total_in_vehicle_time += r.in_vehicle_time
                        total_detour_time += r.detour_time
                    else:
                        r_status = 'Canceled'
                    r_info = {
                        'id': r.id,
                        'status': r_status,
                        'waiting_time': r.waiting_time,
                        'in_vehicle_time': r.in_vehicle_time,
                        'detour_time': r.detour_time,
                    }
                    req_info_list.append(r_info)
                    req_info_list.sort(key=lambda x: x['id'])
                mean_waiting_time = total_waiting_time / served_count
                mean_in_vehicle_time = total_in_vehicle_time / served_count
                mean_detour_time = total_detour_time / served_count

                print('====== Ep: {} / Reward: {} / Loss: {} / eps: {} ======'.format(ep, total_reward, total_loss, agent.epsilon))
                e_info = {
                    'episode': ep,
                    'total_reward': total_reward,
                    'total_loss': total_loss,
                    'total_num_accept': total_num_accept,
                    'total_num_serve': total_num_serve,
                    'mean_waiting_time': mean_waiting_time,
                    'mean_in_vehicle_time': mean_in_vehicle_time,
                    'mean_detour_time': mean_detour_time,
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                log_episode(run_path, e_info)
                e_info_list.append(e_info)

                # Save Model
                if total_reward > best_reward:
                    best_reward = total_reward
                    model_name = "{}.h5".format(get_run_folder_name(config))
                    model_path = os.path.join(RESULT_PATH, model_name)
                    agent.save_model(model_path)

                break

            env.sync_state()
            state = env.state

        agent.decay_epsilon()

    log_all_episodes(run_path, e_info_list)


def main():
    env_builder = EnvBuilder(data_dir=DATA_PATH, result_dir=RESULT_PATH)

    for params in cfg.config_list:
        train_ddqn(env_builder, params)


if __name__ == "__main__":
    main()
