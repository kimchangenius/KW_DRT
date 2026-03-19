import os
import csv
import app.config as cfg
from app.env_builder import EnvBuilder
from app.agent import PPOAgent
from app.request_status import RequestStatus
from app.action_type import ActionType
import time

CURR_PATH = os.getcwd()
DATA_PATH = os.path.join(CURR_PATH, 'data')
RESULT_PATH = os.path.join(CURR_PATH, 'result')


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

    seq_list = info['event_sequence']
    filename = f'episode_{ep:03}_seq.txt'
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as f:
        for i, route in enumerate(seq_list):
            route_str = " -> ".join(route)
            f.write(f"DRT{i + 1}: {route_str}\n")


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
    mbs = config.get("mini_batch_size", "x")
    pi_lr = config.get("pi_lr", "x")
    vf_lr = config.get("vf_lr", "x")
    return f"hd{hd}_mbs{mbs}_pi{pi_lr}_vf{vf_lr}"


def train_ppo(env_builder, config, write_result=False):
    episodes = 1000

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< Training Session (PPO): {config_str} >>>>")

    if write_result:
        run_name = get_run_folder_name(config)
        run_path = os.path.join(RESULT_PATH, run_name)
        os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    mbs = config["mini_batch_size"]
    pi_lr = config["pi_lr"]
    vf_lr = config["vf_lr"]

    env = env_builder.build()
    agent = PPOAgent(hidden_dim=hd, pi_lr=pi_lr, vf_lr=vf_lr, mini_batch_size=mbs)

    e_info_list = []
    best_reward = float('-inf')

    for ep in range(1, episodes + 1):
        total_reward = 0.0
        total_loss = 0.0
        state = env.reset()

        veh_event_list = [[] for _ in range(len(env.vehicle_list))]

        start_time = time.time()

        while True:
            while env.has_idle_vehicle():
                action_mask = env.get_action_mask()
                action = agent.act(state, action_mask)
                log_prob = action[2]['log_prob']
                value = agent.get_value(state)

                env.enrich_action(action)
                if action[2]['type'] != ActionType.REJECT:
                    at = 'D' if action[2]['type'] == ActionType.DROPOFF else 'P'
                    seq = "{}_{}".format(at, action[2]['r_id'])
                    veh_event_list[action[0]].append(seq)

                next_state, reward, info = env.step(action)

                traj_idx = len(agent.trajectory)
                agent.store_transition(state, action, reward, action_mask, log_prob, value)

                if info['is_pending']:
                    action_id = action[2]['id']
                    agent.add_pending(action_id, traj_idx)

                if info['has_delayed_reward']:
                    for action_id in info['action_id_list']:
                        agent.confirm_pending(action_id, info['reward'])

                total_reward += reward
                state = next_state

            env.curr_time += 1
            d_reward_list = env.handle_time_update()

            for pair in d_reward_list:
                agent.confirm_pending(pair[0], pair[1])
                total_reward += pair[1]

            if env.is_done():
                if agent.trajectory:
                    agent.trajectory[-1]['done'] = True

                last_value = agent.get_value(state)
                loss = agent.train(last_value)
                if loss is not None:
                    total_loss = loss

                if len(agent.pending_actions) != 0:
                    print("[Warning] Pending actions not empty!")

                agent.clear_trajectory()

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
                mean_waiting_time = total_waiting_time / served_count if served_count else 0
                mean_in_vehicle_time = total_in_vehicle_time / served_count if served_count else 0
                mean_detour_time = total_detour_time / served_count if served_count else 0

                print('====== Ep: {} / Reward: {:.2f} / Loss: {:.4f} / Accept: {} / Serve: {} ======'.format(
                    ep, total_reward, total_loss, total_num_accept, total_num_serve))
                e_info = {
                    'episode': ep,
                    'total_reward': total_reward,
                    'total_loss': total_loss,
                    'total_num_accept': total_num_accept,
                    'total_num_serve': total_num_serve,
                    'mean_waiting_time': mean_waiting_time,
                    'mean_in_vehicle_time': mean_in_vehicle_time,
                    'mean_detour_time': mean_detour_time,
                    'event_sequence': veh_event_list,
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                if write_result:
                    log_episode(run_path, e_info)
                e_info_list.append(e_info)

                if total_reward > best_reward:
                    best_reward = total_reward
                    if write_result:
                        model_name = get_run_folder_name(config)
                        model_path = os.path.join(RESULT_PATH, model_name)
                        agent.save_model(model_path)

                break

            env.sync_state()
            state = env.state

        end_time = time.time()
        print(f"실행 시간: {end_time - start_time:.6f}초")

    if write_result:
        log_all_episodes(run_path, e_info_list)


def test_ppo(env_builder, config, write_result=False):
    episodes = 1

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< Test Session (PPO): {config_str} >>>>")

    hd = config["hidden_dim"]
    mbs = config.get("mini_batch_size", 64)
    pi_lr = config.get("pi_lr", 3e-4)
    vf_lr = config.get("vf_lr", 1e-3)

    model_name = get_run_folder_name(config)
    model_path = os.path.join(RESULT_PATH, model_name)
    env = env_builder.build()
    agent = PPOAgent(hidden_dim=hd, pi_lr=pi_lr, vf_lr=vf_lr, mini_batch_size=mbs)
    agent.load_model(model_path)

    e_info_list = []

    for ep in range(1, episodes + 1):
        total_reward = 0.0
        state = env.reset()
        veh_event_list = [[] for _ in range(len(env.vehicle_list))]

        while True:
            while env.has_idle_vehicle():
                action_mask = env.get_action_mask()
                action = agent.act_greedy(state, action_mask)

                env.enrich_action(action)
                if action[2]['type'] != ActionType.REJECT:
                    at = 'D' if action[2]['type'] == ActionType.DROPOFF else 'P'
                    seq = "{}_{}".format(at, action[2]['r_id'])
                    veh_event_list[action[0]].append(seq)

                next_state, reward, info = env.step(action)
                total_reward += reward
                state = next_state

            env.curr_time += 1
            env.handle_time_update()

            if env.is_done():
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
                mean_waiting_time = total_waiting_time / served_count if served_count else 0
                mean_in_vehicle_time = total_in_vehicle_time / served_count if served_count else 0
                mean_detour_time = total_detour_time / served_count if served_count else 0

                print('[Test] Ep: {} / Reward: {:.2f} / Accept: {} / Serve: {}'.format(
                    ep, total_reward, total_num_accept, total_num_serve))

                e_info = {
                    'episode': ep,
                    'total_reward': total_reward,
                    'total_loss': 0.0,
                    'total_num_accept': total_num_accept,
                    'total_num_serve': total_num_serve,
                    'mean_waiting_time': mean_waiting_time,
                    'mean_in_vehicle_time': mean_in_vehicle_time,
                    'mean_detour_time': mean_detour_time,
                    'event_sequence': veh_event_list,
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                if write_result:
                    log_episode(run_path, e_info)
                e_info_list.append(e_info)
                break

            env.sync_state()
            state = env.state

    # 다수 에피소드 평균 성능 출력
    n = len(e_info_list)
    mean_reward = sum(e['total_reward'] for e in e_info_list) / n
    mean_accept = sum(e['total_num_accept'] for e in e_info_list) / n
    mean_serve = sum(e['total_num_serve'] for e in e_info_list) / n
    mean_wait = sum(e['mean_waiting_time'] for e in e_info_list) / n
    mean_inv = sum(e['mean_in_vehicle_time'] for e in e_info_list) / n
    mean_detour = sum(e['mean_detour_time'] for e in e_info_list) / n

    print("\n" + "=" * 50)
    print(f"[PPO Evaluation] {episodes} episodes (act_greedy, deterministic)")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Mean Accept: {mean_accept:.1f}")
    print(f"  Mean Serve: {mean_serve:.1f}")
    print(f"  Mean Waiting Time: {mean_wait:.2f}")
    print(f"  Mean In-Vehicle Time: {mean_inv:.2f}")
    print(f"  Mean Detour Time: {mean_detour:.2f}")
    print("=" * 50)

    if write_result:
        log_all_episodes(run_path, e_info_list)


def main():
    env_builder = EnvBuilder(data_dir=DATA_PATH, result_dir=RESULT_PATH)

    # 학습
    for params in cfg.config_list:
        # train_ppo(env_builder, params, write_result=True)

        test_ppo(env_builder, params, write_result=False)


if __name__ == "__main__":
    main()
