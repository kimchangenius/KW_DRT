import os
import csv
import time

import app.config as cfg
from app.env_builder import EnvBuilder
from app.agent import DQNAgent
from app.request_status import RequestStatus
from app.action_type import ActionType
from app.vehicle_status import VehicleStatus
from app.state_builder import (
    capture_replay_frame,
    create_replay,
    save_simulation_replay_json,
)

CURR_PATH = os.getcwd()
DATA_PATH = os.path.join(CURR_PATH, 'data')
RESULT_PATH = os.path.join(CURR_PATH, 'result')


# ===========================================================================
# Logging
# ===========================================================================
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

    seq_list = info.get('event_sequence', [])
    if seq_list:
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
                         'Total Num. Cancel',
                         'Mean Waiting Time', 'Mean In-Vehicle Time', 'Mean Detour Time',
                         'Mean Occupancy'])
        for e in info_list:
            curr_row = [
                e['episode'],
                f"{e['total_reward']:.2f}",
                f"{e['total_loss']:.2f}",
                e['total_num_accept'],
                e['total_num_serve'],
                e['total_num_cancel'],
                f"{e['mean_waiting_time']:.2f}",
                f"{e['mean_in_vehicle_time']:.2f}",
                f"{e['mean_detour_time']:.2f}",
                f"{e['mean_occupancy']:.2f}",
            ]
            writer.writerow(curr_row)


def get_run_folder_name(config):
    hd = config.get("hidden_dim", "x")
    bs = config.get("batch_size", "x")
    lr = config.get("learning_rate", "x")
    return f"hd{hd}_bs{bs}_lr{lr}"


# ===========================================================================
# Helpers
# ===========================================================================
def _idle_vehicles(env):
    return [v for v in env.vehicle_list if v.status == VehicleStatus.IDLE]


def _fleet_total_passengers(env):
    return sum(v.num_passengers for v in env.vehicle_list)


def _next_pair_indices(env, vehicle_idx):
    """env 현 상태에서 같은 차량의 다음 페어 후보만 반환한다."""
    if vehicle_idx is None or vehicle_idx >= len(env.vehicle_list):
        return []
    vehicle = env.vehicle_list[vehicle_idx]
    if vehicle.status != VehicleStatus.IDLE:
        return []

    cands = env.enumerate_pair_candidates([vehicle], include_wait=True)
    if not any(c.get('is_real', 0) for c in cands[vehicle.id]):
        return []

    out = []
    for c in cands[vehicle.id]:
        out.append({
            'v_idx': c['v_idx'],
            'r_slot_idx': c['r_slot_idx'],
            'is_reject': c['is_reject'],
            'rel_feat': c['rel_feat'],
        })
    return out


def _make_transition(
    action, snapshot_pre, reward, next_snapshot, next_pair_indices,
    transition_id, done=False,
):
    """
    transition payload (dict):
        'snapshot'           : pre-decision env snapshot (current state)
        'vehicle_idx'        : vehicle whose Q(v,r) transition is being trained
        'action_pair'        : current pair index info
        'reward'             : float (delayed reward 누적될 수 있음)
        'next_snapshot'      : shared env snapshot AFTER action batch
        'next_pair_indices'  : List[dict] of same-vehicle feasible next pairs
        'done'               : bool
        'meta'               : {'id', 'action_id'}
    """
    return {
        'snapshot': snapshot_pre,
        'vehicle_idx': action['vehicle_idx'],
        'action_pair': action['pair_info'],
        'reward': reward,
        'next_snapshot': next_snapshot,
        'next_pair_indices': next_pair_indices,
        'done': done,
        'meta': {
            'id': transition_id,
            'action_id': action['action_id'],
            'vehicle_idx': action['vehicle_idx'],
        },
    }


def _remember_transition(agent, transition, info):
    if info['is_pending']:
        agent.pending(transition)
    else:
        agent.remember(transition)


def _confirm_delayed_rewards(agent, reward_items):
    total_reward = 0.0
    for action_id, reward in reward_items:
        agent.confirm_and_remember(action_id, reward)
        total_reward += reward
    return total_reward


def _delayed_items_from_info(info):
    if not info['has_delayed_reward']:
        return []
    reward = info['reward']
    return [(action_id, reward) for action_id in info['action_id_list']]


def _append_action_event(veh_event_list, action):
    if action['action_type'] == ActionType.REJECT:
        return
    r_obj = action['request']
    at = 'P' if action['action_type'] == ActionType.PICKUP else 'D'
    veh_event_list[action['vehicle_idx']].append(f"{at}_{r_obj.id}")


def build_transition_batch(
    env, snapshot_pre, action_records, next_snapshot, transition_id,
):
    transitions = []
    done = env.is_done()
    for action, reward, info in action_records:
        next_pair_indices = _next_pair_indices(env, action['vehicle_idx'])
        transition = _make_transition(
            action=action,
            snapshot_pre=snapshot_pre,
            reward=reward,
            next_snapshot=next_snapshot,
            next_pair_indices=next_pair_indices,
            transition_id=transition_id,
            done=done,
        )
        transitions.append((transition, info))
        transition_id += 1
    return transitions, transition_id


def _apply_action_batch(env, agent, snapshot_pre, actions, transition_id,
                        veh_event_list, training):
    action_records = []
    immediate_reward = 0.0
    train_steps = []

    for action in actions:
        r_obj = action['request']
        if r_obj is not None and r_obj not in env.active_request_list:
            continue
        _append_action_event(veh_event_list, action)

        reward, info = env.step(action)
        action_records.append((action, reward, info))
        immediate_reward += reward
        train_steps.append(env.curr_step)

    if not action_records:
        return transition_id, immediate_reward, 0.0, train_steps

    delayed_items = []
    total_loss = 0.0
    next_snapshot = env.get_snapshot()
    transitions, transition_id = build_transition_batch(
        env, snapshot_pre, action_records, next_snapshot, transition_id
    )

    if training:
        for transition, info in transitions:
            _remember_transition(agent, transition, info)
            delayed_items.extend(_delayed_items_from_info(info))
    else:
        for _, info in transitions:
            delayed_items.extend(_delayed_items_from_info(info))

    if training:
        immediate_reward += _confirm_delayed_rewards(agent, delayed_items)
    else:
        immediate_reward += sum(reward for _, reward in delayed_items)

    return transition_id, immediate_reward, total_loss, train_steps


def summarize_episode(env, episode, total_reward, total_loss, mean_occupancy,
                      veh_event_list=None):
    drt_info_list = []
    total_num_accept = 0
    total_num_serve = 0
    for v in env.vehicle_list:
        total_num_accept += v.num_accept
        total_num_serve += v.num_serve
        v.on_service_driving_time = env.curr_time - v.idle_time
        drt_info_list.append({
            'id': v.id,
            'num_accept': v.num_accept,
            'num_serve': v.num_serve,
            'idle_time': v.idle_time,
            'on_service_driving_time': v.on_service_driving_time,
        })

    req_info_list = []
    total_waiting_time = 0
    total_in_vehicle_time = 0
    total_detour_time = 0
    served_count = 0
    total_num_cancel = 0
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
            total_num_cancel += 1
        req_info_list.append({
            'id': r.id,
            'status': r_status,
            'waiting_time': r.waiting_time,
            'in_vehicle_time': r.in_vehicle_time,
            'detour_time': r.detour_time,
        })
    req_info_list.sort(key=lambda x: x['id'])

    return {
        'episode': episode,
        'total_reward': total_reward,
        'total_loss': total_loss,
        'total_num_accept': total_num_accept,
        'total_num_serve': total_num_serve,
        'total_num_cancel': total_num_cancel,
        'mean_waiting_time': total_waiting_time / served_count if served_count else 0,
        'mean_in_vehicle_time': total_in_vehicle_time / served_count if served_count else 0,
        'mean_detour_time': total_detour_time / served_count if served_count else 0,
        'mean_occupancy': mean_occupancy,
        'event_sequence': veh_event_list or [],
        'drt_info': drt_info_list,
        'request_info': req_info_list,
    }


def run_episode(
    env, agent, episode=0, training=False, transition_id=0,
    update_freq=10, final_train_steps=5, replay=None, replay_config=None,
):
    total_loss = 0.0
    total_reward = 0.0
    env.reset()

    occ_sum_pts = float(_fleet_total_passengers(env))
    occ_snapshots = 1
    veh_event_list = [[] for _ in range(len(env.vehicle_list))]
    capture_replay_frame(env, replay, replay_config)

    while True:
        while env.has_idle_vehicle():
            idle_vehicles = _idle_vehicles(env)
            candidates_by_v = env.enumerate_pair_candidates(
                idle_vehicles, include_wait=True
            )
            has_real_candidate = any(
                c.get('is_real', 0)
                for cand_list in candidates_by_v.values()
                for c in cand_list
            )
            if not has_real_candidate:
                break
            snapshot_pre = env.get_snapshot()
            actions = agent.act_pickup_assignments(
                env, snapshot=snapshot_pre, candidates_by_v=candidates_by_v
            )
            if not actions:
                break

            transition_id, reward, _, train_steps = _apply_action_batch(
                env, agent, snapshot_pre, actions, transition_id,
                veh_event_list, training,
            )
            total_reward += reward

            if training:
                for step in train_steps:
                    if step % update_freq == 0:
                        curr_loss = agent.train()
                        if curr_loss is not None:
                            total_loss += curr_loss

        env.curr_time += 1
        d_reward_list = env.handle_time_update()
        occ_sum_pts += float(_fleet_total_passengers(env))
        occ_snapshots += 1
        capture_replay_frame(env, replay, replay_config)

        if training:
            total_reward += _confirm_delayed_rewards(agent, d_reward_list)
        else:
            total_reward += sum(reward for _, reward in d_reward_list)

        if env.is_done():
            if training:
                last_transition = agent.replay_buffer.get_last()
                if last_transition is not None:
                    last_transition['done'] = True

                for _ in range(final_train_steps):
                    curr_loss = agent.train()
                    if curr_loss is not None:
                        total_loss += curr_loss

                if len(agent.pending_buffer) != 0:
                    print("[Warning] Pending Buffer is not empty!")
                    agent.pending_buffer.clear()

            mean_occupancy = (
                occ_sum_pts
                / max(occ_snapshots, 1)
                / max(cfg.MAX_NUM_VEHICLES, 1)
            )
            e_info = summarize_episode(
                env, episode, total_reward, total_loss,
                mean_occupancy, veh_event_list,
            )
            return e_info, transition_id


# ===========================================================================
# Train loop
# ===========================================================================
def train_ddqn(env_builder, config, write_result=False):
    episodes = 700
    update_freq = 25
    final_train_steps = 10

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< Training Session: {config_str} >>>>")

    if write_result:
        run_name = get_run_folder_name(config)
        run_path = os.path.join(RESULT_PATH, run_name)
        os.makedirs(run_path, exist_ok=True)

    env = env_builder.build()
    agent = DQNAgent(
        hidden_dim=config["hidden_dim"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        edge_weight_np=env.network.edge_weight,
    )

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')

    for ep in range(1, episodes + 1):
        start_time = time.time()
        e_info, transition_id = run_episode(
            env, agent, episode=ep, training=True,
            transition_id=transition_id, update_freq=update_freq,
            final_train_steps=final_train_steps,
        )
        e_info_list.append(e_info)

        print('====== Ep: {} / Reward: {:.2f} / Loss: {:.2f} / eps: {:.4f} / Served: {}/{} ======'.format(
            ep, e_info['total_reward'], e_info['total_loss'], agent.epsilon,
            e_info['total_num_serve'], len(env.done_request_list)))

        if write_result:
            log_episode(run_path, e_info)

        if e_info['total_reward'] > best_reward:
            best_reward = e_info['total_reward']
            if write_result:
                model_name = "{}.h5".format(get_run_folder_name(config))
                model_path = os.path.join(RESULT_PATH, model_name)
                agent.save_model(model_path)

        end_time = time.time()
        print(f"실행 시간: {end_time - start_time:.6f}초")
        agent.decay_epsilon()

    if write_result:
        log_all_episodes(run_path, e_info_list)


# ===========================================================================
# Test loop
# ===========================================================================
def test_ddqn(env_builder, config):
    print(f"\n<<<< Test Session: {config} >>>>")

    run_name = get_run_folder_name(config)
    run_path = os.path.join(RESULT_PATH, run_name, "_test")
    os.makedirs(run_path, exist_ok=True)

    model_path = os.path.join(RESULT_PATH, f"{run_name}.h5")

    env = env_builder.build()
    agent = DQNAgent(
        hidden_dim=config["hidden_dim"], batch_size=0, learning_rate=0,
        edge_weight_np=env.network.edge_weight,
    )
    agent.load_model(model_path)
    agent.epsilon = 0.0

    replay = create_replay()
    e_info, _ = run_episode(
        env, agent, episode=0, training=False,
        replay=replay, replay_config=config,
    )
    print(f"[TEST] Reward: {e_info['total_reward']:.2f} / Served: {e_info['total_num_serve']}/{len(env.done_request_list)}")
    log_episode(run_path, e_info)
    log_all_episodes(run_path, [e_info])
    json_path = save_simulation_replay_json(run_path, env, replay, config)
    print(f"[TEST] simulation replay saved: {json_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    # 학습용 수요 파일 — n=320 / horizon=240 시나리오로 학습
    request_filename = "requests_S1_seed0_n320.csv"
    # request_filename = "requests_S2_seed0_n320.csv"
    env_builder = EnvBuilder(
        data_dir=DATA_PATH, result_dir=RESULT_PATH,
        request_filename=request_filename,
    )

    for params in cfg.config_list:
        # train_ddqn(env_builder, params, write_result=True)
        test_ddqn(env_builder, params)



if __name__ == "__main__":
    main()
