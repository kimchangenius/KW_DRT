import os
import csv
import app.config as cfg
from pprint import pprint
from app.env_builder import EnvBuilder
from app.dqn_agent import DQNAgent
from app.ppo_agent import PPOAgent
from app.mappo_agent import MAPPOAgent
from app.request_status import RequestStatus
from app.action_type import ActionType
from app.vehicle_status import VehicleStatus
import time
import numpy as np

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
        writer.writerow(['Request ID', 'Status', 'Waiting Time', 'In-Vehicle Time', 'Detour Time', 'From Node', 'To Node'])
        for r in req_info_list:
            curr_row = [r['id'], r['status'], r['waiting_time'], r['in_vehicle_time'], r['detour_time'], r['from_node_id'], r['to_node_id']]
            writer.writerow(curr_row)

    seq_list = info['event_sequence']
    filename = f'episode_{ep:03}_seq.txt'
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as f:
        for i, route in enumerate(seq_list):
            route_str = " -> ".join(route)
            f.write(f"DRT{i + 1}: {route_str}\n")


def log_all_episodes(path, info_list, total_time):
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
    filename = 'episodes_time.txt'
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        # 총 실행 시간 포맷팅 (파일 저장용)
        if total_time < 60:
            time_str = f"{total_time:.2f}초"
        elif total_time < 3600:
            minutes = int(total_time // 60)
            seconds = total_time % 60
            time_str = f"{minutes}분 {seconds:.2f}초"
        else:
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = total_time % 60
            time_str = f"{hours}시간 {minutes}분 {seconds:.2f}초"
        f.write(f"Total Time: {time_str}")


def get_run_folder_name(config):
    hd = config.get("hidden_dim", "x")
    bs = config.get("batch_size", "x")
    lr = config.get("learning_rate", "x")
    return f"hd{hd}_bs{bs}_lr{lr}"


def train_ppo(env_builder, config, write_result=False, load_model=False):
    episodes = 50
    final_train_steps = 5

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< PPO Training Session: {config_str} >>>>")

    # Create Result Directory
    if write_result is True:
        ppo_folder_config = config.copy()
        ppo_folder_config["learning_rate"] = config["ppo_learning_rate"]
        run_name = get_run_folder_name(ppo_folder_config)
        run_path = os.path.join(RESULT_PATH, "ppo_" + run_name)
        os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config["ppo_learning_rate"]

    env = env_builder.build()
    agent = PPOAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)
    
    # 기존 모델 로드 시도
    if load_model is True:
        ppo_config = config.copy()
        ppo_config["learning_rate"] = config["ppo_learning_rate"]
        model_name = "{}.h5".format(get_run_folder_name(ppo_config))
        model_path = os.path.join(RESULT_PATH, model_name)
        agent.load_model(model_path)

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')
    total_time = 0.0
    
    for ep in range(1, episodes + 1):
        total_loss = 0.0
        total_reward = 0.0
        state = env.reset()

        delayed_reward_confirm = 0

        # veh_event_list는 더 이상 사용하지 않음 (환경에서 처리)

        start_time = time.time()

        while True:
            

            while env.has_idle_vehicle():
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
                    print(f"[Debug] Step {env.curr_step}: Pending Buffer에 추가 - {action[2]['r_id']}")
                    agent.pending(transition)
                else:
                    agent.remember(transition)

                if info['has_delayed_reward'] is True:
                    for action_id in info['action_id_list']:
                        d_reward = info['reward']
                        print(f"[Debug] Step {env.curr_step}: 즉시 완료로 인한 지연 보상 - {action_id} (보상: {d_reward})")
                        agent.confirm_and_remember(action_id, d_reward)
                        delayed_reward_confirm += 1

                # PPO는 더 적은 빈도로, 더 많은 데이터로 학습
                if agent.should_train():
                    curr_loss = agent.train()
                    if curr_loss is not None:
                        if isinstance(curr_loss, tuple) and len(curr_loss) >= 2:
                            total_loss += curr_loss[0] + curr_loss[1]  # actor_loss + critic_loss
                        else:
                            total_loss += curr_loss
                    
                    # Pending Buffer 상태 확인 (디버깅용)
                    if len(agent.pending_buffer) > 0:
                        print(f"[Debug] Step {env.curr_step}: Pending Buffer에 {len(agent.pending_buffer)}개 행동 대기 중")
                        for action_id in agent.pending_buffer.pending.keys():
                            print(f"  - {action_id}")
                    
                    # 행동 정보 출력 (디버깅용)
                    print(f"[Debug] Step {env.curr_step}: Vehicle {action[0]} -> {action[2]['type']} (Request {action[2]['r_id']})")

                total_reward += reward
                state = next_state

            env.curr_time += 1
            d_reward_list, cancelled_request_ids = env.handle_time_update()
            
            # 실제 픽업/드롭오프 발생 시 이벤트 기록 (제거 - 환경에서 처리)

            for pair in d_reward_list:
                action_id, reward = pair
                print(f"[Debug] Step {env.curr_step}: 시간 업데이트로 인한 지연 보상 - {action_id} (보상: {reward})")
                agent.confirm_and_remember(action_id, reward)
                delayed_reward_confirm += 1

            # 취소된 request 관련 pending action 즉시 제거
            for rid in cancelled_request_ids:
                agent.pending_buffer.cancel(f"{rid}_1")  # PICKUP

            if env.is_done():
                # 에피소드 완료 시 처리
                if len(agent.episode_buffer) > 0:
                    agent.episode_buffer[-1][4] = True  # 마지막 transition의 done을 True로 설정
                
                # 에피소드 완료 후 trajectory buffer에 추가
                agent.finish_episode()

                # 에피소드 끝에서 학습 수행
                for _ in range(final_train_steps):
                    if agent.should_train():
                        curr_loss = agent.train()
                        if curr_loss is not None:
                            if isinstance(curr_loss, tuple) and len(curr_loss) >= 2:
                                total_loss += curr_loss[0] + curr_loss[1]
                            else:
                                total_loss += curr_loss

                if len(agent.pending_buffer) != 0:
                    print("[Warning] Pending Buffer is not empty!")
                    for k, v in agent.pending_buffer.pending.items():
                        print(k)
                        #pprint(v)
                    
                    # Pending Buffer 정보를 파일로 저장
                    if write_result is True:
                        pending_log_filename = f'episode_{ep:03}_pending_buffer_log.txt'
                        pending_log_filepath = os.path.join(run_path, pending_log_filename)
                        with open(pending_log_filepath, 'w') as f:
                            f.write(f"Episode: {ep}\n")
                            f.write(f"Step: {env.curr_step}\n")
                            f.write(f"Time: {env.curr_time}\n")
                            f.write(f"Pending Buffer 크기: {len(agent.pending_buffer)}\n")
                            f.write("남은 Action IDs 및 Request 정보:\n")
                            for action_id in agent.pending_buffer.pending.keys():
                                # Action ID에서 Request ID 추출
                                request_id = action_id.split('_')[0]
                                action_type = action_id.split('_')[1]
                                
                                # Request 정보 찾기
                                request_info = "Unknown"
                                for r in env.done_request_list + env.active_request_list:
                                    if str(r.id) == request_id:
                                        action_type_name = "PICKUP" if action_type == "1" else "DROPOFF"
                                        request_info = f"Request {r.id}: {r.from_node_id} -> {r.to_node_id} ({action_type_name})"
                                        break
                                
                                f.write(f"  - {action_id}: {request_info}\n")
                    
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
                        'from_node_id': r.from_node_id,
                        'to_node_id': r.to_node_id
                    }
                    req_info_list.append(r_info)
                    req_info_list.sort(key=lambda x: x['id'])
                    
                # ZeroDivisionError 방지
                if served_count > 0:
                    mean_waiting_time = total_waiting_time / served_count
                    mean_in_vehicle_time = total_in_vehicle_time / served_count
                    mean_detour_time = total_detour_time / served_count
                else:
                    mean_waiting_time = 0.0
                    mean_in_vehicle_time = 0.0
                    mean_detour_time = 0.0

                print('====== Ep: {} / Reward: {} / Loss: {} / LR: {:.6f} ======'.format(
                    ep, total_reward, total_loss, agent.current_learning_rate))
                e_info = {
                    'episode': ep,
                    'total_reward': total_reward,
                    'total_loss': total_loss,
                    'total_num_accept': total_num_accept,
                    'total_num_serve': total_num_serve,
                    'mean_waiting_time': mean_waiting_time,
                    'mean_in_vehicle_time': mean_in_vehicle_time,
                    'mean_detour_time': mean_detour_time,
                    'event_sequence': env.event_sequences if hasattr(env, 'event_sequences') else veh_event_list,
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                if write_result is True:
                    log_episode(run_path, e_info)
                e_info_list.append(e_info)

                # Save Model
                if total_reward > best_reward:
                    best_reward = total_reward
                    if write_result is True:
                        # PPO 전용 config 사용
                        ppo_save_config = config.copy()
                        ppo_save_config["learning_rate"] = config["ppo_learning_rate"]
                        model_name = "{}.h5".format(get_run_folder_name(ppo_save_config))
                        model_path = os.path.join(RESULT_PATH, model_name)
                        agent.save_model(model_path)

                break

            env.sync_state()
            state = env.state

        end_time = time.time()
        progress_time = end_time - start_time
        total_time += progress_time
        print(f"실행 시간: {progress_time:.6f}초")

    # 총 실행 시간
    if total_time < 60:
        print(f"총 실행 시간: {total_time:.6f}초")
    elif total_time < 3600:
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f"총 실행 시간: {minutes}분 {seconds:.6f}초")
    else:
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"총 실행 시간: {hours}시간 {minutes}분 {seconds:.6f}초")

    if write_result is True:
        log_all_episodes(run_path, e_info_list, total_time)


def train_ddqn(env_builder, config, write_result=False, load_model=False):
    episodes = 50
    update_freq = 10
    final_train_steps = 5

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< Training Session: {config_str} >>>>")

    # Create Result Directory
    if write_result is True:
        dqn_folder_config = config.copy()
        dqn_folder_config["learning_rate"] = config["dqn_learning_rate"]
        run_name = get_run_folder_name(dqn_folder_config)
        run_path = os.path.join(RESULT_PATH, "dqn_" + run_name)
        os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config["dqn_learning_rate"]

    env = env_builder.build()
    agent = DQNAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)
    
    # 기존 모델 로드 시도
    # DQN 전용 config 생성 (dqn_learning_rate를 learning_rate로 변환)
    if load_model is True:
        dqn_config = config.copy()
        dqn_config["learning_rate"] = config["dqn_learning_rate"]
        model_name = "{}.h5".format(get_run_folder_name(dqn_config))
        model_path = os.path.join(RESULT_PATH, model_name)
        agent.load_model(model_path)

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')
    total_time = 0.0

    for ep in range(1, episodes + 1):
        # print('\n============ Ep : {} ============'.format(ep))
        total_loss = 0.0
        total_reward = 0.0
        state = env.reset()

        delayed_reward_confirm = 0

        # veh_event_list는 더 이상 사용하지 않음 (환경에서 처리)

        start_time = time.time()

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
                    print(f"[Debug] Step {env.curr_step}: Pending Buffer에 추가 - {action[2]['r_id']}")
                    agent.pending(transition)
                else:
                    agent.remember(transition)

                if info['has_delayed_reward'] is True:
                    for action_id in info['action_id_list']:
                        d_reward = info['reward']
                        print(f"[Debug] Step {env.curr_step}: 즉시 완료로 인한 지연 보상 - {action_id} (보상: {d_reward})")
                        agent.confirm_and_remember(action_id, d_reward)
                        delayed_reward_confirm += 1

                if env.curr_step % update_freq == 0:
                    curr_loss = agent.train()
                    if curr_loss is not None:
                        total_loss += curr_loss
                    
                    # Pending Buffer 상태 확인 (디버깅용)
                    if len(agent.pending_buffer) > 0:
                        print(f"[Debug] Step {env.curr_step}: Pending Buffer에 {len(agent.pending_buffer)}개 행동 대기 중")
                        for action_id in agent.pending_buffer.pending.keys():
                            print(f"  - {action_id}")
                    
                    # 행동 정보 출력 (디버깅용)
                    print(f"[Debug] Step {env.curr_step}: Vehicle {action[0]} -> {action[2]['type']} (Request {action[2]['r_id']})")

                total_reward += reward
                state = next_state

            env.curr_time += 1
            d_reward_list, cancelled_request_ids = env.handle_time_update()
            
            # 실제 픽업/드롭오프 발생 시 이벤트 기록 (제거 - 환경에서 처리)

            for pair in d_reward_list:
                action_id, reward = pair
                print(f"[Debug] Step {env.curr_step}: 시간 업데이트로 인한 지연 보상 - {action_id} (보상: {reward})")
                agent.confirm_and_remember(action_id, reward)
                delayed_reward_confirm += 1

            # 취소된 request 관련 pending action 즉시 제거
            for rid in cancelled_request_ids:
                agent.pending_buffer.cancel(f"{rid}_1")  # PICKUP

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
                    for k, v in agent.pending_buffer.pending.items():
                        print(k)
                        #pprint(v)
                    
                    # Pending Buffer 정보를 파일로 저장
                    if write_result is True:
                        pending_log_filename = f'episode_{ep:03}_pending_buffer_log.txt'
                        pending_log_filepath = os.path.join(run_path, pending_log_filename)
                        with open(pending_log_filepath, 'w') as f:
                            f.write(f"Episode: {ep}\n")
                            f.write(f"Step: {env.curr_step}\n")
                            f.write(f"Time: {env.curr_time}\n")
                            f.write(f"Pending Buffer 크기: {len(agent.pending_buffer)}\n")
                            f.write("남은 Action IDs 및 Request 정보:\n")
                            for action_id in agent.pending_buffer.pending.keys():
                                # Action ID에서 Request ID 추출
                                request_id = action_id.split('_')[0]
                                action_type = action_id.split('_')[1]
                                
                                # Request 정보 찾기
                                request_info = "Unknown"
                                for r in env.done_request_list + env.active_request_list:
                                    if str(r.id) == request_id:
                                        action_type_name = "PICKUP" if action_type == "1" else "DROPOFF"
                                        request_info = f"Request {r.id}: {r.from_node_id} -> {r.to_node_id} ({action_type_name})"
                                        break
                                
                                f.write(f"  - {action_id}: {request_info}\n")
                    
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
                        'from_node_id': r.from_node_id,
                        'to_node_id': r.to_node_id
                    }
                    req_info_list.append(r_info)
                    req_info_list.sort(key=lambda x: x['id'])
                    
                # ZeroDivisionError 방지
                if served_count > 0:
                    mean_waiting_time = total_waiting_time / served_count
                    mean_in_vehicle_time = total_in_vehicle_time / served_count
                    mean_detour_time = total_detour_time / served_count
                else:
                    mean_waiting_time = 0.0
                    mean_in_vehicle_time = 0.0
                    mean_detour_time = 0.0

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
                    'event_sequence': env.event_sequences if hasattr(env, 'event_sequences') else veh_event_list,
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                if write_result is True:
                    log_episode(run_path, e_info)
                e_info_list.append(e_info)

                # Save Model
                if total_reward > best_reward:
                    best_reward = total_reward
                    if write_result is True:
                        # DQN 전용 config 사용
                        dqn_save_config = config.copy()
                        dqn_save_config["learning_rate"] = config["dqn_learning_rate"]
                        model_name = "{}.h5".format(get_run_folder_name(dqn_save_config))
                        model_path = os.path.join(RESULT_PATH, model_name)
                        agent.save_model(model_path)

                break

            env.sync_state()
            state = env.state

        end_time = time.time()
        progress_time = end_time - start_time
        total_time += progress_time
        print(f"실행 시간: {progress_time:.6f}초")

        agent.decay_epsilon()

    # 총 실행 시간
    if total_time < 60:
        print(f"총 실행 시간: {total_time:.6f}초")
    elif total_time < 3600:
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f"총 실행 시간: {minutes}분 {seconds:.6f}초")
    else:
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"총 실행 시간: {hours}시간 {minutes}분 {seconds:.6f}초")

    if write_result is True:
        log_all_episodes(run_path, e_info_list, total_time)


def train_mappo(env_builder, config, write_result=False, load_model=False):
    episodes = 50
    final_train_steps = 5

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< MAPPO Training Session: {config_str} >>>>")

    # Create Result Directory
    if write_result is True:
        mappo_folder_config = config.copy()
        mappo_folder_config["learning_rate"] = config.get("mappo_learning_rate", config["learning_rate"])
        run_name = get_run_folder_name(mappo_folder_config)
        run_path = os.path.join(RESULT_PATH, "mappo_" + run_name)
        os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config.get("mappo_learning_rate", config["learning_rate"])

    env = env_builder.build()
    agent = MAPPOAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)
    
    # 기존 모델 로드 시도
    if load_model is True:
        mappo_config = config.copy()
        mappo_config["learning_rate"] = config.get("mappo_learning_rate", config["learning_rate"])
        model_name = "{}.h5".format(get_run_folder_name(mappo_config))
        model_path = os.path.join(RESULT_PATH, model_name)
        agent.load_model(model_path)

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')
    total_time = 0.0
    
    for ep in range(1, episodes + 1):
        total_loss = 0.0
        total_reward = 0.0
        state = env.reset()
        
        # Learning rate 업데이트
        agent.update_learning_rate(ep)

        delayed_reward_confirm = 0

        # veh_event_list는 더 이상 사용하지 않음 (환경에서 처리)

        start_time = time.time()

        while True:
            
            while env.has_idle_vehicle():
                action_mask = env.get_action_mask()
                action = agent.act(state, action_mask, env)
                
                # MAPPO의 경우 모든 액션을 동시에 처리
                if action[2]['mode'] == 'multi_agent_simultaneous':
                    # 모든 액션을 환경에 전달
                    all_actions = action[2]['all_actions']
                    
                    # 각 액션에 대해 enrich_action 수행
                    enriched_actions = []
                    for act in all_actions:
                        enriched_action = act.copy()
                        enriched_action.append({'r_id': None, 'type': None, 'id': None})
                        env.enrich_action(enriched_action)
                        enriched_actions.append(enriched_action)
                    
                    # 동시 실행
                    next_state, reward, info = env.step_multi(enriched_actions)
                    next_action_mask = env.get_action_mask()
                    
                    # 디버깅 정보 출력
                    print(f"[Debug] Step {env.curr_step}: {len(all_actions)}개 에이전트 동시 실행")
                    for i, act in enumerate(all_actions):
                        if act[1] == cfg.POSSIBLE_ACTION - 1:  # REJECT 액션
                            print(f"  - Vehicle {act[0]} -> REJECT")
                        else:
                            print(f"  - Vehicle {act[0]} -> Request {enriched_actions[i][2]['r_id']}")
                else:
                    # 기존 방식 (단일 액션)
                    env.enrich_action(action)
                    next_state, reward, info = env.step(action)
                    next_action_mask = env.get_action_mask()
                    

                # MAPPO의 경우 모든 액션에 대한 transition 저장
                if action[2]['mode'] == 'multi_agent_simultaneous':
                    all_actions = action[2]['all_actions']
                    individual_rewards = info.get('individual_rewards', [reward] * len(all_actions))
                    individual_infos = info.get('individual_infos', [info] * len(all_actions))
                    
                    for i, (act, individual_reward, individual_info) in enumerate(zip(all_actions, individual_rewards, individual_infos)):
                        t_info = {
                            'id': transition_id,
                            'm': action_mask,
                            'nm': next_action_mask,
                        }
                        
                        # 개별 액션 정보로 transition 생성
                        individual_action = act.copy()
                        individual_action.append({
                            'r_id': enriched_actions[i][2]['r_id'],
                            'type': enriched_actions[i][2]['type'],
                            'id': enriched_actions[i][2]['id'],
                            'action_prob': action[2]['all_probs'][i],
                            'agent_id': act[0]
                        })
                        
                        transition = [state, individual_action, individual_reward, next_state, False, t_info]
                        transition_id += 1
                        
                        if individual_info['is_pending'] is True:
                            print(f"[Debug] Step {env.curr_step}: Pending Buffer에 추가 - {individual_action[2]['r_id']}")
                            agent.pending(transition)
                        else:
                            agent.remember(transition)
                        
                        # 지연 보상 처리
                        if individual_info['has_delayed_reward'] is True:
                            for action_id in individual_info['action_id_list']:
                                d_reward = individual_info['reward']
                                print(f"[Debug] Step {env.curr_step}: 즉시 완료로 인한 지연 보상 - {action_id} (보상: {d_reward})")
                                agent.confirm_and_remember(action_id, d_reward)
                                delayed_reward_confirm += 1
                else:
                    # 기존 방식 (단일 액션)
                    t_info = {
                        'id': transition_id,
                        'm': action_mask,
                        'nm': next_action_mask,
                    }

                    transition = [state, action, reward, next_state, False, t_info]
                    transition_id += 1
                    if info['is_pending'] is True:
                        print(f"[Debug] Step {env.curr_step}: Pending Buffer에 추가 - {action[2]['r_id']}")
                        agent.pending(transition)
                    else:
                        agent.remember(transition)

                    if info['has_delayed_reward'] is True:
                        for action_id in info['action_id_list']:
                            d_reward = info['reward']
                            print(f"[Debug] Step {env.curr_step}: 즉시 완료로 인한 지연 보상 - {action_id} (보상: {d_reward})")
                            agent.confirm_and_remember(action_id, d_reward)
                            delayed_reward_confirm += 1

                # MAPPO도 에피소드 기반 학습
                if agent.should_train():
                    curr_loss = agent.train()
                    if curr_loss is not None:
                        if isinstance(curr_loss, tuple) and len(curr_loss) >= 2:
                            total_loss += curr_loss[0] + curr_loss[1]  # actor_loss + critic_loss
                        else:
                            total_loss += curr_loss
                    
                    # Pending Buffer 상태 확인 (디버깅용)
                    if len(agent.pending_buffer) > 0:
                        print(f"[Debug] Step {env.curr_step}: Pending Buffer에 {len(agent.pending_buffer)}개 행동 대기 중")
                        for action_id in agent.pending_buffer.pending.keys():
                            print(f"  - {action_id}")
                    
                    # 행동 정보 출력 (디버깅용)
                    print(f"[Debug] Step {env.curr_step}: Vehicle {action[0]} -> {action[2]['type']} (Request {action[2]['r_id']})")

                total_reward += reward
                state = next_state

            env.curr_time += 1
            d_reward_list, cancelled_request_ids = env.handle_time_update()
            
            # 실제 픽업/드롭오프 발생 시 이벤트 기록 (제거 - 환경에서 처리)

            for pair in d_reward_list:
                action_id, reward = pair
                print(f"[Debug] Step {env.curr_step}: 시간 업데이트로 인한 지연 보상 - {action_id} (보상: {reward})")
                agent.confirm_and_remember(action_id, reward)
                delayed_reward_confirm += 1

            # 취소된 request 관련 pending action 즉시 제거
            for rid in cancelled_request_ids:
                agent.pending_buffer.cancel(f"{rid}_1")  # PICKUP

            if env.is_done():
                # 에피소드 완료 시 처리
                if len(agent.episode_buffer) > 0:
                    agent.episode_buffer[-1][4] = True  # 마지막 transition의 done을 True로 설정
                
                # 에피소드 완료 후 trajectory buffer에 추가
                agent.finish_episode()

                # 에피소드 끝에서 학습 수행
                for _ in range(final_train_steps):
                    if agent.should_train():
                        curr_loss = agent.train()
                        if curr_loss is not None:
                            if isinstance(curr_loss, tuple) and len(curr_loss) >= 2:
                                total_loss += curr_loss[0] + curr_loss[1]
                            else:
                                total_loss += curr_loss

                if len(agent.pending_buffer) != 0:
                    print("[Warning] Pending Buffer is not empty!")
                    for k, v in agent.pending_buffer.pending.items():
                        print(k)
                    
                    # Pending Buffer 정보를 파일로 저장
                    if write_result is True:
                        pending_log_filename = f'episode_{ep:03}_pending_buffer_log.txt'
                        pending_log_filepath = os.path.join(run_path, pending_log_filename)
                        with open(pending_log_filepath, 'w') as f:
                            f.write(f"Episode: {ep}\n")
                            f.write(f"Step: {env.curr_step}\n")
                            f.write(f"Time: {env.curr_time}\n")
                            f.write(f"Pending Buffer 크기: {len(agent.pending_buffer)}\n")
                            f.write("남은 Action IDs 및 Request 정보:\n")
                            for action_id in agent.pending_buffer.pending.keys():
                                # Action ID에서 Request ID 추출
                                request_id = action_id.split('_')[0]
                                action_type = action_id.split('_')[1]
                                
                                # Request 정보 찾기
                                request_info = "Unknown"
                                for r in env.done_request_list + env.active_request_list:
                                    if str(r.id) == request_id:
                                        action_type_name = "PICKUP" if action_type == "1" else "DROPOFF"
                                        request_info = f"Request {r.id}: {r.from_node_id} -> {r.to_node_id} ({action_type_name})"
                                        break
                                
                                f.write(f"  - {action_id}: {request_info}\n")
                    
                    agent.pending_buffer.clear()

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
                        'from_node_id': r.from_node_id,
                        'to_node_id': r.to_node_id
                    }
                    req_info_list.append(r_info)
                    req_info_list.sort(key=lambda x: x['id'])
                    
                # ZeroDivisionError 방지
                if served_count > 0:
                    mean_waiting_time = total_waiting_time / served_count
                    mean_in_vehicle_time = total_in_vehicle_time / served_count
                    mean_detour_time = total_detour_time / served_count
                else:
                    mean_waiting_time = 0.0
                    mean_in_vehicle_time = 0.0
                    mean_detour_time = 0.0

                print('====== Ep: {} / Reward: {} / Loss: {} / LR: {:.6f} ======'.format(
                    ep, total_reward, total_loss, agent.current_learning_rate))
                e_info = {
                    'episode': ep,
                    'total_reward': total_reward,
                    'total_loss': total_loss,
                    'total_num_accept': total_num_accept,
                    'total_num_serve': total_num_serve,
                    'mean_waiting_time': mean_waiting_time,
                    'mean_in_vehicle_time': mean_in_vehicle_time,
                    'mean_detour_time': mean_detour_time,
                    'event_sequence': env.event_sequences if hasattr(env, 'event_sequences') else veh_event_list,
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                if write_result is True:
                    log_episode(run_path, e_info)
                e_info_list.append(e_info)

                # Save Model
                if total_reward > best_reward:
                    best_reward = total_reward
                    if write_result is True:
                        # MAPPO 전용 config 사용
                        mappo_save_config = config.copy()
                        mappo_save_config["learning_rate"] = config.get("mappo_learning_rate", config["learning_rate"])
                        model_name = "{}.h5".format(get_run_folder_name(mappo_save_config))
                        model_path = os.path.join(RESULT_PATH, model_name)
                        agent.save_model(model_path)

                break

            env.sync_state()
            state = env.state

        end_time = time.time()
        progress_time = end_time - start_time
        total_time += progress_time
        print(f"실행 시간: {progress_time:.6f}초")

    # 총 실행 시간
    if total_time < 60:
        print(f"총 실행 시간: {total_time:.6f}초")
    elif total_time < 3600:
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f"총 실행 시간: {minutes}분 {seconds:.6f}초")
    else:
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"총 실행 시간: {hours}시간 {minutes}분 {seconds:.6f}초")

    if write_result is True:
        log_all_episodes(run_path, e_info_list, total_time)


def main():
    env_builder = EnvBuilder(data_dir=DATA_PATH, result_dir=RESULT_PATH)


    for params in cfg.config_list:
        train_ddqn(env_builder, params, write_result=True, load_model=False)
        train_ppo(env_builder, params, write_result=True, load_model=False)
        train_mappo(env_builder, params, write_result=True, load_model=False)


if __name__ == "__main__":
    main()
