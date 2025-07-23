import os
import csv
import app.config as cfg
from pprint import pprint
from app.env_builder import EnvBuilder
from app.dqn_agent import DQNAgent
from app.ddpg_agent import DDPGAgent
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
            route_str = " -> ".join([f"{event}({node})" for event, node in route])
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
    bs = config.get("batch_size", "x")
    lr = config.get("learning_rate", "x")
    return f"hd{hd}_bs{bs}_lr{lr}"


def train_ddqn(env_builder, config, write_result=False):
    episodes = 500
    update_freq = 10
    final_train_steps = 5

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< Training Session: {config_str} >>>>")

    # Create Result Directory
    if write_result is True:
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

        veh_event_list = []
        for _ in range(len(env.vehicle_list)):
            veh_event_list.append([])

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
                if action[2]['type'] != ActionType.REJECT:
                    at = 'D'
                    if action[2]['type'] == ActionType.PICKUP:
                        at = 'P'
                    seq = "{}_{}".format(at, action[2]['r_id'])
                    # 차량의 현재 위치와 함께 이벤트 저장
                    vehicle = env.vehicle_list[action[0]]
                    event_with_location = (seq, vehicle.curr_node)
                    veh_event_list[action[0]].append(event_with_location)
                    # print(seq)

                next_state, reward, info = env.dqn_step(action)
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
                    'event_sequence': veh_event_list,
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
                        model_name = "{}.h5".format(get_run_folder_name(config))
                        model_path = os.path.join(RESULT_PATH, model_name)
                        agent.save_model(model_path)

                break

            env.sync_state()
            state = env.state

        end_time = time.time()
        print(f"실행 시간: {end_time - start_time:.6f}초")

        agent.decay_epsilon()

    if write_result is True:
        log_all_episodes(run_path, e_info_list)

def train_ddpg(env_builder, config, write_result=False):
    episodes = 1
    update_freq = 5
    final_train_steps = 5

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< DDPG Training Session: {config_str} >>>>")

    # Create Result Directory
    if write_result is True:
        run_name = get_run_folder_name(config)
        run_path = os.path.join(RESULT_PATH, run_name)
        os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config["learning_rate"]

    env = env_builder.build()
    agent = DDPGAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')

    for ep in range(1, episodes + 1):
        total_loss = 0.0
        total_reward = 0.0
        state = env.reset()
        
        # === 에피소드 시작: 전체 요청 목록 출력 ===
        print(f"\n{'='*50}")
        print(f"Episode {ep} 시작 - 전체 요청 목록:")
        print(f"{'='*50}")
        all_requests = sorted(env.original_request_list, key=lambda r: r.request_time)
        for r in all_requests:
            print(f"  Request {r.id}: Node {r.from_node_id} → {r.to_node_id} (Time {r.request_time})")
        print(f"총 {len(all_requests)}개 요청")
        print(f"{'='*50}")
        
        # 에피소드 시작 시 noise 리셋
        agent.reset_noise()

        veh_event_list = [[] for _ in range(len(env.vehicle_list))]

        start_time = time.time()

        while True:
            while env.has_idle_vehicle():
                # 에피소드 진행에 따른 noise scale 적용
                raw_action = agent.act(state, add_noise=True, noise_scale=max(0.1, 1.0 - ep / episodes))
                
                # === Action 전처리로 더 부드러운 학습 ===
                action = agent.preprocess_action(raw_action)
                
                # 연속 action을 환경에 전달 (IDLE 차량만 실제 행동)
                next_state, reward, info = env.ddpg_step(action)

                # === Event 저장 ===
                if 'action_details' in info:
                    for detail in info['action_details']:
                        if 'chosen_action' in detail and detail['chosen_action'] != 'reject':
                            v_id = detail['vehicle']
                            action_type = detail['chosen_action']
                            
                            if action_type == 'pickup' and 'request_id' in detail:
                                at = 'P'
                                seq = "{}_{}".format(at, detail['request_id'])
                                # 차량의 현재 위치와 함께 이벤트 저장
                                vehicle = env.vehicle_list[v_id]
                                event_with_location = (seq, vehicle.curr_node)
                                veh_event_list[v_id].append(event_with_location)
                            elif action_type == 'dropoff' and 'request_id' in detail:
                                at = 'D'
                                seq = "{}_{}".format(at, detail['request_id'])
                                # 차량의 현재 위치와 함께 이벤트 저장
                                vehicle = env.vehicle_list[v_id]
                                event_with_location = (seq, vehicle.curr_node)
                                veh_event_list[v_id].append(event_with_location)
                
                # === Transition 저장 (IDLE 차량 행동마다) ===
                t_info = {
                    'id': transition_id,
                }

                transition = [state, action, reward, next_state, False, t_info]
                transition_id += 1
                agent.remember(transition)

                # === 학습 (step마다) ===
                if env.curr_step % update_freq == 0:
                    train_result = agent.train()
                    if train_result is not None:
                        actor_loss, critic_loss = train_result
                        total_loss += actor_loss + critic_loss
                    
                    # === DQN과 동일한 차량 행동 출력 ===
                    print(f"[Debug] Step {env.curr_step}: Raw Action: {raw_action}")
                    print(f"  Final Action: {action}")
                    print(f"  Total Reward: {reward:.3f}")
                    
                    # 각 차량의 구체적인 행동 출력
                    if 'action_details' in info:
                        for detail in info['action_details']:
                            if 'chosen_action' in detail:
                                v_id = detail['vehicle']
                                action_type = detail['chosen_action']
                                reward_val = detail.get('reward', 0)
                                has_pass = detail.get('has_passengers', False)
                                
                                if action_type == 'pickup' and 'request_id' in detail:
                                    print(f"  Vehicle {v_id} -> PICKUP (Request {detail['request_id']}) | Reward: {reward_val:.3f}")
                                elif action_type == 'pickup' and 'reason' in detail:
                                    reason = detail['reason']
                                    if reason == 'no_valid_requests':
                                        print(f"  Vehicle {v_id} -> PICKUP (유효한 요청 없음) | Penalty: {reward_val:.3f}")
                                    elif reason == 'pickup_with_passengers':
                                        print(f"  Vehicle {v_id} -> PICKUP (이미 승객 있음) | Penalty: {reward_val:.3f}")
                                    else:
                                        print(f"  Vehicle {v_id} -> PICKUP ({reason}) | Penalty: {reward_val:.3f}")
                                elif action_type == 'pickup':
                                    print(f"  Vehicle {v_id} -> PICKUP (상세정보 없음) | Reward: {reward_val:.3f}")
                                elif action_type == 'dropoff' and 'request_id' in detail:
                                    print(f"  Vehicle {v_id} -> DROPOFF (Request {detail['request_id']}) | Reward: {reward_val:.3f}")
                                elif action_type == 'dropoff' and 'reason' in detail:
                                    reason = detail['reason']
                                    if reason == 'no_passengers_to_dropoff':
                                        print(f"  Vehicle {v_id} -> DROPOFF (드롭오프할 승객 없음) | Penalty: {reward_val:.3f}")
                                    else:
                                        print(f"  Vehicle {v_id} -> DROPOFF ({reason}) | Penalty: {reward_val:.3f}")
                                elif action_type == 'dropoff':
                                    print(f"  Vehicle {v_id} -> DROPOFF (상세정보 없음) | Reward: {reward_val:.3f}")
                                elif action_type == 'reject':
                                    status = "승객 O" if has_pass else "승객 X"
                                    print(f"  Vehicle {v_id} -> REJECT ({status}) | Penalty: {reward_val:.3f}")
                                else:
                                    print(f"  Vehicle {v_id} -> {action_type.upper()} | Reward: {reward_val:.3f}")
                        
                        print("-" * 40)

                total_reward += reward
                state = next_state

            # === 시간 진행 및 환경 업데이트  ===
            prev_active_count = len(env.active_request_list)
            prev_future_count = len(env.future_request_list)
            
            env.curr_time += 1
            env.handle_time_update_ddpg()
            
            # === 요청 활성화 시 상태 출력 ===
            current_active_count = len(env.active_request_list)
            current_future_count = len(env.future_request_list)
            
            # 새로운 요청이 활성화되었거나 상태가 변경된 경우
            if current_active_count != prev_active_count or current_future_count != prev_future_count:
                print(f"[Time {env.curr_time}] 요청 상태 변화:")
                print(f"  Future: {current_future_count}개 (이전: {prev_future_count}개)")
                print(f"  Active: {current_active_count}개 (이전: {prev_active_count}개)")
                
                if len(env.future_request_list) > 0:
                    next_requests = []
                    for r in env.future_request_list[:5]:  # 최대 5개만 표시
                        next_requests.append(f"R{r.id}(T{r.request_time})")
                    print(f"  다음 요청들: {', '.join(next_requests)}")
                
                if len(env.active_request_list) > 0:
                    active_requests = []
                    for r in env.active_request_list[:5]:  # 최대 5개만 표시
                        status_short = str(r.status)[0]  # 첫 글자만
                        active_requests.append(f"R{r.id}({status_short})")
                    print(f"  활성 요청들: {', '.join(active_requests)}")
                
                if len(env.done_request_list) > 0:
                    done_count = len(env.done_request_list)
                    served_count = sum(1 for r in env.done_request_list if r.status == RequestStatus.SERVED)
                    cancelled_count = done_count - served_count
                    print(f"  완료: {done_count}개 (서비스: {served_count}, 취소: {cancelled_count})")
                
                print("-" * 40)

            # === 에피소드 종료 조건 ===
            if env.is_done():
                last_transition = agent.replay_buffer.get_last()
                if last_transition is not None:
                    last_transition[4] = True

                for _ in range(final_train_steps):
                    train_result = agent.train()
                    if train_result is not None:
                        actor_loss, critic_loss = train_result
                        total_loss += actor_loss + critic_loss

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
                mean_waiting_time = total_waiting_time / served_count if served_count else 0
                mean_in_vehicle_time = total_in_vehicle_time / served_count if served_count else 0
                mean_detour_time = total_detour_time / served_count if served_count else 0

                print('====== Ep: {} / Reward: {} / Loss: {} ======'.format(ep, total_reward, total_loss))
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
                if write_result is True:
                    log_episode(run_path, e_info)
                e_info_list.append(e_info)

                if total_reward > best_reward:
                    best_reward = total_reward
                    if write_result is True:
                        model_name = "{}.h5".format(get_run_folder_name(config))
                        model_path = os.path.join(RESULT_PATH, model_name)
                        agent.save_model(model_path)
                break
            
            # === 추가 안전장치: 무한 루프 방지 ===
            if env.curr_time > 50:
                print(f"[Warning] Episode {ep} 강제 종료 - Time limit reached (curr_time: {env.curr_time})")
                print(f"  Active requests: {len(env.active_request_list)}")
                print(f"  Future requests: {len(env.future_request_list)}")
                print(f"  Vehicle status: {[str(v.status) for v in env.vehicle_list]}")
                print(f"  Done requests: {len(env.done_request_list)}")
                # 강제로 남은 요청들을 취소 처리
                for r in env.active_request_list + env.future_request_list:
                    r.status = RequestStatus.CANCELLED
                    env.done_request_list.append(r)
                env.active_request_list.clear()
                env.future_request_list.clear()
                break

            env.sync_state()
            state = env.state

        end_time = time.time()
        print(f"실행 시간: {end_time - start_time:.6f}초")
        
        # 에피소드 종료 시 noise 강도 감소
        agent.decay_noise()

    if write_result is True:
        log_all_episodes(run_path, e_info_list)

def main():
    env_builder = EnvBuilder(data_dir=DATA_PATH, result_dir=RESULT_PATH)

    # test_ddqn(env_builder, 128, "hd128_bs16_lr1e-06")

    for params in cfg.config_list:
        train_ddpg(env_builder, params, write_result=False)
        # train_ddqn(env_builder, params, write_result=True)



if __name__ == "__main__":
    main()
