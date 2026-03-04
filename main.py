import os
import sys

# GPU 설정
import tensorflow as tf
import csv
import app.config as cfg
from pprint import pprint
from app.env_builder import EnvBuilder
from app.ddqn_agent import DDQNAgent
from app.dqn_logger import log_dqn_metrics
from app.ppo_agent import PPOAgent
from app.ppo_logger import log_ppo_metrics
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
        writer.writerow(['Vehicle ID', 'Num. Accept', 'Num. Serve', 'On-Service Driving Time', 'Idle Time', 'Occupancy'])
        for v in drt_info_list:
            curr_row = [
                v['id'],
                v['num_accept'],
                v['num_serve'],
                v['on_service_driving_time'],
                v['idle_time'],
                v.get('occupancy', 0.0)
            ]
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
                         'Mean Waiting Time', 'Mean In-Vehicle Time', 'Mean Detour Time', 'Mean Occupancy'])
        for e in info_list:
            curr_row = [
                e['episode'],
                f"{e['total_reward']:.2f}",
                f"{e['total_loss']:.2f}",
                e['total_num_accept'],
                e['total_num_serve'],
                f"{e['mean_waiting_time']:.2f}",
                f"{e['mean_in_vehicle_time']:.2f}",
                f"{e['mean_detour_time']:.2f}",
                f"{e.get('mean_occupancy', 0.0):.2f}"
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
    # 학습률은 소수점 4자리까지 표시하되, 변환이 실패하면 원본을 사용
    if isinstance(lr, (int, float)):
        lr_str = f"{lr:.4f}"
    else:
        try:
            lr_str = f"{float(lr):.4f}"
        except Exception:
            lr_str = lr
    return f"hd{hd}_bs{bs}_lr{lr_str}"


def train_ddqn(env_builder, config, write_result=False, load_model=False):
    episodes = 500
    update_freq = 10
    final_train_steps = 10

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< Training Session: {config_str} >>>>")

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config["dqn_learning_rate"]
    dqn_config = config.copy()
    dqn_config["learning_rate"] = config["dqn_learning_rate"]
    dqn_config["batch_size"] = bs

    # Create Result Directory
    if write_result is True:
        run_name = get_run_folder_name(dqn_config)
        run_path = os.path.join(RESULT_PATH, "dqn_" + run_name)
        os.makedirs(run_path, exist_ok=True)
        # 이전 실행 로그가 남지 않도록 DQN 로그 파일 초기화
        dqn_log_path = os.path.join(run_path, "dqn_train_log.csv")
        if os.path.exists(dqn_log_path):
            try:
                os.remove(dqn_log_path)
            except PermissionError:
                print(f"[Warn] 기존 DQN 로그 파일을 삭제하지 못했습니다: {dqn_log_path}")
        # 에피소드별 보상 로그 초기화 파일 경로
        rewards_log_path = os.path.join(run_path, "episodes_reward.csv")
        if os.path.exists(rewards_log_path):
            try:
                os.remove(rewards_log_path)
            except PermissionError:
                print(f"[Warn] 기존 보상 로그 파일을 삭제하지 못했습니다: {rewards_log_path}")

    env = env_builder.build()
    agent = DDQNAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)
    
    # 기존 모델 로드 시도
    # DQN 전용 config 생성 (dqn_learning_rate를 learning_rate로 변환)
    if load_model is True:
        model_name = "{}.h5".format(get_run_folder_name(dqn_config))
        model_path = os.path.join(RESULT_PATH, model_name)
        agent.load_model(model_path)

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')
    total_time = 0.0
    reward_log_list = []
    # 에피소드별 추가 로깅용 변수
    gamma = 0.99

    for ep in range(1, episodes + 1):
        # print('\n============ Ep : {} ============'.format(ep))
        total_loss = 0.0
        total_reward = 0.0
        q_sum = 0.0
        q_count = 0
        episode_discounted_return = 0.0
        step_in_ep = 0
        state = env.reset()
        state = sanitize_state(state)

        agent.pending_buffer.clear()

        # DDQN 학습률 업데이트
        # agent.update_learning_rate(ep)

        delayed_reward_confirm = 0

        # veh_event_list는 더 이상 사용하지 않음 (환경에서 처리)

        start_time = time.time()

        while True:
            # print('\n============ Time : {} ============'.format(env.curr_time))
            # env.print_vehicles()
            # env.print_active_requests()

            while env.has_idle_vehicle():
                # 활성 요청이 없으면 액션을 선택하지 않고 시간 진행으로 넘김
                if len(env.active_request_list) == 0:
                    break

                # print('\n------------ Step : {} (Time : {}) ------------'.format(env.curr_step, env.curr_time))

                action_mask = env.get_action_mask()
                # 유효 액션이 전혀 없으면 무한 REJECT를 방지하기 위해 루프 탈출
                if np.all(action_mask == 0):
                    print(f"[Debug] Step {env.curr_step}: 액션 마스크 전부 0")
                    break
                # LLM은 탐험 구간에서만, 설정한 스텝 간격에 따라 사용
                # use_llm_now = (
                #     USE_LLM_ASSIST
                #     and (env.curr_step % LLM_STEP_INTERVAL == 0)
                #     and (np.random.rand() <= agent.epsilon)
                # )
                # if use_llm_now:
                #     action = agent.act_with_llm(
                #         env=env,
                #         state=state,c
                #         action_mask=action_mask,
                #         priority="대기시간 최소화 > 시간창 준수 > 승차율",
                #         constraints=LLM_CONSTRAINTS,
                #         task="recommend",
                #     )
                # else:
                #    action = agent.act(state, action_mask)
                action = agent.act(state, action_mask)
                env.enrich_action(action)

                next_state, reward, info = env.step(action)
                next_state = sanitize_state(next_state)
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
                # Q-value 평균 계산용: 현재 상태에서 선택된 액션의 Q를 accumulate
                try:
                    q_values = agent.get_action_q_values(state, action_mask)
                    q_val = float(q_values[action[0], action[1]])
                    q_sum += q_val
                    q_count += 1
                except Exception:
                    pass
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
                # 할인 보상 누적
                episode_discounted_return += (gamma ** step_in_ep) * reward
                step_in_ep += 1
                state = next_state

            env.curr_time += 1
            d_reward_list, cancelled_request_ids = env.handle_time_update()
            
            # 실제 픽업/드롭오프 발생 시 이벤트 기록 (제거 - 환경에서 처리)

            for pair in d_reward_list:
                action_id, reward = pair
                agent.confirm_and_remember(action_id, reward)
                delayed_reward_confirm += 1
                if action_id == "STEP_BONUS":
                    continue
                else:
                    print(f"[Debug] Step {env.curr_step}: 시간 업데이트로 인한 지연 보상 - {action_id} (보상: {reward})")

            # 시간 경과로 발생한 보상을 total_reward에 합산
            if len(d_reward_list) > 0:
                time_reward = sum(reward for _, reward in d_reward_list)
                total_reward += time_reward

            # 취소된 request 관련 pending action 즉시 제거 (PICKUP, DROPOFF 모두)
            for rid in cancelled_request_ids:
                agent.pending_buffer.cancel(f"{rid}_1")  # PICKUP
                agent.pending_buffer.cancel(f"{rid}_2")  # DROPOFF
                # 해당 요청을 수락한 차량 상태도 정리
                for v in env.vehicle_list:
                    # 타겟이 취소된 요청이면 즉시 해제
                    if getattr(v, "target_request", None) is not None and str(v.target_request.id) == str(rid):
                        v.status = VehicleStatus.IDLE
                        v.target_request = None
                        v.target_arrival_time = -1
                        v.next_node = 0
                    # active_request_list에서 제거
                    v.active_request_list = [r for r in v.active_request_list if str(r.id) != str(rid)]

            # PendingBuffer에 남아 있지만 더 이상 active/done에 없는 요청 정리 (누적 방지)
            if len(agent.pending_buffer) > 0:
                valid_rids = {str(r.id) for r in env.active_request_list + env.done_request_list}
                to_cancel = []
                for action_id in list(agent.pending_buffer.pending.keys()):
                    req_id = action_id.split("_")[0]
                    if req_id not in valid_rids:
                        to_cancel.append(action_id)
                for action_id in to_cancel:
                    agent.pending_buffer.cancel(action_id)
                    # 차량 상태도 함께 정리
                    for v in env.vehicle_list:
                        if getattr(v, "target_request", None) is not None and str(v.target_request.id) == req_id:
                            v.status = VehicleStatus.IDLE
                            v.target_request = None
                            v.target_arrival_time = -1
                            v.next_node = 0
                        v.active_request_list = [r for r in v.active_request_list if str(r.id) != req_id]

            # # 모든 요청이 소진되었는데 pending이 남았다면 강제로 정리해 에피소드 종료 유도
            # if len(env.active_request_list) == 0 and len(env.future_request_list) == 0 and len(agent.pending_buffer) > 0:
            #     print("[세이프가드] 남은 요청 없음 + Pending 잔여 -> Pending 강제 정리")
            #     agent.pending_buffer.clear()
            #     for v in env.vehicle_list:
            #         v.status = VehicleStatus.IDLE
            #         v.target_request = None
            #         v.target_arrival_time = -1
            #         v.next_node = 0

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
                total_occupancy = 0.0
                for v in env.vehicle_list:
                    total_num_accept += v.num_accept
                    total_num_serve += v.num_serve
                    v.on_service_driving_time = env.curr_time - v.idle_time
                    occ = 0.0
                    if hasattr(v, "occupancy_sum") and hasattr(v, "occupancy_cnt") and v.occupancy_cnt > 0:
                        occ = v.occupancy_sum / v.occupancy_cnt
                    total_occupancy += occ
                    v_info = {
                        'id': v.id,
                        'num_accept': v.num_accept,
                        'num_serve': v.num_serve,
                        'idle_time': v.idle_time,
                        'on_service_driving_time': v.on_service_driving_time,
                        'occupancy': occ,
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
                mean_occupancy = total_occupancy / len(env.vehicle_list) if len(env.vehicle_list) > 0 else 0.0

                # 성능 추적 정보 추가
                recent_avg = sum(agent.recent_rewards[-5:]) / 5 if len(agent.recent_rewards) >= 5 else total_reward
                print('====== Ep: {} / Reward: {} / Loss: {} / eps: {:.4f} / LR: {:.6f} / Avg5: {:.2f} ======'.format(
                    ep, total_reward, total_loss, agent.epsilon, agent.current_learning_rate, recent_avg))

                # 파일 저장은 전체 정보 사용
                e_info_full = {
                    'episode': ep,
                    'total_reward': total_reward,
                    'total_loss': total_loss,
                    'total_num_accept': total_num_accept,
                    'total_num_serve': total_num_serve,
                    'mean_waiting_time': mean_waiting_time,
                    'mean_in_vehicle_time': mean_in_vehicle_time,
                    'mean_detour_time': mean_detour_time,
                    'mean_occupancy': mean_occupancy,
                    'event_sequence': env.event_sequences if hasattr(env, 'event_sequences') else [],
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                if write_result is True:
                    log_episode(run_path, e_info_full)
                # 에피소드별 보상 기록 누적
                rr = env.reward_record
                reward_log_list.append({
                    "episode": ep,
                    "decision": rr.DecisionReward,
                    "immediate": rr.ImmediateReward,
                    "delayed": rr.DelayedReward,
                    "cancel_penalty": rr.CancelPenalty,
                    "maintenance": rr.MaintenanceReward,
                    "total": rr.total(),
                })

                # 메모리 절약: 요약만 리스트에 저장
                e_info_summary = {
                    'episode': ep,
                    'total_reward': total_reward,
                    'total_loss': total_loss,
                    'total_num_accept': total_num_accept,
                    'total_num_serve': total_num_serve,
                    'mean_waiting_time': mean_waiting_time,
                    'mean_in_vehicle_time': mean_in_vehicle_time,
                    'mean_detour_time': mean_detour_time,
                    'mean_occupancy': mean_occupancy
                }
                e_info_list.append(e_info_summary)

                # DQN 학습 로그 기록
                if write_result is True:
                    avg_q = (q_sum / q_count) if q_count > 0 else 0.0
                    metrics = {
                        "avg_q": avg_q,
                        "episode_length": step_in_ep,
                        "discounted_return": episode_discounted_return,
                    }
                    log_dqn_metrics(run_path, ep, metrics)

                # Save Model
                if total_reward > best_reward:
                    best_reward = total_reward
                    if write_result is True:
                        model_name = "{}.h5".format(get_run_folder_name(dqn_config))
                        model_path = os.path.join(RESULT_PATH, model_name)
                        agent.save_model(model_path)

                break

            env.sync_state()
            state = env.state

            # 스텝 하드캡(세이프가드): 무한 루프 방지
            if cfg.ENABLE_SAFEGUARD and env.curr_step >= cfg.MAX_STEPS_CAP:
                print(f"[세이프가드] 스텝 하드캡 도달({cfg.MAX_STEPS_CAP}) → 에피소드 강제 종료")
                env.active_request_list = []
                env.future_request_list = []

        end_time = time.time()
        progress_time = end_time - start_time
        total_time += progress_time
        print(f"실행 시간: {progress_time:.6f}초")


        agent.decay_epsilon()

        # # 세션/캐시 정리 (주기적 OOM 완화)
        # if ep % 20 == 0:
        #     try:
        #         tf.keras.backend.clear_session()
        #         import gc
        #         gc.collect()
        #         print("[메모리 정리] Keras 세션 정리 및 GC 수행")
        #     except Exception as e:
        #         print(f"[Info] 세션 정리 스킵: {e}")

        # # GPU 메모리 정리 (10 에피소드마다 - OOM 방지 강화)
        # if ep % 10 == 0:
        #     import gc
        #     gc.collect()
        #     # Replay Buffer 크기 제한 강제
        #     if len(agent.replay_buffer) > agent.replay_buffer.capacity:
        #         print(f"[경고] Replay Buffer 초과: {len(agent.replay_buffer)} > {agent.replay_buffer.capacity}")
        #     print(f"[메모리 정리] Episode {ep} 완료 (Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.capacity})")

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
        # 에피소드별 보상 CSV 저장
        rewards_log_path = os.path.join(run_path, "episodes_reward.csv")
        with open(rewards_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode",
                "DecisionReward",
                "ImmediateReward",
                "DelayedReward",
                "CancelPenalty",
                "MaintenanceReward",
                "TotalReward"
            ])
            for rec in reward_log_list:
                writer.writerow([
                    rec["episode"],
                    f"{rec['decision']:.4f}",
                    f"{rec['immediate']:.4f}",
                    f"{rec['delayed']:.4f}",
                    f"{rec['cancel_penalty']:.4f}",
                    f"{rec['maintenance']:.4f}",
                    f"{rec['total']:.4f}",
                ])

def sanitize_state(state):
    """
    환경에서 받은 state에 불필요한 배치 차원이 있다면 제거
    """
    if isinstance(state, list):
        clean_list = []
        for s in state:
            s_array = np.array(s, dtype=np.float32)
            if s_array.ndim > 1 and s_array.shape[0] == 1:
                s_array = np.squeeze(s_array, axis=0)
            clean_list.append(s_array)
        return clean_list
    else:
        s_array = np.array(state, dtype=np.float32)
        if s_array.ndim > 1 and s_array.shape[0] == 1:
            s_array = np.squeeze(s_array, axis=0)
        return s_array


def train_ppo(env_builder, config, write_result=False, load_model=False):
    episodes = 500

    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< PPO Training Session: {config_str} >>>>")

    # Create Result Directory
    if write_result is True:
        ppo_folder_config = config.copy()
        ppo_folder_config["learning_rate"] = config["critic_learning_rate"] + config["actor_learning_rate"]
        ppo_folder_config["batch_size"] = config["ppo_batch_size"]
        run_name = get_run_folder_name(ppo_folder_config)
        run_path = os.path.join(RESULT_PATH, "ppo_" + run_name)
        os.makedirs(run_path, exist_ok=True)
        # 이전 실행의 학습 로그가 남아 중복 기록되지 않도록 초기화
        ppo_log_path = os.path.join(run_path, "ppo_train_log.csv")
        if os.path.exists(ppo_log_path):
            try:
                os.remove(ppo_log_path)
            except PermissionError:
                # 열려 있으면 건너뛰되 이후 append 시도는 계속됨
                print(f"[Warn] 기존 PPO 로그 파일을 삭제하지 못했습니다: {ppo_log_path}")

    hd = config["hidden_dim"]
    bs = config["ppo_batch_size"]
    actor_lr = config["actor_learning_rate"]
    critic_lr = config["critic_learning_rate"]

    env = env_builder.build()
    agent = PPOAgent(hidden_dim=hd, batch_size=bs, actor_learning_rate=actor_lr, critic_learning_rate=critic_lr)
    
    # 기존 모델 로드 시도
    if load_model is True:
        ppo_config = config.copy()
        ppo_config["learning_rate"] = config["critic_learning_rate"] + config["actor_learning_rate"]
        model_name = "{}.h5".format(get_run_folder_name(ppo_config))
        model_path = os.path.join(RESULT_PATH, model_name)
        agent.load_model(model_path)

    transition_id = 0
    e_info_list = []
    best_reward = float('-inf')
    total_time = 0.0
    
    for ep in range(1, episodes + 1):
        train_calls_in_ep = 0  # 에피소드마다 초기화
        total_loss = 0.0
        total_reward = 0.0
        state = env.reset()
        state = sanitize_state(state)

        delayed_reward_confirm = 0

        # veh_event_list는 더 이상 사용하지 않음 (환경에서 처리)

        start_time = time.time()

        # 정체 세이프가드용 상태 (config 기반)
        stagnation_window = cfg.STAGNATION_WINDOW
        recent_rewards_window = []
        recent_pending_counts = []
        max_steps_cap = cfg.MAX_STEPS_CAP

        while True:
            

            while env.has_idle_vehicle():
                # 활성 요청이 없으면 액션을 선택하지 않고 시간 진행으로 넘김
                if len(env.active_request_list) == 0:
                    break

                action_mask = env.get_action_mask()
                if np.all(action_mask == 0):
                    print(f"[Debug] Step {env.curr_step}: 액션 마스크 전부 0")
                    break
                action = agent.act(state, action_mask)
                env.enrich_action(action)

                next_state, reward, info = env.step(action)
                next_state = sanitize_state(next_state)
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
                    print(f"[Debug] Train_update: {agent.last_train_stats['train_updates']}, buffer_size: {agent.last_train_stats['buffer_size']}")
                    if curr_loss is not None:
                        print(f"[Debug] Step {env.curr_step}: 학습 수행")
                        train_calls_in_ep += 1
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

            # 취소된 request 관련 pending action 즉시 제거 (PICKUP, DROPOFF 모두)
            for rid in cancelled_request_ids:
                agent.pending_buffer.cancel(f"{rid}_1")  # PICKUP
                agent.pending_buffer.cancel(f"{rid}_2")  # DROPOFF

            # 정체 세이프가드: 보상·대기크기 변화 모니터링 (토글)
            if cfg.ENABLE_SAFEGUARD:
                recent_rewards_window.append(total_reward)
                recent_pending_counts.append(len(agent.pending_buffer))
                if len(recent_rewards_window) > stagnation_window:
                    recent_rewards_window.pop(0)
                    recent_pending_counts.pop(0)
                    if (max(recent_rewards_window) - min(recent_rewards_window) == 0) and \
                       (max(recent_pending_counts) - min(recent_pending_counts) == 0):
                        print("[세이프가드] 보상/대기변화 없음 → 에피소드 종료")
                        # 남은 pending은 로그 후 정리
                        if len(agent.pending_buffer) != 0:
                            print("[세이프가드] 남은 Pending 강제 정리")
                            agent.pending_buffer.clear()
                        # 종료 처리 경로로 이동
                        env.active_request_list = []
                        env.future_request_list = []

            # 스텝 하드캡 (토글)
            if cfg.ENABLE_SAFEGUARD and env.curr_step >= max_steps_cap:
                print(f"[세이프가드] 스텝 하드캡 도달({max_steps_cap}) → 에피소드 종료")
                env.active_request_list = []
                env.future_request_list = []

            if env.is_done():
                # 에피소드 완료 시 처리
                if len(agent.episode_buffer) > 0:
                    agent.episode_buffer[-1][4] = True  # 마지막 transition의 done을 True로 설정
                
                # 에피소드 완료 후 trajectory buffer에 추가
                agent.finish_episode()

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
                total_occupancy = 0.0
                for v in env.vehicle_list:
                    total_num_accept += v.num_accept
                    total_num_serve += v.num_serve
                    v.on_service_driving_time = env.curr_time - v.idle_time
                    occ = 0.0
                    if hasattr(v, "occupancy_sum") and hasattr(v, "occupancy_cnt") and v.occupancy_cnt > 0:
                        occ = v.occupancy_sum / v.occupancy_cnt
                    total_occupancy += occ
                    v_info = {
                        'id': v.id,
                        'num_accept': v.num_accept,
                        'num_serve': v.num_serve,
                        'idle_time': v.idle_time,
                        'on_service_driving_time': v.on_service_driving_time,
                        'occupancy': occ
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
                mean_occupancy = total_occupancy / len(env.vehicle_list) if len(env.vehicle_list) > 0 else 0.0

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
                    'mean_occupancy': mean_occupancy,
                    'event_sequence': env.event_sequences if hasattr(env, 'event_sequences') else [],
                    'drt_info': drt_info_list,
                    'request_info': req_info_list
                }
                if write_result is True:
                    log_episode(run_path, e_info)
                    # PPO 학습 메트릭 로그 (에피소드 단위)
                    metrics = agent.last_train_stats.copy() if hasattr(agent, "last_train_stats") else {}
                    metrics["train_calls"] = train_calls_in_ep
                    log_ppo_metrics(run_path, ep, metrics)
                e_info_list.append(e_info)

                # Save Model
                if total_reward > best_reward:
                    best_reward = total_reward
                    if write_result is True:
                        # PPO 전용 config 사용
                        ppo_save_config = config.copy()
                        ppo_save_config["learning_rate"] = config["critic_learning_rate"] + config["actor_learning_rate"]
                        ppo_save_config["batch_size"] = config["ppo_batch_size"]
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
    


def train_mappo(env_builder, config, write_result=False, load_model=False):
    episodes = 5000
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
                    'event_sequence': env.event_sequences if hasattr(env, 'event_sequences') else [],
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
        # train_ppo(env_builder, params, write_result=True, load_model=False)
        # train_mappo(env_builder, params, write_result=True, load_model=False)


if __name__ == "__main__":
    main()
