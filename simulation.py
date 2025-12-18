import os
import sys

conda_env_path = r'C:\Users\khlk0\anaconda3\envs\KW_DRT'
cuda_bin_path = os.path.join(conda_env_path, 'Library', 'bin')
if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ.get('PATH', '')
    print(f"CUDA DLL 경로 추가: {cuda_bin_path}")
else:
    print(f"경고: CUDA 경로를 찾을 수 없습니다: {cuda_bin_path}")


import tensorflow as tf
import csv
import app.config as cfg
from pprint import pprint
from app.env_builder import EnvBuilder
from app.ddqn_agent import DDQNAgent
from app.ppo_agent import PPOAgent
from app.mappo_agent import MAPPOAgent
from app.request_status import RequestStatus
from app.action_type import ActionType
from app.vehicle_status import VehicleStatus
import time
import numpy as np
import random

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # libdevice 경고 숨기기
tf.config.optimizer.set_jit(False)

print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)

try:
    if len(tf.config.list_physical_devices('GPU')) > 0:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("[OK] Mixed precision 활성화 (mixed_float16)")
except Exception as e:
    print(f"[Info] Mixed precision 설정 스킵: {e}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]
            )
        print(f"[OK] GPU {len(gpus)}개 사용 가능 - 메모리 제한: 3GB")
        print(f"[OK] XLA JIT 컴파일 비활성화 완료")
    except RuntimeError as e:
        print(f"GPU 메모리 제한 설정 실패 (이미 초기화됨): {e}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] GPU {len(gpus)}개 사용 - 메모리 증가 모드 활성화")
        except Exception as e2:
            print(f"GPU 설정 오류: {e2}")
else:
    print("WARNING: GPU를 찾을 수 없습니다. CPU로 실행됩니다.")

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
    
    # 차량 초기 위치 로깅
    if 'initial_vehicle_positions' in info:
        initial_positions = info['initial_vehicle_positions']
        filename = f'episode_{ep:03}_initial_positions.txt'
        filepath = os.path.join(path, filename)
        with open(filepath, "w") as f:
            f.write(f"Episode {ep} - Initial Vehicle Positions\n")
            f.write("=" * 50 + "\n")
            for i, pos in enumerate(initial_positions):
                f.write(f"Vehicle {i}: Node {pos}\n")
            f.write("\n")
            f.write(f"Positions: {initial_positions}\n")


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

def simulation_ddqn(env_builder, config, model_path=None, episodes=20, write_result=True):
    """
    DDQN 모델 시뮬레이션 함수 (학습 없이 가중치만 로드하여 시뮬레이션)
    
    Args:
        env_builder: 환경 빌더
        config: 설정 딕셔너리
        model_path: 모델 파일 경로 (None이면 자동으로 찾음)
        episodes: 시뮬레이션 에피소드 수
        write_result: 결과 저장 여부
    """

    hd = config["hidden_dim"]
    bs = 16
    #bs = config["batch_size"]
    lr = config["dqn_learning_rate"]
    dqn_config = config.copy()
    dqn_config["learning_rate"] = config["dqn_learning_rate"]
    dqn_config["batch_size"] = bs

    # 결과 디렉토리 생성
    if write_result is True:

        run_name = get_run_folder_name(dqn_config)
    run_path = os.path.join(RESULT_PATH, "simul_dqn_" + run_name)
    os.makedirs(run_path, exist_ok=True)

    env = env_builder.build()
    agent = DDQNAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)
    
    # 탐험 완전 비활성화 (시뮬레이션 모드)
    agent.epsilon = 0.0
    
    # 모델 로드
    if model_path is None:
        # 자동으로 모델 파일 찾기
        model_name = "{}.h5".format(get_run_folder_name(dqn_config))
        model_path = os.path.join(RESULT_PATH, model_name)
    
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"[시뮬레이션] DDQN 모델 로드 완료: {model_path}")
    else:
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {model_path}")
        return None

    e_info_list = []
    total_time = 0.0

    for ep in range(1, episodes + 1):
        print(f"\n[Episode {ep}/{episodes}] DDQN 시뮬레이션 실행 중...")
        
        # 각 에피소드마다 동일한 시드 사용 (다른 모델과 동일한 환경 조건 보장)
        episode_seed = ep * 100  # 에피소드별 고유 시드
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        tf.random.set_seed(episode_seed)
        
        # agent.epsilon = 0.05
        agent.epsilon = 0.0

        # 랜덤 차량 위치로 환경 리셋 (에피소드마다 다른 위치)
        state = env.reset(random_vehicle_positions=True)
        
        # 차량 초기 위치 저장 (로깅용)
        initial_vehicle_positions = env.vehicle_init_pos.copy() if hasattr(env, 'vehicle_init_pos') else []
        
        total_reward = 0.0
        step_count = 0
        action_count = 0

        delayed_reward_confirm = 0
        start_time = time.time()

        while True:
            while env.has_idle_vehicle():
                action_mask = env.get_action_mask()
                action = agent.act(state, action_mask)
                env.enrich_action(action)

                next_state, reward, info = env.step(action)
                next_action_mask = env.get_action_mask()
                
                total_reward += reward
                action_count += 1
                state = next_state

            env.curr_time += 1
            d_reward_list, cancelled_request_ids = env.handle_time_update()
            
            # 지연 보상 처리
            for action_id, delayed_reward in d_reward_list:
                if hasattr(agent, 'confirm_and_remember'):
                    agent.confirm_and_remember(action_id, delayed_reward)
            
            # 취소된 요청 처리
            for rid in cancelled_request_ids:
                if hasattr(agent, 'pending_buffer'):
                    agent.pending_buffer.cancel(f"{rid}_1")
                    agent.pending_buffer.cancel(f"{rid}_2")
            
            step_count += 1
            
            # 에피소드 종료 조건
            if env.is_done():
                break
            
            # 안전장치: 최대 스텝 제한 (매우 높은 값으로 설정)
            if step_count > 1000:
                print(f"[Warning] Episode {ep}: 최대 스텝 제한 도달")
                break
            
            # 상태 동기화
            env.sync_state()
            state = env.state

        # 에피소드 결과 수집
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
        
        # 평균 계산
        if served_count > 0:
            mean_waiting_time = total_waiting_time / served_count
            mean_in_vehicle_time = total_in_vehicle_time / served_count
            mean_detour_time = total_detour_time / served_count
        else:
            mean_waiting_time = 0.0
            mean_in_vehicle_time = 0.0
            mean_detour_time = 0.0

        print(f"[Episode {ep}] 완료 - 보상: {total_reward:.2f}, 스텝: {step_count}, 행동: {action_count}")
        print(f"  서비스율: {served_count}/{len(env.done_request_list)} ({served_count/len(env.done_request_list)*100:.1f}%)")
        print(f"  평균 대기시간: {mean_waiting_time:.2f}, 평균 승차시간: {mean_in_vehicle_time:.2f}")
        print(f"  차량 초기 위치: {initial_vehicle_positions}")

        e_info = {
            'episode': ep,
            'total_reward': total_reward,
            'total_loss': 0.0,  # 시뮬레이션에서는 학습 없음
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

        end_time = time.time()
        progress_time = end_time - start_time
        total_time += progress_time

    # 전체 결과 요약
    if e_info_list:
        avg_reward = sum(e['total_reward'] for e in e_info_list) / len(e_info_list)
        avg_service_rate = sum(e['total_num_serve'] for e in e_info_list) / sum(e['total_num_accept'] for e in e_info_list) if sum(e['total_num_accept'] for e in e_info_list) > 0 else 0
        avg_waiting_time = sum(e['mean_waiting_time'] for e in e_info_list) / len(e_info_list)
        
        print(f"\n{'='*60}")
        print(f"DDQN 시뮬레이션 결과 요약")
        print(f"{'='*60}")
        print(f"에피소드 수: {episodes}")
        print(f"평균 보상: {avg_reward:.2f}")
        print(f"평균 서비스율: {avg_service_rate:.3f}")
        print(f"평균 대기시간: {avg_waiting_time:.2f}")
        print(f"총 실행 시간: {total_time:.2f}초")

    if write_result is True:
        log_all_episodes(run_path, e_info_list, total_time)
        print(f"[저장] 결과 저장 완료: {run_path}")

    return e_info_list


def simulation_ppo(env_builder, config, model_path=None, episodes=20, write_result=True):
    """
    PPO 모델 시뮬레이션 함수 (학습 없이 가중치만 로드하여 시뮬레이션)
    """
    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< PPO Simulation Session: {config_str} >>>>")

    # 결과 디렉토리 생성
    if write_result is True:
        ppo_folder_config = config.copy()
        ppo_folder_config["learning_rate"] = config["ppo_learning_rate"]
        run_name = get_run_folder_name(ppo_folder_config)
        run_path = os.path.join(RESULT_PATH, "simul_ppo_" + run_name)
        os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config["ppo_learning_rate"]

    env = env_builder.build()
    agent = PPOAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)
    
    # 모델 로드
    if model_path is None:
        ppo_config = config.copy()
        ppo_config["learning_rate"] = config["ppo_learning_rate"]
        model_name = "{}.h5".format(get_run_folder_name(ppo_config))
        model_path = os.path.join(RESULT_PATH, model_name)
    
    # PPO는 actor와 critic 파일이 모두 존재해야 함
    actor_path = model_path.replace('.h5', '_actor.h5')
    critic_path = model_path.replace('.h5', '_critic.h5')
    
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        agent.load_model(model_path)
        print(f"[시뮬레이션] PPO 모델 로드 완료: {actor_path}, {critic_path}")
    else:
        print(f"[오류] PPO 모델 파일을 찾을 수 없습니다:")
        print(f"  Actor: {actor_path} - {'존재' if os.path.exists(actor_path) else '없음'}")
        print(f"  Critic: {critic_path} - {'존재' if os.path.exists(critic_path) else '없음'}")
        return None

    e_info_list = []
    total_time = 0.0

    for ep in range(1, episodes + 1):
        print(f"\n[Episode {ep}/{episodes}] PPO 시뮬레이션 실행 중...")
        
        # 각 에피소드마다 동일한 시드 사용 (DDQN과 동일한 환경 조건 보장)
        episode_seed = ep * 100  # 에피소드별 고유 시드 (DDQN과 동일)
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        tf.random.set_seed(episode_seed)
        
        # 랜덤 차량 위치로 환경 리셋 (에피소드마다 다른 위치, DDQN과 동일한 시드 사용)
        state = env.reset(random_vehicle_positions=True)
        
        # 차량 초기 위치 저장 (로깅용)
        initial_vehicle_positions = env.vehicle_init_pos.copy() if hasattr(env, 'vehicle_init_pos') else []
        
        total_reward = 0.0
        step_count = 0
        action_count = 0

        delayed_reward_confirm = 0
        start_time = time.time()

        while True:
            while env.has_idle_vehicle():
                action_mask = env.get_action_mask()
                action = agent.act(state, action_mask)
                env.enrich_action(action)

                next_state, reward, info = env.step(action)
                next_action_mask = env.get_action_mask()

                total_reward += reward
                action_count += 1
                state = next_state

            env.curr_time += 1
            d_reward_list, cancelled_request_ids = env.handle_time_update()
            
            # 지연 보상 처리
            for action_id, delayed_reward in d_reward_list:
                if hasattr(agent, 'confirm_and_remember'):
                    agent.confirm_and_remember(action_id, delayed_reward)
            
            # 취소된 요청 처리
            for rid in cancelled_request_ids:
                if hasattr(agent, 'pending_buffer'):
                    agent.pending_buffer.cancel(f"{rid}_1")
                    agent.pending_buffer.cancel(f"{rid}_2")
            
            step_count += 1
            
            # 에피소드 종료 조건
            if env.is_done():
                break
            
            # 안전장치: 최대 스텝 제한
            if step_count > 1000:
                print(f"[Warning] Episode {ep}: 최대 스텝 제한 도달")
                break
            
            # 상태 동기화
            env.sync_state()
            state = env.state

        # 에피소드 결과 수집 (DDQN과 동일한 로직)
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
        
        # 평균 계산
        if served_count > 0:
            mean_waiting_time = total_waiting_time / served_count
            mean_in_vehicle_time = total_in_vehicle_time / served_count
            mean_detour_time = total_detour_time / served_count
        else:
            mean_waiting_time = 0.0
            mean_in_vehicle_time = 0.0
            mean_detour_time = 0.0

        print(f"[Episode {ep}] 완료 - 보상: {total_reward:.2f}, 스텝: {step_count}, 행동: {action_count}")
        print(f"  서비스율: {served_count}/{len(env.done_request_list)} ({served_count/len(env.done_request_list)*100:.1f}%)")
        print(f"  평균 대기시간: {mean_waiting_time:.2f}, 평균 승차시간: {mean_in_vehicle_time:.2f}")
        print(f"  차량 초기 위치: {initial_vehicle_positions}")

        e_info = {
            'episode': ep,
            'total_reward': total_reward,
            'total_loss': 0.0,  # 시뮬레이션에서는 학습 없음
            'total_num_accept': total_num_accept,
            'total_num_serve': total_num_serve,
            'mean_waiting_time': mean_waiting_time,
            'mean_in_vehicle_time': mean_in_vehicle_time,
            'mean_detour_time': mean_detour_time,
            'event_sequence': env.event_sequences if hasattr(env, 'event_sequences') else [],
            'drt_info': drt_info_list,
            'request_info': req_info_list,
            'initial_vehicle_positions': initial_vehicle_positions
        }
        
        if write_result is True:
            log_episode(run_path, e_info)
        e_info_list.append(e_info)

        end_time = time.time()
        progress_time = end_time - start_time
        total_time += progress_time

    # 전체 결과 요약
    if e_info_list:
        avg_reward = sum(e['total_reward'] for e in e_info_list) / len(e_info_list)
        avg_service_rate = sum(e['total_num_serve'] for e in e_info_list) / sum(e['total_num_accept'] for e in e_info_list) if sum(e['total_num_accept'] for e in e_info_list) > 0 else 0
        avg_waiting_time = sum(e['mean_waiting_time'] for e in e_info_list) / len(e_info_list)
        
        print(f"\n{'='*60}")
        print(f"PPO 시뮬레이션 결과 요약")
        print(f"{'='*60}")
        print(f"에피소드 수: {episodes}")
        print(f"평균 보상: {avg_reward:.2f}")
        print(f"평균 서비스율: {avg_service_rate:.3f}")
        print(f"평균 대기시간: {avg_waiting_time:.2f}")
        print(f"총 실행 시간: {total_time:.2f}초")

    if write_result is True:
        log_all_episodes(run_path, e_info_list, total_time)
        print(f"[저장] 결과 저장 완료: {run_path}")

    return e_info_list


def simulation_mappo(env_builder, config, model_path=None, episodes=20, write_result=True):
    """
    MAPPO 모델 시뮬레이션 함수 (학습 없이 가중치만 로드하여 시뮬레이션)
    """
    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"\n<<<< MAPPO Simulation Session: {config_str} >>>>")

    # 결과 디렉토리 생성
    if write_result is True:
        mappo_folder_config = config.copy()
        mappo_folder_config["learning_rate"] = config.get("mappo_learning_rate", config["learning_rate"])
        run_name = get_run_folder_name(mappo_folder_config)
        run_path = os.path.join(RESULT_PATH, "simul_mappo_" + run_name)
        os.makedirs(run_path, exist_ok=True)

    hd = config["hidden_dim"]
    bs = config["batch_size"]
    lr = config.get("mappo_learning_rate", config["learning_rate"])

    env = env_builder.build()
    agent = MAPPOAgent(hidden_dim=hd, batch_size=bs, learning_rate=lr)
    
    # 모델 로드
    if model_path is None:
        mappo_config = config.copy()
        mappo_config["learning_rate"] = config.get("mappo_learning_rate", config["learning_rate"])
        model_name = "{}.h5".format(get_run_folder_name(mappo_config))
        model_path = os.path.join(RESULT_PATH, model_name)
    
    # MAPPO는 각 에이전트의 actor와 centralized critic 파일이 모두 존재해야 함
    critic_path = model_path.replace('.h5', '_critic.h5')
    actor_files_exist = True
    
    # 각 에이전트의 actor 파일 확인
    for i in range(agent.num_agents):
        actor_path = model_path.replace('.h5', f'_actor_{i}.h5')
        if not os.path.exists(actor_path):
            actor_files_exist = False
            break
    
    if actor_files_exist and os.path.exists(critic_path):
        agent.load_model(model_path)
        print(f"[시뮬레이션] MAPPO 모델 로드 완료: {agent.num_agents}개 actor + 1개 critic")
    else:
        print(f"[오류] MAPPO 모델 파일을 찾을 수 없습니다:")
        print(f"  Critic: {critic_path} - {'존재' if os.path.exists(critic_path) else '없음'}")
        print(f"  Actor 파일들: {'모두 존재' if actor_files_exist else '일부 누락'}")
        # 누락된 actor 파일들 표시
        if not actor_files_exist:
            for i in range(agent.num_agents):
                actor_path = model_path.replace('.h5', f'_actor_{i}.h5')
                if not os.path.exists(actor_path):
                    print(f"    Actor {i}: {actor_path} - 없음")
        return None

    e_info_list = []
    total_time = 0.0

    for ep in range(1, episodes + 1):
        print(f"\n[Episode {ep}/{episodes}] MAPPO 시뮬레이션 실행 중...")
        
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        action_count = 0

        delayed_reward_confirm = 0
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
                    
                else:
                    # 기존 방식 (단일 액션)
                    env.enrich_action(action)
                    next_state, reward, info = env.step(action)
                    next_action_mask = env.get_action_mask()
                
                total_reward += reward
                action_count += 1
                state = next_state

            env.curr_time += 1
            d_reward_list, cancelled_request_ids = env.handle_time_update()
            
            # 지연 보상 처리
            for action_id, delayed_reward in d_reward_list:
                if hasattr(agent, 'confirm_and_remember'):
                    agent.confirm_and_remember(action_id, delayed_reward)
            
            # 취소된 요청 처리
            for rid in cancelled_request_ids:
                if hasattr(agent, 'pending_buffer'):
                    agent.pending_buffer.cancel(f"{rid}_1")
            
            step_count += 1
            
            # 에피소드 종료 조건
            if env.is_done():
                break
            
            # 안전장치: 최대 스텝 제한
            if step_count > 1000:
                print(f"[Warning] Episode {ep}: 최대 스텝 제한 도달")
                break
            
            # 상태 동기화
            env.sync_state()
            state = env.state

        # 에피소드 결과 수집 (DDQN과 동일한 로직)
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
        
        # 평균 계산
        if served_count > 0:
            mean_waiting_time = total_waiting_time / served_count
            mean_in_vehicle_time = total_in_vehicle_time / served_count
            mean_detour_time = total_detour_time / served_count
        else:
            mean_waiting_time = 0.0
            mean_in_vehicle_time = 0.0
            mean_detour_time = 0.0

        print(f"[Episode {ep}] 완료 - 보상: {total_reward:.2f}, 스텝: {step_count}, 행동: {action_count}")
        print(f"  서비스율: {served_count}/{len(env.done_request_list)} ({served_count/len(env.done_request_list)*100:.1f}%)")
        print(f"  평균 대기시간: {mean_waiting_time:.2f}, 평균 승차시간: {mean_in_vehicle_time:.2f}")

        e_info = {
            'episode': ep,
            'total_reward': total_reward,
            'total_loss': 0.0,  # 시뮬레이션에서는 학습 없음
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

        end_time = time.time()
        progress_time = end_time - start_time
        total_time += progress_time

    # 전체 결과 요약
    if e_info_list:
        avg_reward = sum(e['total_reward'] for e in e_info_list) / len(e_info_list)
        avg_service_rate = sum(e['total_num_serve'] for e in e_info_list) / sum(e['total_num_accept'] for e in e_info_list) if sum(e['total_num_accept'] for e in e_info_list) > 0 else 0
        avg_waiting_time = sum(e['mean_waiting_time'] for e in e_info_list) / len(e_info_list)
        
        print(f"\n{'='*60}")
        print(f"MAPPO 시뮬레이션 결과 요약")
        print(f"{'='*60}")
        print(f"에피소드 수: {episodes}")
        print(f"평균 보상: {avg_reward:.2f}")
        print(f"평균 서비스율: {avg_service_rate:.3f}")
        print(f"평균 대기시간: {avg_waiting_time:.2f}")
        print(f"총 실행 시간: {total_time:.2f}초")

    if write_result is True:
        log_all_episodes(run_path, e_info_list, total_time)
        print(f"[저장] 결과 저장 완료: {run_path}")

    return e_info_list


def main():
    env_builder = EnvBuilder(data_dir=DATA_PATH, result_dir=RESULT_PATH)

    print("="*60)
    print("DRT 모델 시뮬레이션 실행")
    print("="*60)
    
    # 사용 가능한 설정들
    available_configs = cfg.config_list
    
    # 첫 번째 설정으로 DDQN 시뮬레이션 실행
    if available_configs:
        config = available_configs[0]        
        # DDQN 시뮬레이션
        ddqn_results = simulation_ddqn(env_builder, config, episodes=30, write_result=True)
        
        # PPO 시뮬레이션
        ppo_results = simulation_ppo(env_builder, config, episodes=30, write_result=True)
        
        # MAPPO 시뮬레이션
        # mappo_results = simulation_mappo(env_builder, config, episodes=10, write_result=True)
        
        print(f"\n{'='*60}")
        print("시뮬레이션 완료!")
        print(f"{'='*60}")
        
        # 결과 비교
        if ddqn_results and ppo_results: # and mappo_results:
            print("\n모델별 성능 비교:")
            print(f"{'모델':<10} {'평균 보상':<12} {'평균 서비스율':<15}")
            print("-" * 40)
            
            ddqn_avg_reward = sum(e['total_reward'] for e in ddqn_results) / len(ddqn_results)
            ppo_avg_reward = sum(e['total_reward'] for e in ppo_results) / len(ppo_results)
            # mappo_avg_reward = sum(e['total_reward'] for e in mappo_results) / len(mappo_results)
            
            ddqn_service_rate = sum(e['total_num_serve'] for e in ddqn_results) / sum(e['total_num_accept'] for e in ddqn_results) if sum(e['total_num_accept'] for e in ddqn_results) > 0 else 0
            ppo_service_rate = sum(e['total_num_serve'] for e in ppo_results) / sum(e['total_num_accept'] for e in ppo_results) if sum(e['total_num_accept'] for e in ppo_results) > 0 else 0
            # mappo_service_rate = sum(e['total_num_serve'] for e in mappo_results) / sum(e['total_num_accept'] for e in mappo_results) if sum(e['total_num_accept'] for e in mappo_results) > 0 else 0
            
            print(f"{'DDQN':<10} {ddqn_avg_reward:<12.2f} {ddqn_service_rate:<15.3f}")
            print(f"{'PPO':<10} {ppo_avg_reward:<12.2f} {ppo_service_rate:<15.3f}")
            # print(f"{'MAPPO':<10} {mappo_avg_reward:<12.2f} {mappo_service_rate:<15.3f}")
    

if __name__ == "__main__":
    main()