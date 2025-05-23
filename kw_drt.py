import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Reshape
import networkx as nx
import os
import csv
import json
from pprint import pprint
import enum

from tensorflow.python.distribute.combinations import tf_function
from tensorflow.python.keras.utils.version_utils import training
from tensorflow.python.ops.gen_batch_ops import batch


#########################################################
# SiouxFallsNetwork
#########################################################
class SiouxFallsNetwork:
    def __init__(self, net_data, flow_data, node_coord_data, node_xy_data):
        self.sioux_falls_df, self.node_coord, self.node_xy = self.load_data(net_data, flow_data, node_coord_data, node_xy_data)
        self.graph = self.create_graph()
        self.travel_time = self.initialize_travel_time()

    def load_data(self, net_data, flow_data, node_coord_data, node_xy_data):
        net = pd.read_csv(net_data, skiprows=8, sep='\t').drop(['~', ';'], axis=1, errors='ignore')
        net['edge'] = net.index + 1
        flow = pd.read_csv(flow_data, sep='\t').drop(['From ', 'To '], axis=1, errors='ignore')
        flow.rename(columns={"Volume ": "flow", "Cost ": "cost"}, inplace=True)
        node_coord = pd.read_csv(node_coord_data, sep='\t').drop([';'], axis=1, errors='ignore')
        node_xy = pd.read_csv(node_xy_data, sep='\t')

        sioux_falls_df = pd.concat([net, flow], axis=1)
        return sioux_falls_df, node_coord, node_xy

    def create_graph(self):
        G = nx.from_pandas_edgelist(self.sioux_falls_df, 'init_node', 'term_node',
                                    ['capacity', 'length', 'free_flow_time', 'b', 'power', 'speed', 'toll', 'link_type',
                                     'edge', 'flow', 'cost'],
                                    create_using=nx.MultiDiGraph())

        # Coordinate position (using pos_xy for better visualization)
        pos_xy = dict([(i, (a, b)) for i, a, b in zip(self.node_xy.Node, self.node_xy.X, self.node_xy.Y)])
        nx.set_node_attributes(G, pos_xy, 'pos')

        return G

    def initialize_travel_time(self):
        travel_time = {}
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            # 항상 랜덤 값을 사용
            random_time = np.random.randint(1, 4)
            travel_time[(u, v, k)] = random_time

        nx.set_edge_attributes(self.graph, travel_time, "weight")
        self.travel_time = travel_time
        return travel_time

    def get_shortest_path(self, start, end):
        try:
            path = nx.shortest_path(self.graph, source=start, target=end, weight="weight")
            return path
        except:
            print(f"No path exists between Node {start} and Node {end}.")
            return []

    def get_travel_time(self, path):
        if len(path) < 2:
            return float('inf')
        total_time = 0
        for i in range(len(path) - 1):
            edges = self.graph.get_edge_data(path[i], path[i + 1])
            if edges:
                total_time += min([edge_data.get('weight', float('inf')) for key, edge_data in edges.items()])
            else:
                print(f"No dege exists betwwen Node {path[i]} and Node {path[i + 1]}.")
                return float('inf')
        return total_time

    def save_travel_time(self, output):
        with open(output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['From', 'To', 'Key', 'TravelTime'])
            for (u, v, k), travel_time in self.travel_time.items():
                writer.writerow([u, v, k, travel_time])
        print(f"Travel time data saved to {output}")

    def generate_od_matrix(self, output):
        nodes = list(self.graph.nodes)
        num_nodes = len(nodes)
        od_matrix = np.full((num_nodes, num_nodes), np.inf)

        for i, origin in enumerate(nodes):
            for j, destination in enumerate(nodes):
                if origin == destination:
                    od_matrix[i, j] = 0
                else:
                    try:
                        travel_time = nx.shortest_path_length(self.graph, source=origin, target=destination,
                                                              weight="weight")
                        od_matrix[i, j] = travel_time
                    except nx.NetworkXNoPath:
                        pass

        pd.DataFrame(od_matrix, index=nodes, columns=nodes).to_csv(output, index_label='Origin')
        print(f"OD matrix saved to {output}")


#########################################################
# Passenger
#########################################################
class Passenger:
    def __init__(self, id, start, end, request_time, network):
        self.id = id
        self.start = start
        self.end = end
        self.request_time = request_time
        self.network = network
        self.pickup_time = None
        self.dropoff_time = None
        self.direct_route_time = network.get_travel_time(network.get_shortest_path(start, end))


#########################################################
# Vehicle
#########################################################
class Vehicle:
    def __init__(self, id, capacity, current_location, network, env):
        self.id = id
        self.capacity = capacity
        self.current_location = current_location
        self.network = network
        self.env = env
        self.passengers = []
        self.status = 'idle'
        self.current_request = None

        # 기존 환경 변수
        self.remaining_travel_time = 0
        self.current_path = []
        self.rebalance_target = None
        self.rebalance_count = 0
        self.idle_time = 0
        self.dropped_passengers = 0

    def add_passenger(self, passenger):
        if len(self.passengers) < self.capacity:
            self.passengers.append(passenger)
            self.idle_time = 0
            self.status = self.env.status_map.get(self.status, 2)
            self.current_request = {'type': 'pickup', 'passenger_id': passenger.id}
            return True
        return False

    def remove_passenger(self, passenger):
        if passenger in self.passengers:
            self.passengers.remove(passenger)

    def move_to_next_location(self):
        dropped = []

        if self.remaining_travel_time > 0:
            self.remaining_travel_time -= 1

        if self.remaining_travel_time == 0 and self.current_path:
            self.current_location = self.current_path.pop(0)

            if self.current_path:
                next_node = self.current_path[0]
                self.remaining_travel_time = self.network.get_travel_time([self.current_location, next_node])
            else:
                self.remaining_travel_time = 0

        if not self.current_path and self.remaining_travel_time == 0:
            self.idle_time += 1

        for passenger in self.passengers[:]:
            if passenger.end == self.current_location:
                self.remove_passenger(passenger)
                passenger.dropoff_time = self.env.time
                self.dropped_passengers += 1
                self.env.dropped_passengers += 1
                dropped.append(passenger)

        return dropped



#########################################################
# RideSharingEnvironment
#########################################################
class RideSharingEnvironment:
    def __init__(self, network, capacity, passenger_data, vehicle_positions):
        self.network = network
        self.vehicles = self.initialize_vehicles(capacity, vehicle_positions)
        # self.passengers = self.initialize_passengers(passenger_data)
        self.passengers = []
        self.passengers_data = passenger_data
        # 기존 환경 변수
        self.time = 0
        self.canceled_passengers = 0
        self.rebalancing_count = 0
        self.old_rebalance = 0
        self.total_passenger_id = 0
        self.matched_passengers = 0
        self.dropped_passengers = 0
        self.last_dropped = []
        self.high_demand_nodes = []
        self.max_request_time = self.passengers_data['Request_time'].max()
        self.waiting_passenger_count = np.zeros(len(self.network.graph.nodes))

        self.matched_ids = set()
        self.canceled_ids = set()

        self.base_fare = 1
        self.VOT = 1.0

        self.status_map = {'idle': 0, 'reject': 1, 'pickup': 2, 'dropoff': 3, 'rebalance': 4}

    def initialize_vehicles(self, capacity, vehicle_position):
        vehicle = []
        for idx, pos in enumerate(vehicle_position):
            vehicle.append(Vehicle(idx, capacity, pos, self.network, self))
        return vehicle

    # def initialize_passengers(self, passenger_data):
    #     passengers = []
    #     for _, row in passenger_data.iterrows():
    #         passenger = Passenger(
    #             id = row['User_ID'],
    #             start = row['Start_node'],
    #             end = row['End_node'],
    #             request_time = row['Request_time'],
    #             network=self.network
    #         )
    #         passengers.append(passenger)
    #     return passengers

    def get_high_demand_nodes(self):
        recent_requests = [p.start for p in self.passengers if self.time - p.request_time <= 10]
        demand_count = {node: recent_requests.count(node) for node in self.network.graph.nodes}
        historical_data_factor = {node: random.uniform(1.0, 1.5) for node in self.network.graph.nodes}
        for node in demand_count:
            demand_count[node] *= historical_data_factor[node]
        if sum(demand_count.values()) == 0:
            return random.sample(list(self.network.graph.nodes), 3)
        return sorted(demand_count, key=demand_count.get, reverse=True)[:3]

    def generate_passengers_for_current_time(self):
        if self.time <= self.max_request_time:
            new = self.passengers_data[self.passengers_data['Request_time'] == self.time]
            existing_ids = {p.id for p in self.passengers}
            for _, row in new.iterrows():
                if row['User_ID'] not in existing_ids:
                    p = Passenger(row['User_ID'], row['Start_node'], row['End_node'], row['Request_time'], self.network)
                    self.passengers.append(p)
                    self.waiting_passenger_count[p.start] += 1

    def get_state(self):
        """
        V_t
        차량의 상태: vehicle.status
        현재 차량의 위치: vehicle.current_location
        차량이 이동해야하는 리스트: vehicle.current_path[0] == next_node
        남은 좌석수: vehicle.capacity - len(vehicle.passengers)
        재배치 목표: vehicle.rebalance_target

        R_t
        요청 최대 20개 queue: 대기 중(request_time None)인 승객을 request_time 순으로 정렬한 후 상위 20개
        부족 시 [0,0,-1]로 패딩
        """
        # 차량 상태 V_t
        vehicle_features = []
        for v in self.vehicles:
            next_node = v.current_path[0] if v.current_path else -1
            vehicle_features.append([
                v.current_location,
                next_node,
                v.capacity - len(v.passengers),
                self.status_map.get(v.status, 0),
                v.rebalance_target or -1
            ])
        vehicle_array = np.array(vehicle_features, dtype=float)

        # 요청 상태 R_t
        waiting = [p for p in self.passengers if p.pickup_time is None]
        waiting.sort(key=lambda p: p.request_time, reverse=True)
        top20 = waiting[:20]
        req_feats = [[p.start, p.end, p.request_time] for p in top20]
        while len(req_feats) < 20:
            req_feats.append([0, 0, -1])
        requests_array = np.array(req_feats, dtype=float)

        return {'vehicles': vehicle_array, 'requests': requests_array}

    def flatten_state(self, state):
        vehicles_flat = state['vehicles'].flatten()
        requests_flat = state['requests'].flatten()

        return np.concatenate([vehicles_flat, requests_flat])

    def single_action(self, v_idx, action):
        """
        차량 v_idx에 대해 단일 action 적용, 반환 보상
        """
        vehicle = self.vehicles[v_idx]

        step_matched = 0
        qos_reward = 0.0

        if action['action_type'] == ActionType.REJECT:
            self.status_map.get(vehicle.status, 1)
            vehicle.current_request = {'type': 'reject'}

        elif action['action_type'] == ActionType.MATCHING:
            p_idx = action['discrete_index']
            if 0 <= p_idx < len(self.passengers):
                p = self.passengers[p_idx]
                if p.pickup_time is None:
                    p.pickup_time = self.time
                    vehicle.add_passenger(p)
                    self.matched_ids.add(p.id)
                    step_matched += 1
                    self.passengers.remove(p)
                    self.waiting_passenger_count[p.start] -= 1

                    path_to_pickup = self.network.get_shortest_path(
                        vehicle.current_location, p.start
                    )
                    path_to_dropoff = self.network.get_shortest_path(
                        p.start, p.end
                    )
                    full_path = path_to_pickup + path_to_dropoff[1:]
                    vehicle.current_path = full_path

                    if len(full_path) > 1:
                        vehicle.remaining_travel_time = self.network.get_travel_time(
                            [full_path[0], full_path[1]]
                        )
                    else:
                        vehicle.remaining_travel_time = 0

                    self.status_map.get(vehicle.status, 2)
                    vehicle.current_request = {'type': 'pickup', 'passenger_id': p.id}


        elif action['action_type'] == ActionType.REBALANCING:
            high_nodes = self.get_high_demand_nodes()
            r = action['discrete_index']
            target = high_nodes[r]
            vehicle.rebalance_target = target

            full_path = self.network.get_shortest_path(
                vehicle.current_location, target
            )
            vehicle.current_path = full_path
            if len(full_path) > 1:
                vehicle.remaining_travel_time = self.network.get_travel_time(
                    [full_path[0], full_path[1]]
                )
            else:
                vehicle.remaining_travel_time = 0

            self.status_map.get(vehicle.status, 4)
            vehicle.current_request = {'type': 'rebalance', 'target': target}
            self.rebalancing_count += 1

        dropped = vehicle.move_to_next_location()
        if dropped:
            self.status_map.get(vehicle.status, 3)
            vehicle.current_request = {'type': 'dropoff',
                                       'passenger_ids': [p.id for p in dropped]}

        for p in list(self.passengers):
            if p.start == vehicle.current_location and p.pickup_time is None:
                if vehicle.add_passenger(p):
                    p.pickup_time = self.time
                    self.matched_ids.add(p.id)
                    step_matched += 1
                    self.passengers.remove(p)
                    self.status_map.get(vehicle.status, 2)

        cumulative = len(self.matched_ids) + len(self.canceled_ids) + len(self.passengers)
        matched_ratio = len(self.matched_ids) / cumulative if cumulative > 0 else 0
        matched_reward = step_matched * self.base_fare * matched_ratio

        step_drop = len(dropped)
        dropoff_ratio = (self.dropped_passengers / len(self.matched_ids)
                         if len(self.matched_ids) > 0 else 0)
        dropoff_reward = step_drop * self.base_fare * 1.2 * dropoff_ratio

        for p in dropped:
            if p.pickup_time is not None and p.dropoff_time == self.time:
                t_wait = max(0, p.pickup_time - p.request_time)
                trav = p.dropoff_time - p.pickup_time
                detour = max(0, trav - p.direct_route_time)
                qos_reward -= (t_wait + detour)

        # moving_vehicles = sum(1 for v in self.vehicles if v.remaining_travel_time > 0)
        # emmision_reward = - moving_vehicles

        reward = matched_reward + dropoff_reward + qos_reward

        # reward = np.clip(reward, -5.0, 5.0)
        return reward

    def update_current_requirement(self):
        if self.time <= self.max_request_time:
            self.generate_passengers_for_current_time()
        cancel_limit = 10
        for p in list(self.passengers):
            if p.pickup_time is None and (self.time - p.request_time) > cancel_limit:
                self.canceled_ids.add(p.id)
                self.passengers.remove(p)

    def step(self, veh_i, act):
        # self.get_current_requirement()
        reward = self.single_action(veh_i, act)

        next_state = self.get_state()
        return reward, next_state

#########################################################
# DQN Agent
#########################################################
class ActionType(enum.Enum):
    REJECT = 0
    MATCHING = 1
    REBALANCING = 2

class DQNAgent:
    def __init__(self, state_size, max_episodes=500):
        self.state_size = state_size

        self.num_reject = 1
        self.num_matching = 20
        self.num_rebalancing = 3
        self.param_dim_matching = 1
        self.param_dim_rebalancing = 1

        self.total_actions = (
            self.num_reject +
            self.num_matching +
            self.num_rebalancing
        )

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_delta = (self.epsilon - self.epsilon_min) / max_episodes
        self.batch_size = 32

        self.memory = deque(maxlen=20000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_update_counter = 0
        self.train_counter = 0
        self.target_update_freq = 1000


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, file_path):
        self.model.save_weights(file_path)
        print(f"Model weights saved at {file_path}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model.load_weights(file_path)
            self.update_target_model()
            print(f"Model weights loaded at {file_path}")
        else:
            print(f"No model weights loaded at{file_path}")

    def build_model(self):
        state = Input(shape=(self.state_size,), name='state')
        x = Dense(128, activation='relu')(state)
        x = Dense(128, activation='relu')(x)

        q_reject = Dense(self.num_reject, activation='linear', name='q_reject')(x)
        q_matching = Dense(self.num_matching, activation='linear', name='q_matching')(x)
        q_rebalance = Dense(self.num_rebalancing, activation='linear', name='q_rebalance')(x)

        q_all = Concatenate(name='q_all')(
            [q_reject, q_matching, q_rebalance]
        )

        model = Model(inputs=state, outputs=q_all)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=5e-5,
            clipnorm=0.5
        )
        model.compile(
            optimizer=optimizer,
            loss='huber'
        )
        return model

    @tf_function
    def predict_q_all(self, states: tf.Tensor) -> tf.Tensor:
        return self.model(states, training=False)

    @tf_function
    def predict_target_q_all(self, states: tf.Tensor) -> tf.Tensor:
        return self.target_model(states, training=False)

    def act(self, flat_states: np.ndarray, vehicles: list):
        batch = tf.convert_to_tensor(flat_states, dtype=tf.float32)
        q_all = self.predict_q_all(batch).numpy()

        n_v = flat_states.shape[0]
        req_flat_start = self.state_size - 20 * 3
        requests_flat = flat_states[:, req_flat_start:]
        request_times = requests_flat.reshape(n_v, 20, 3)[:, :, 2]

        for i in range(n_v):
            for m in range(self.num_matching):
                if request_times[i, m] < 0:
                    q_all[i, 1 + m] = -np.inf

        for i, v in enumerate(vehicles):
            if v.status != 'idle':
                q_all[i, :] = -np.inf

        idle_idxs = [i for i, v in enumerate(vehicles) if v.status == 'idle']
        valid_pairs = []
        for i in idle_idxs:
            for j, q in enumerate(q_all[i]):
                if not np.isneginf(q):
                    valid_pairs.append((i, j))

        # ε-greedy 탐색 or 활용
        if np.random.rand() < self.epsilon and valid_pairs:
            veh_i, idx = random.choice(valid_pairs)
        else:
            flat_q = q_all.reshape(-1)
            if np.all(np.isneginf(flat_q)):
                veh_i = random.choice(idle_idxs) if idle_idxs else 0
                idx = 0
            else:
                best_flat = int(np.argmax(flat_q))
                veh_i = best_flat // self.total_actions
                idx = best_flat % self.total_actions

        # action dict 생성
        if idx < self.num_reject:
            act = {
                'action_type': ActionType.REJECT,
                'discrete_index': 0,
                'parameter': None
            }
        elif idx < self.num_reject + self.num_matching:
            act = {
                'action_type': ActionType.MATCHING,
                'discrete_index': idx - self.num_reject,
                'parameter': None
            }
        else:
            act = {
                'action_type': ActionType.REBALANCING,
                'discrete_index': idx - self.num_reject - self.num_matching,
                'parameter': None
            }

        return veh_i, act

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        self.train_counter += 1
        # 매 5스텝마다, 메모리가 충분할 때만 학습
        if self.train_counter % 5 != 0 or len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        losses = []

        for state, action_list, reward, next_state, done in batch:
            # 0) 텐서 준비 (batch size=1)
            st_t = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            nxt_t = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)

            # 1) online 네트워크로부터 next Q 선택용
            q_next_on = self.predict_q_all(nxt_t).numpy().reshape(-1)  # shape=(A,)
            a_max = int(np.argmax(q_next_on))

            # 2) target 네트워크로부터 next Q 평가용
            q_next_tg = self.predict_target_q_all(nxt_t).numpy().reshape(-1)
            max_q_next = float(q_next_tg[a_max])

            # 3) 벨만 타깃값
            target_q_value = reward if done else reward + self.gamma * max_q_next

            # 4) 현재 상태에서 Q값
            q_current = self.predict_q_all(st_t).numpy().reshape(-1)

            # 5) 실제 취한 action index로 Q 업데이트
            for act in action_list:
                if act['action_type'] == ActionType.REJECT:
                    idx = 0
                elif act['action_type'] == ActionType.MATCHING:
                    idx = act['discrete_index'] + self.num_reject
                else:  # REBALANCING
                    idx = act['discrete_index'] + self.num_reject + self.num_matching
                q_current[idx] = target_q_value

            loss = self.model.train_on_batch(
                st_t,
                q_current[np.newaxis, :]  # shape=(1, A)
            )
            losses.append(loss)

        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.update_target_model()

        return float(np.mean(losses))

    def decay_epsilon(self, ep, episodes):
        if ep < 100:
            self.epsilon = 1.0
        else:
            self.epsilon = max(self.epsilon_min,
                               1.0 - (ep - 100) * (1.0 - self.epsilon_min) / (episodes - 100))


#########################################################
# main
#########################################################
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
