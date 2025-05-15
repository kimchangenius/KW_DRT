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
        status_map = {'idle': 0, 'moving': 1, 'pickup': 2, 'dropoff': 3, 'rebalance': 4}
        for v in self.vehicles:
            next_node = v.current_path[0] if v.current_path else -1
            vehicle_features.append([
                v.current_location,
                next_node,
                v.capacity - len(v.passengers),
                status_map.get(v.status, 0),
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
            pass

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

        elif action['action_type'] == ActionType.REBALANCING:
            cont = action['parameter'][0]
            max_node = len(self.network.graph.nodes)
            mapped = int(np.round(((cont + 1) / 2) * (max_node - 1))) + 1
            vehicle.rebalance_target = mapped
            vehicle.status = 'rebalance'
            self.rebalancing_count += 1

        dropped = vehicle.move_to_next_location()
        # status 갱신
        if dropped:
            vehicle.status = 'dropoff'
        elif vehicle.passengers:
            vehicle.status = 'moving'
        else:
            vehicle.status = 'idle'

        for p in list(self.passengers):
            if p.start == vehicle.current_location and p.pickup_time is None:
                if vehicle.add_passenger(p):
                    p.pickup_time = self.time
                    self.matched_ids.add(p.id)
                    step_matched += 1
                    self.passengers.remove(p)
                    vehicle.status = 'pickup'

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

        rebalancing_in_this_step = self.rebalancing_count - self.old_rebalance
        rebalance_reward = - rebalancing_in_this_step

        moving_vehicles = sum(1 for v in self.vehicles if v.remaining_travel_time > 0)
        emmision_reward = - moving_vehicles

        reward = matched_reward + dropoff_reward + qos_reward + rebalance_reward + emmision_reward

        return reward

    def step(self, agent):
        if self.time <= self.max_request_time:
            new = self.passengers_data[self.passengers_data['Request_time'] == self.time]
            existing = {p.id for p in self.passengers}
            for _, row in new.iterrows():
                if row['User_ID'] not in existing:
                    p = Passenger(row['User_ID'], row['Start_node'], row['End_node'], row['Request_time'], self.network)
                    self.passengers.append(p)
                    self.waiting_passenger_count[p.start] += 1

        cancel_limit = 10
        for p in list(self.passengers):
            if p.pickup_time is None and (self.time - p.request_time) > cancel_limit:
                self.canceled_ids.add(p.id)
                self.passengers.remove(p)
        self.canceled_passengers = len(self.canceled_ids)

        n_v = len(self.vehicles)
        self.old_rebalance = self.rebalancing_count
        total_reward = 0.0
        assigned = set()
        actions = [None]*n_v

        state = self.get_state()
        flat_state = self.flatten_state(state).reshape(1, -1)

        for _ in range(n_v):
            # 남은 차량 리스트
            remaining = [i for i in range(n_v) if i not in assigned]

            # 배치용 입력 생성
            batch = []
            for v_idx in remaining:
                V = state['vehicles'].copy()
                # 나머지 차량 마스킹
                mask = [i for i in range(n_v) if i!=v_idx]
                V[mask] = 0.0
                batch.append(np.concatenate([V.flatten(), state['requests'].flatten()]))
            batch = tf.convert_to_tensor(np.stack(batch), dtype=tf.float32)

            # 한 번에 예측
            out = agent.model(batch, training=False)
            Q = out['q_all'].numpy()           # shape (len(remaining), total_actions)
            best_flat = Q.reshape(-1).argmax()  # 전체에서 best
            veh_i = best_flat // agent.total_actions
            act_i = best_flat % agent.total_actions

            v_idx = remaining[veh_i]
            # 액션 dict 생성
            if act_i < agent.num_reject:
                act = {'action_type':ActionType.REJECT,'discrete_index':0,'parameter':None}
            elif act_i < agent.num_reject+agent.num_matching:
                m = act_i - agent.num_reject
                param = out['p_matching'].numpy()[veh_i][m]
                act = {'action_type':ActionType.MATCHING,'discrete_index':m,'parameter':param}
            else:
                r = act_i - agent.num_reject - agent.num_matching
                param = out['p_rebalance'].numpy()[veh_i][r]
                act = {'action_type':ActionType.REBALANCING,'discrete_index':r,'parameter':param}

            if act['action_type'] == ActionType.REBALANCING:
                cont = act['parameter'][0]
                max_n = len(self.network.graph.nodes)
                mapped = int(np.round(((cont + 1) / 2) * (max_n - 1))) + 1
                v = self.vehicles[v_idx]
                v.rebalance_target = mapped
                v.status = 'rebalance'
                self.rebalancing_count += 1

            # 적용
            r = self.single_action(v_idx, act)
            total_reward += r
            actions[v_idx] = act
            assigned.add(v_idx)

            next_state = self.get_state()
            flat_next = self.flatten_state(next_state).reshape(1, -1)

            agent.remember(flat_state, [act], r, flat_next, done=False)
            agent.replay()

            state = next_state
            flat_state = flat_next


        self.time += 1
        return total_reward, actions

#########################################################
# DQN Agent
#########################################################
class ActionType(enum.Enum):
    REJECT = 0
    MATCHING = 1
    REBALANCING = 2

class DQNAgent:
    def __init__(self, state_size):
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
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32

        self.memory = deque(maxlen=20000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_update_counter = 0
        self.train_counter = 0

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
        p_matching = Dense(self.num_matching * self.param_dim_matching,
                           activation='tanh', name='p_matching')(x)
        p_matching = Reshape((self.num_matching, self.param_dim_matching),
                             name='p_matching_reshape')(p_matching)

        q_rebalance = Dense(self.num_rebalancing, activation='linear', name='q_rebalance')(x)
        p_rebalance = Dense(self.num_rebalancing * self.param_dim_rebalancing,
                            activation='tanh', name='p_rebalance')(x)
        p_rebalance = Reshape((self.num_rebalancing, self.param_dim_rebalancing),
                              name='p_rebalance_reshape')(p_rebalance)

        q_all = Concatenate(name='q_all')(
            [q_reject, q_matching, q_rebalance]
        )

        model = Model(inputs=state, outputs={
            'q_all': q_all,
            'p_matching': p_matching,
            'p_rebalance': p_rebalance
        })
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            idx = random.randrange(self.total_actions)
            if idx < self.num_reject:
                return {'action_type': ActionType.REJECT,
                        'discrete_index': 0, 'parameter': None}
            elif idx < self.num_reject + self.num_matching:
                m = idx - self.num_reject
                p = np.random.uniform(-1, 1, self.param_dim_matching)
                return {'action_type': ActionType.MATCHING,
                        'discrete_index': m, 'parameter': p}
            else:
                r = idx - self.num_reject - self.num_matching
                p = np.random.uniform(-1, 1, self.param_dim_rebalancing)
                return {'action_type': ActionType.REBALANCING,
                        'discrete_index': r, 'parameter': p}

        tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        out = self.model(tensor, training=False)
        q_all = out['q_all'].numpy()[0]
        p_m = out['p_matching'].numpy()[0]
        p_r = out['p_rebalance'].numpy()[0]
        best = np.argmax(q_all)

        if best < self.num_reject:
            return {'action_type': ActionType.REJECT,
                    'discrete_index': 0, 'parameter': None}
        elif best < self.num_reject + self.num_matching:
            m = best - self.num_reject
            return {'action_type': ActionType.MATCHING,
                    'discrete_index': m, 'parameter': p_m[m]}
        else:
            r = best - self.num_reject - self.num_matching
            return {'action_type': ActionType.REBALANCING,
                    'discrete_index': r, 'parameter': p_r[r]}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        self.train_counter += 1
        if self.train_counter % 5 != 0 or len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        loss = 0.0
        for state, action_list, reward, next_state, done in batch:
            # 1) next Q 계산 (target network)
            next_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
            out_n = self.target_model(next_tensor, training=False)
            q_next = out_n['q_all'].numpy()[0]
            target = reward + (0 if done else self.gamma * np.max(q_next))

            # 2) current Q 계산 (online network)
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            out_c = self.model(state_tensor, training=False)
            q_all = out_c['q_all'].numpy()[0]

            # 3) Q 업데이트
            for act in action_list:
                if act['action_type'] == ActionType.REJECT:
                    idx = 0
                elif act['action_type'] == ActionType.MATCHING:
                    idx = act['discrete_index'] + self.num_reject
                else:
                    idx = act['discrete_index'] + self.num_reject + self.num_matching
                q_all[idx] = target

            # 4) train_on_batch
            target_out = {
                'q_all': q_all.reshape((1, -1)),
                'p_matching': out_c['p_matching'].numpy(),
                'p_rebalance': out_c['p_rebalance'].numpy()
            }
            loss = self.model.train_on_batch(
                state_tensor, target_out
            )[0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.target_update_counter += 1
        if self.target_update_counter % 100 == 0:
            self.update_target_model()

        return loss


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

    travel_time_output = os.path.join(result_dir, 'travel_time.csv')
    network.save_travel_time(travel_time_output)

    od_matrix_output = os.path.join(result_dir, 'od_matrix.csv')
    network.generate_od_matrix(od_matrix_output)

    print("Simulation setup complete")

    vehicle_positions = pd.read_csv(vehicle_positions)['initial_position'].tolist()
    passenger_data = pd.read_csv(passenger_data)

    state_size = len(vehicle_positions) * 5 + 20 * 3

    weight_path = os.path.join(result_dir, "dqn_model_weights_final.weight.h5")
    agent = DQNAgent(state_size)
    agent.load_model(weight_path)

    episode_logs = []
    episodes = 500
    sec= 60

    for ep in range(episodes):
        step_logs = []
        env = RideSharingEnvironment(
            network=network,
            capacity=5,
            passenger_data=passenger_data,
            vehicle_positions=vehicle_positions
        )

        total_reward = 0.0
        total_loss = 0.0
        prev_dropped = 0

        for t in range(sec):
            # 빈 차량 재배치 목표
            for v in env.vehicles:
                if not v.passengers:
                    hd = env.get_high_demand_nodes()
                    if hd:
                        v.rebalance_target = min(
                            hd,
                            key=lambda nd: nx.shortest_path_length(
                                env.network.graph, v.current_location, nd, weight="weight"
                            )
                        )

            reward, actions = env.step(agent)
            total_reward += reward

            loss = agent.replay()
            if loss is None:
                loss = 0.0
            total_loss += loss

            dropped_this_step = env.dropped_passengers - prev_dropped
            prev_dropped = env.dropped_passengers

            waiting = sum(1 for p in env.passengers if p.pickup_time is None)

            matched = len(env.matched_ids)
            canceled = len(env.canceled_ids)
            total_accounted = matched + canceled + waiting

            step_logs.append({
                "Step": t + 1,
                "Total Reward": total_reward,
                "Loss": loss,
                "Matched Passengers": matched,
                "Canceled": canceled,
                "Dropped Passengers": dropped_this_step,
                "Waiting Passengers": waiting,
                "Total Accounted Passengers": total_accounted,
                "Rebalancing Count": env.rebalancing_count
            })

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
