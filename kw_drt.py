import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
import networkx as nx
import os
import csv
import json

#########################################################
# class SiouxFallsNetwork
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
# main
#########################################################
def main():
    # data load & save
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    result_dir = os.path.join(current_dir, 'result')
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

    try:
        vehicle_positions = pd.read_csv(vehicle_positions)['initial_position'].tolist()
    except FileNotFoundError:
        print(f"Error: {vehicle_positions} not found.")
        return
    except KeyError:
        print(f"Error: 'initial_position' column not found in {vehicle_positions}.")
        return

if __name__ == "__main__":
    main()