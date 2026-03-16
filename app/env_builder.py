import os
import csv
import pandas as pd
from app.request import Request
from app.network import DRTNetwork
from app.env import RideSharingEnvironment


class EnvBuilder:
    def __init__(self, data_dir, result_dir):
        self.data_dir = data_dir
        self.result_dir = result_dir

        self.request_path = os.path.join(data_dir, 'requests_80.csv')
        self.vehicle_pos_path = os.path.join(data_dir, 'vehicle_positions.csv')
        self.od_matrix_path = os.path.join(data_dir, 'od_matrix.csv')

    def load_requests(self, network):
        requests = []
        with open(self.request_path, newline='', encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                req = Request(
                    request_id=int(row["User_ID"]),
                    from_node_id=int(row["Start_node"]),
                    to_node_id=int(row["End_node"]),
                    request_time=int(row["Request_time"]),
                    network=network
                )
                requests.append(req)
        return sorted(requests, key=lambda r: r.request_time)

    def build(self):
        network = DRTNetwork()
        network.set_od_matrix(self.od_matrix_path)

        request_list = self.load_requests(network)
        for r in request_list:
            r.set_travel_time(network.get_duration(r.from_node_id, r.to_node_id))

        vehicle_positions = pd.read_csv(self.vehicle_pos_path)['initial_position'].tolist()

        env = RideSharingEnvironment(
            network=network,
            original_request_list=request_list,
            vehicle_init_pos=vehicle_positions
        )
        return env
