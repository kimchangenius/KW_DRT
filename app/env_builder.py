import os
import csv
import json
import pandas as pd
import app.config as cfg
from app.request import Request
from app.network import DRTNetwork
from app.env2 import RideSharingEnvironment


class EnvBuilder:
    def __init__(
        self, data_dir, result_dir, request_filename='requests_80.csv',
        test_episode_path=None, num_vehicles=None,
    ):
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.test_episode_path = test_episode_path
        self.num_vehicles = int(num_vehicles or cfg.MAX_NUM_VEHICLES)
        self.test_metadata = None

        self.request_path = (
            os.path.join(data_dir, request_filename)
            if test_episode_path is None
            else None
        )
        self.vehicle_pos_path = os.path.join(data_dir, 'vehicle_positions.csv')
        self.od_matrix_path = os.path.join(data_dir, 'od_matrix_v2.csv')

    def load_requests(self, network):
        requests = []
        with open(self.request_path, newline='', encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pop = row.get("Population")
                num_passengers = int(pop) if pop not in (None, "") else 1
                req = Request(
                    request_id=int(row["User_ID"]),
                    from_node_id=int(row["Start_node"]),
                    to_node_id=int(row["End_node"]),
                    request_time=int(row["Request_time"]),
                    network=network,
                    num_passengers=num_passengers,
                )
                requests.append(req)
        return sorted(requests, key=lambda r: r.request_time)

    def _resolve_test_episode_path(self):
        if os.path.isabs(self.test_episode_path):
            return self.test_episode_path
        return os.path.join(self.data_dir, self.test_episode_path)

    def load_test_episode(self):
        path = self._resolve_test_episode_path()
        with open(path, encoding='utf-8') as jsonfile:
            payload = json.load(jsonfile)
        self.test_metadata = payload.get('metadata', {})
        return payload

    def load_test_requests(self, network, payload):
        requests = []
        for item in payload.get('requests', []):
            req = Request(
                request_id=int(item['id']),
                from_node_id=int(item['origin']),
                to_node_id=int(item['dest']),
                request_time=int(item['request_time']),
                network=network,
                num_passengers=int(item.get('n_passengers', 1)),
            )
            req.travel_time = float(item.get(
                'direct_travel',
                network.get_duration(req.from_node_id, req.to_node_id),
            ))
            req.pickup_due = float(item.get(
                'pickup_deadline',
                req.request_time + cfg.MAX_WAIT_TIME,
            ))
            req.arrival_due = float(item.get(
                'arrival_deadline',
                req.request_time + req.travel_time + cfg.MAX_INVEHICLE_TIME,
            ))
            requests.append(req)
        return sorted(requests, key=lambda r: (r.request_time, r.id))

    def load_test_vehicle_positions(self, payload):
        vehicles_by_n = payload.get('vehicles_by_N', {})
        key = str(self.num_vehicles)
        if key not in vehicles_by_n:
            available = ', '.join(sorted(vehicles_by_n.keys()))
            raise ValueError(
                f"Test episode has no vehicles_by_N['{key}']; "
                f"available fleet sizes: {available}"
            )
        return [int(node_id) for node_id in vehicles_by_n[key]]

    def build(self):
        network = DRTNetwork()
        network.set_od_matrix(self.od_matrix_path)

        if self.test_episode_path is None:
            request_list = self.load_requests(network)
            for r in request_list:
                r.set_travel_time(network.get_duration(r.from_node_id, r.to_node_id))
            vehicle_positions = pd.read_csv(self.vehicle_pos_path)['initial_position'].tolist()
        else:
            payload = self.load_test_episode()
            request_list = self.load_test_requests(network, payload)
            vehicle_positions = self.load_test_vehicle_positions(payload)

        env = RideSharingEnvironment(
            network=network,
            original_request_list=request_list,
            vehicle_init_pos=vehicle_positions
        )
        return env
