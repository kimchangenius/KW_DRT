from collections import defaultdict
from datetime import datetime, timezone
import json
import os

import app.config as cfg
from app.vehicle_status import VehicleStatus
from app.request_status import RequestStatus


STATUS_MAP_VEH = {
    VehicleStatus.IDLE: 'idle',
    VehicleStatus.REJECT: 'idle',
    VehicleStatus.PICKUP: 'picking_up',
    VehicleStatus.DROPOFF: 'carrying',
}

STATUS_MAP_REQ = {
    RequestStatus.PENDING: 'waiting',
    RequestStatus.ACCEPTED: 'waiting',
    RequestStatus.PICKEDUP: 'picked_up',
    RequestStatus.SERVED: 'delivered',
    RequestStatus.CANCELLED: 'cancelled',
}


def sim_config_payload(config=None, max_num_request=None):
    config = config or {}
    return {
        'maxNumVehicles': cfg.MAX_NUM_VEHICLES,
        'vehCapacity': cfg.VEH_CAPACITY,
        'maxNumRequest': max_num_request,
        'maxWaitTime': cfg.MAX_WAIT_TIME,
        'hiddenDim': config.get('hidden_dim'),
        'batchSize': config.get('batch_size'),
        'learningRate': config.get('learning_rate'),
    }


def extract_vehicle(v, environment):
    status_str = STATUS_MAP_VEH.get(v.status, 'idle')

    path = []
    path_progress = 0
    if v.status in (VehicleStatus.PICKUP, VehicleStatus.DROPOFF) and v.next_node > 0:
        path = [v.curr_node, v.next_node]
        total_dur = environment.network.get_duration(v.curr_node, v.next_node)
        if total_dur > 0 and v.target_arrival_time > 0:
            remaining = v.target_arrival_time - environment.curr_time
            progress = 1.0 - (remaining / total_dur)
            path_progress = max(0, min(1, progress))

    return {
        'id': v.id + 1,
        'currentNodeId': v.curr_node,
        'targetNodeId': v.next_node if v.next_node > 0 else None,
        'path': path,
        'pathProgress': round(path_progress, 2),
        'status': status_str,
        'passengerId': v.target_request.id if v.target_request else None,
        'totalTrips': v.num_serve,
        'totalDistance': 0,
    }


def _request_cancellation_time(r):
    cancel_at = getattr(r, 'cancel_at', None)
    if cancel_at is not None:
        return cancel_at
    if r.status == RequestStatus.CANCELLED and r.waiting_time is not None and r.waiting_time >= 0:
        return r.request_time + r.waiting_time
    return None


def extract_passenger(r):
    return {
        'id': r.id,
        'originNodeId': r.from_node_id,
        'destinationNodeId': r.to_node_id,
        'requestTime': r.request_time,
        'pickupTime': r.pickup_at,
        'deliveryTime': r.dropoff_at,
        'cancellationTime': _request_cancellation_time(r),
        'status': STATUS_MAP_REQ.get(r.status, 'waiting'),
        'assignedVehicleId': (r.assigned_v_id + 1) if r.assigned_v_id >= 0 else None,
    }


def compute_metrics(environment):
    served = [r for r in environment.done_request_list if r.status == RequestStatus.SERVED]
    waiting = [
        r for r in environment.active_request_list
        if r.status in (RequestStatus.PENDING, RequestStatus.ACCEPTED)
    ]
    in_transit = [
        r for r in environment.active_request_list
        if r.status == RequestStatus.PICKEDUP
    ]
    busy = [
        v for v in environment.vehicle_list
        if v.status not in (VehicleStatus.IDLE, VehicleStatus.REJECT)
    ]

    avg_wait = 0.0
    if served:
        avg_wait = sum(r.waiting_time for r in served) / len(served)

    avg_travel = 0.0
    if served:
        avg_travel = sum(r.in_vehicle_time for r in served if r.in_vehicle_time > 0) / max(len(served), 1)

    util = round(len(busy) / len(environment.vehicle_list) * 100) if environment.vehicle_list else 0
    cancelled = sum(1 for r in environment.done_request_list if r.status == RequestStatus.CANCELLED)

    return {
        'currentTime': environment.curr_time,
        'totalPassengersServed': len(served),
        'totalPassengersWaiting': len(waiting),
        'totalPassengersInTransit': len(in_transit),
        'averageWaitTime': round(avg_wait, 1),
        'averageTravelTime': round(avg_travel, 1),
        'vehicleUtilization': util,
        'cancelCount': cancelled,
        'activeVehicles': len(busy),
        'totalVehicles': len(environment.vehicle_list),
    }


def compute_wait_time_distribution(environment):
    buckets = {'0-2': 0, '3-5': 0, '6-10': 0, '10+': 0}
    for r in environment.done_request_list:
        if r.status == RequestStatus.SERVED and r.waiting_time is not None:
            wt = r.waiting_time
            if wt <= 2:
                buckets['0-2'] += 1
            elif wt <= 5:
                buckets['3-5'] += 1
            elif wt <= 10:
                buckets['6-10'] += 1
            else:
                buckets['10+'] += 1
    return [{'range': k, 'count': v} for k, v in buckets.items()]


def compute_request_status(environment):
    served = sum(1 for r in environment.done_request_list if r.status == RequestStatus.SERVED)
    in_transit = sum(1 for r in environment.active_request_list if r.status == RequestStatus.PICKEDUP)
    waiting = sum(
        1 for r in environment.active_request_list
        if r.status in (RequestStatus.PENDING, RequestStatus.ACCEPTED)
    )
    cancelled = sum(1 for r in environment.done_request_list if r.status == RequestStatus.CANCELLED)
    return [
        {'name': 'Served', 'value': served, 'color': '#10b981'},
        {'name': 'In vehicle', 'value': in_transit, 'color': '#3b82f6'},
        {'name': 'Waiting', 'value': waiting, 'color': '#f59e0b'},
        {'name': 'Cancelled', 'value': cancelled, 'color': '#ef4444'},
    ]


def compute_link_loads(environment):
    loads = defaultdict(int)
    for v in environment.vehicle_list:
        if v.status in (VehicleStatus.PICKUP, VehicleStatus.DROPOFF) and v.next_node > 0:
            loads[f'{v.curr_node}-{v.next_node}'] += 1
    return dict(loads)


def append_history_sample(environment, utilization_history, passenger_history, limit=200):
    metrics = compute_metrics(environment)
    utilization_history.append({
        'time': environment.curr_time,
        'utilization': metrics['vehicleUtilization'],
    })
    if len(utilization_history) > limit:
        utilization_history.pop(0)

    passenger_history.append({
        'time': environment.curr_time,
        'served': metrics['totalPassengersServed'],
        'waiting': metrics['totalPassengersWaiting'],
        'cancelled': metrics['cancelCount'],
    })
    if len(passenger_history) > limit:
        passenger_history.pop(0)


def build_state(environment, utilization_history=None, passenger_history=None, config=None):
    utilization_history = utilization_history or []
    passenger_history = passenger_history or []

    metrics = compute_metrics(environment)
    max_num_request = len(getattr(environment, 'original_request_list', []) or [])

    active_passengers = [
        extract_passenger(r) for r in environment.active_request_list
        if r.status in (RequestStatus.PENDING, RequestStatus.ACCEPTED, RequestStatus.PICKEDUP)
    ]
    served_passengers = [
        extract_passenger(r) for r in environment.done_request_list
        if r.status == RequestStatus.SERVED
    ]
    cancelled_passengers = [
        extract_passenger(r) for r in environment.done_request_list
        if r.status == RequestStatus.CANCELLED
    ]
    visible_passengers = active_passengers + served_passengers + cancelled_passengers

    return {
        'metrics': metrics,
        **sim_config_payload(config, max_num_request=max_num_request),
        'vehicles': [extract_vehicle(v, environment) for v in environment.vehicle_list],
        'passengers': visible_passengers,
        'waitTimeDistribution': compute_wait_time_distribution(environment),
        'utilizationHistory': list(utilization_history),
        'passengerHistory': list(passenger_history),
        'requestStatusData': compute_request_status(environment),
        'linkLoads': compute_link_loads(environment),
    }


def json_default(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'{type(obj).__name__} is not JSON serializable')


def create_replay():
    return {
        'frames': [],
        'utilizationHistory': [],
        'passengerHistory': [],
    }


def capture_replay_frame(environment, replay, config=None, history_sample_interval=2):
    if replay is None:
        return

    utilization_history = replay.setdefault('utilizationHistory', [])
    passenger_history = replay.setdefault('passengerHistory', [])
    curr_time = environment.curr_time

    if curr_time > 0 and curr_time % history_sample_interval == 0:
        if replay.get('lastHistorySampleTime') != curr_time:
            append_history_sample(environment, utilization_history, passenger_history)
            replay['lastHistorySampleTime'] = curr_time

    replay.setdefault('frames', []).append(
        build_state(
            environment,
            utilization_history=utilization_history,
            passenger_history=passenger_history,
            config=config,
        )
    )


def build_simulation_replay_payload(
    environment, replay, config=None, run_name='inference', version=1,
    generated_at=None,
):
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    return {
        'version': version,
        'generatedAt': generated_at,
        'runName': run_name,
        'config': sim_config_payload(
            config,
            max_num_request=len(getattr(environment, 'original_request_list', []) or []),
        ),
        'frames': replay.get('frames', []),
    }


def save_simulation_replay_json(
    path, environment, replay, config=None, run_name='inference',
    filename='simulation_replay.json',
):
    os.makedirs(path, exist_ok=True)
    payload = build_simulation_replay_payload(
        environment, replay, config=config, run_name=run_name,
    )
    filepath = os.path.join(path, filename)
    with open(filepath, mode='w', encoding='utf-8') as jsonfile:
        json.dump(payload, jsonfile, ensure_ascii=False, indent=2, default=json_default)
    return filepath
