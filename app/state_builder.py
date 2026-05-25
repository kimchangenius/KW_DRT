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


def _config_value(config, *keys):
    if not config:
        return None
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return None


def _normalize_operation_time(value, time_offset_min=0, field_name='timestamp',
                              strict=True):
    if value is None:
        return None
    if hasattr(value, 'item'):
        value = value.item()

    normalized = value - time_offset_min
    if hasattr(normalized, 'item'):
        normalized = normalized.item()

    if isinstance(normalized, float):
        if strict and not normalized.is_integer():
            raise ValueError(f"{field_name} must be an integer minute: {value}")
        normalized = round(normalized)

    return int(normalized)


def _rejected_operation_entry(request_id):
    return {
        'request_id': int(request_id),
        'status': 'rejected',
        'vehicle_id': None,
        'accept_time': None,
        'pickup_time': None,
        'dropoff_time': None,
    }


def _served_operation_entry(request, num_drt=None, time_offset_min=0,
                            strict=True):
    request_id = int(request.id)
    vehicle_id = getattr(request, 'assigned_v_id', -1)
    accepted_at = getattr(request, 'accepted_at', None)
    pickup_at = getattr(request, 'pickup_at', None)
    dropoff_at = getattr(request, 'dropoff_at', None)

    missing = []
    if vehicle_id is None or int(vehicle_id) < 0:
        missing.append('vehicle_id')
    if accepted_at is None:
        missing.append('accept_time')
    if pickup_at is None:
        missing.append('pickup_time')
    if dropoff_at is None:
        missing.append('dropoff_time')
    if missing:
        if strict:
            raise ValueError(
                f"Served request {request_id} is missing {', '.join(missing)}"
            )
        return _rejected_operation_entry(request_id)

    vehicle_id = int(vehicle_id)
    if strict and num_drt is not None and not (0 <= vehicle_id < int(num_drt)):
        raise ValueError(
            f"vehicle_id out of range for request {request_id}: {vehicle_id}"
        )

    accept_time = _normalize_operation_time(
        accepted_at, time_offset_min, 'accept_time', strict,
    )
    pickup_time = _normalize_operation_time(
        pickup_at, time_offset_min, 'pickup_time', strict,
    )
    dropoff_time = _normalize_operation_time(
        dropoff_at, time_offset_min, 'dropoff_time', strict,
    )

    if strict:
        if min(accept_time, pickup_time, dropoff_time) < 0:
            raise ValueError(
                f"Negative timestamp for served request {request_id}"
            )
        if not (accept_time <= pickup_time <= dropoff_time):
            raise ValueError(
                f"Non-monotonic timestamps for served request {request_id}: "
                f"{accept_time}, {pickup_time}, {dropoff_time}"
            )

    return {
        'request_id': request_id,
        'status': 'served',
        'vehicle_id': vehicle_id,
        'accept_time': accept_time,
        'pickup_time': pickup_time,
        'dropoff_time': dropoff_time,
    }


def _operation_request_ids(environment, strict=True):
    original_requests = list(getattr(environment, 'original_request_list', []) or [])
    if original_requests:
        request_ids = [int(r.id) for r in original_requests]
    else:
        request_ids = []
        for attr_name in ('future_request_list', 'active_request_list',
                          'done_request_list'):
            for request in getattr(environment, attr_name, []) or []:
                request_ids.append(int(request.id))

    if strict and len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate request_id found in request list")

    return sorted(set(request_ids))


def _operation_request_lookup(environment):
    requests_by_id = {}
    for attr_name in ('future_request_list', 'active_request_list',
                      'done_request_list'):
        for request in getattr(environment, attr_name, []) or []:
            requests_by_id[int(request.id)] = request
    return requests_by_id


def _operation_init_positions(environment, init_positions=None, num_drt=None):
    if init_positions is None:
        init_positions = getattr(environment, 'vehicle_init_pos', None)
    if init_positions is None:
        return None

    positions = [int(pos) for pos in init_positions]
    if num_drt is not None:
        positions = positions[:int(num_drt)]
    return positions


def build_operation_result_payload(
    environment,
    config=None,
    scenario=None,
    seed=None,
    policy_label=None,
    horizon_min=None,
    init_positions=None,
    time_offset_min=0,
    strict=True,
):
    """Build the operation-result JSON payload described in OPERATION_RESULT_SPEC.

    This simulator uses base-hour-relative integer minutes already, so the
    default ``time_offset_min`` is 0. Pass 11 * 60 only if the environment
    being dumped stores absolute minutes from midnight.
    """
    scenario = scenario if scenario is not None else _config_value(config, 'scenario')
    seed = seed if seed is not None else _config_value(
        config, 'scenario_seed', 'seed',
    )
    policy_label = (
        policy_label
        if policy_label is not None
        else _config_value(config, 'policy_label', 'policy')
    )
    horizon_min = horizon_min if horizon_min is not None else _config_value(
        config, 'horizon', 'horizon_min', 'horizon_minutes',
    )

    vehicle_list = getattr(environment, 'vehicle_list', None) or []
    num_drt = len(vehicle_list) if vehicle_list else cfg.MAX_NUM_VEHICLES
    init_positions = _operation_init_positions(
        environment, init_positions=init_positions, num_drt=num_drt,
    )

    meta = {}
    if scenario is not None:
        meta['scenario'] = str(scenario)
    if seed is not None:
        meta['seed'] = int(seed)
    if policy_label is not None:
        meta['policy_label'] = str(policy_label)
    meta['num_drt'] = int(num_drt)
    if horizon_min is not None:
        meta['horizon_min'] = int(horizon_min)
    meta['time_unit'] = 'minute'
    meta['time_origin'] = 'base_hour_relative'
    if init_positions is not None:
        meta['init_positions'] = init_positions

    requests_by_id = _operation_request_lookup(environment)
    results = []
    for request_id in _operation_request_ids(environment, strict=strict):
        request = requests_by_id.get(request_id)
        if request is not None and request.status == RequestStatus.SERVED:
            results.append(_served_operation_entry(
                request,
                num_drt=num_drt,
                time_offset_min=time_offset_min,
                strict=strict,
            ))
        else:
            results.append(_rejected_operation_entry(request_id))

    return {
        'meta': meta,
        'results': results,
    }


def operation_result_filename(scenario=None, seed=None):
    if scenario is not None and seed is not None:
        return f"operation_{scenario}_seed{int(seed)}.json"
    return 'operation_result.json'


def dump_operation_result_json(
    output_path,
    environment,
    config=None,
    scenario=None,
    seed=None,
    policy_label=None,
    horizon_min=None,
    init_positions=None,
    time_offset_min=0,
    strict=True,
):
    payload = build_operation_result_payload(
        environment,
        config=config,
        scenario=scenario,
        seed=seed,
        policy_label=policy_label,
        horizon_min=horizon_min,
        init_positions=init_positions,
        time_offset_min=time_offset_min,
        strict=strict,
    )
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, mode='w', encoding='utf-8') as jsonfile:
        json.dump(payload, jsonfile, ensure_ascii=False, indent=2,
                  default=json_default)
    return output_path


def save_operation_result_json(
    path,
    environment,
    config=None,
    scenario=None,
    seed=None,
    policy_label=None,
    horizon_min=None,
    init_positions=None,
    filename=None,
    time_offset_min=0,
    strict=True,
):
    scenario = scenario if scenario is not None else _config_value(config, 'scenario')
    seed = seed if seed is not None else _config_value(
        config, 'scenario_seed', 'seed',
    )
    filename = filename or operation_result_filename(scenario, seed)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    return dump_operation_result_json(
        filepath,
        environment,
        config=config,
        scenario=scenario,
        seed=seed,
        policy_label=policy_label,
        horizon_min=horizon_min,
        init_positions=init_positions,
        time_offset_min=time_offset_min,
        strict=strict,
    )


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
