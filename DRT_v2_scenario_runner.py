"""
DRT_v2 IH를 공간 편향 시나리오(S1~S4) 위에서 한 번 돌려보는 진입점.

- 알고리즘은 DRT_v2_230823.DrtScheduler 그대로 (변경 없음)
- 시나리오 입력 생성 + DRT_v2 record 스키마 어댑터 + 실행만 담당
- 파라미터는 파일 상단 상수로 박아두고 수정해서 실험

시나리오 생성 원리는 `시나리오 생성 코드.ipynb`의 generate_requests를 계승.
인원 분포(geometric p=0.75)는 `3_request_generation.ipynb` 따라.
"""
import contextlib
import io
import os
import random
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import DRT_v2_230823 as v2

# DRT_v2가 데이터 로드 시 첫 행을 pp.pprint로 찍는 게 시끄러우니 차단.
v2.pp.pprint = lambda *a, **kw: None


# ===========================================================================
# 하드코딩 파라미터 — 수정해서 실험
# ===========================================================================
SCENARIO = "S1"                    # "S1" | "S2" | "S3" | "S4"
SEED = 0
N_REQ = 80
T_HORIZON_MIN = 60                 # 시뮬레이션 입력 시간창 (분)
BASE_HOUR = 11                     # 시뮬 시작 시각 (h). 요청 시각 = BASE_HOUR + t//60
NUM_DRT = 4
LAMBDA_BASE = 1.0                  # 일반 노드 람다
LAMBDA_HIGH = 6.0                  # 고수요 노드 람다 (= 6배 가중)
PICKUP_DEADLINE_MARGIN = 10        # d2 = d1 + margin (분)
ARRIVAL_DEADLINE_MARGIN = 10       # a2 = a1 + margin (분)
POPULATION_GEOMETRIC_P = 0.75      # 인원 ~ Geometric(p)


# ===========================================================================
# 시나리오 프리셋 — 노드별 고수요 집합
# (Sioux Falls 24-노드 네트워크 기준)
# ===========================================================================
SCENARIOS = {
    "S1": (11, 10, 14, 15),         # 중앙~남서
    "S2": (1, 3, 4),                # 북서 코너
    "S3": (2, 5, 6, 13, 24, 23),    # 동쪽 + 남쪽 광역
    "S4": None,                     # 균등 (baseline)
}


# ===========================================================================
# Scenario request generation + DRT_v2 record 어댑터
# ===========================================================================
def generate_scenario_records(scenario, seed, n_req, t_horizon, base_hour,
                              lambda_base, lambda_high, od_dict):
    """공간 편향 시나리오 + 시간 분포로 요청 생성, DRT_v2 record 형식으로 반환.

    Returns: list[dict] — 각 dict는 v2 record 스키마
        (id, request_hour, request_min, from_node_id, to_node_id, population,
         d1_hour, d1_min, d2_hour, d2_min, a1_hour, a1_min, a2_hour, a2_min)
    """
    rng = np.random.default_rng(seed)
    high_nodes = SCENARIOS[scenario]
    node_ids = np.arange(1, 25)

    # 시간: 60분에 균등 multinomial로 분산
    p_min = np.full(t_horizon, 1.0 / t_horizon)
    counts_min = rng.multinomial(n_req, p_min)
    times_min = np.repeat(np.arange(t_horizon), counts_min).astype(int)

    # Origin: 람다 가중 → 정규화 확률
    lambdas = np.full(node_ids.shape, float(lambda_base))
    if high_nodes is not None:
        for n in high_nodes:
            lambdas[node_ids == n] = lambda_high
    p_origin = lambdas / lambdas.sum()
    orig = rng.choice(node_ids, size=n_req, p=p_origin)

    # Destination: 균등, O=D면 재추첨
    dest = rng.choice(node_ids, size=n_req)
    same = (orig == dest)
    while np.any(same):
        dest[same] = rng.choice(node_ids, size=int(np.sum(same)))
        same = (orig == dest)

    # 인원: geometric(p)
    pop = rng.geometric(p=POPULATION_GEOMETRIC_P, size=n_req)

    # 시각 순 정렬
    sort_idx = np.argsort(times_min, kind="stable")
    times_min = times_min[sort_idx]
    orig = orig[sort_idx]
    dest = dest[sort_idx]
    pop = pop[sort_idx]

    # DRT_v2 record 어댑터
    records = []
    for i in range(n_req):
        t = int(times_min[i])
        o = int(orig[i])
        d = int(dest[i])

        d1_t = base_hour * 60 + t                    # 요청 시각 (= d1)
        d2_t = d1_t + PICKUP_DEADLINE_MARGIN         # 픽업 deadline
        a1_t = d1_t + int(od_dict[o][d])             # 직행 도착 시각
        a2_t = a1_t + ARRIVAL_DEADLINE_MARGIN        # 도착 deadline

        records.append({
            "id": i + 1,
            "request_hour": d1_t // 60,
            "request_min": d1_t % 60,
            "from_node_id": o,
            "to_node_id": d,
            "population": int(pop[i]),
            "d1_hour": d1_t // 60,
            "d1_min": d1_t % 60,
            "d2_hour": d2_t // 60,
            "d2_min": d2_t % 60,
            "a1_hour": a1_t // 60,
            "a1_min": a1_t % 60,
            "a2_hour": a2_t // 60,
            "a2_min": a2_t % 60,
        })

    return records


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print(f"Scenario: {SCENARIO}  high_nodes={SCENARIOS[SCENARIO]}")
    print(f"  seed={SEED}  N_REQ={N_REQ}  horizon={T_HORIZON_MIN}min  "
          f"NUM_DRT={NUM_DRT}  λ_high/λ_base={LAMBDA_HIGH}/{LAMBDA_BASE}")
    print("=" * 70)

    sc = v2.DrtScheduler()

    # 네트워크/OD 로드 (read_* 메서드들의 첫줄 print 차단)
    with contextlib.redirect_stdout(io.StringIO()):
        sc.read_nodes("data/node_list.json")
        sc.read_links("data/link_list.json")
        sc.read_od_cost("data/od_travel_time_dict.json")

    # 시나리오 record 주입 (read_requests 우회)
    records = generate_scenario_records(
        scenario=SCENARIO, seed=SEED, n_req=N_REQ,
        t_horizon=T_HORIZON_MIN, base_hour=BASE_HOUR,
        lambda_base=LAMBDA_BASE, lambda_high=LAMBDA_HIGH,
        od_dict=sc.od_dict,
    )
    sc.record_list = records
    sc.record_dict = {r["id"]: r for r in records}

    sc.preprocess_requests()
    random.seed(SEED)
    sc.generate_drt(num_drt=NUM_DRT)
    sc.run_insertion_heuristic()
    sc.show_statistics(verbose=True)


if __name__ == "__main__":
    main()
