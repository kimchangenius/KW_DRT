"""공간 편향 시나리오(S1~S4) 수요 CSV 생성기.

학습 파이프라인(`app/env_builder.py`)이 읽는 컬럼 형식으로 저장한다.
    User_ID, Start_node, End_node, Request_time, Population

원래 `DRT_v2_scenario_runner.py`의 generate_scenario_records 로직을 그대로 가져오되,
DRT_v2 모듈 의존을 끊고 OD는 `data/od_matrix.csv`에서 읽는다.

사용 예
-------
$ python scripts/gen_scenario_csv.py --scenario S2 --seed 0 --n 320 \
        --horizon 240 --out data/requests_S2_seed0_n320.csv

$ python scripts/gen_scenario_csv.py --all  # S1~S4, seed 0~4 일괄 생성
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd


SCENARIOS = {
    "S1": (11, 10, 14, 15),
    "S2": (1, 3, 4),
    "S3": (2, 5, 6, 13, 24, 23),
    "S4": None,
}

DEFAULT_LAMBDA_BASE = 1.0
DEFAULT_LAMBDA_HIGH = 6.0
DEFAULT_POP_P = 0.75


def load_od_dict(od_matrix_path):
    """od_matrix.csv → dict[int o][int d] = duration."""
    df = pd.read_csv(od_matrix_path, index_col=0)
    df.columns = df.columns.astype(int)
    return {
        int(o): {int(d): float(df.loc[o, d]) for d in df.columns}
        for o in df.index
    }


def generate_scenario_rows(
    scenario,
    seed,
    n_req,
    t_horizon,
    od_dict,
    lambda_base=DEFAULT_LAMBDA_BASE,
    lambda_high=DEFAULT_LAMBDA_HIGH,
    pop_p=DEFAULT_POP_P,
):
    """시나리오 기반으로 (User_ID, Start, End, Time, Population) 행 리스트 생성."""
    rng = np.random.default_rng(seed)
    high_nodes = SCENARIOS[scenario]
    node_ids = np.array(sorted(od_dict.keys()))  # 1..24

    p_min = np.full(t_horizon, 1.0 / t_horizon)
    counts_min = rng.multinomial(n_req, p_min)
    times_min = np.repeat(np.arange(t_horizon), counts_min).astype(int)

    lambdas = np.full(node_ids.shape, float(lambda_base))
    if high_nodes is not None:
        for n in high_nodes:
            lambdas[node_ids == n] = lambda_high
    p_origin = lambdas / lambdas.sum()
    orig = rng.choice(node_ids, size=n_req, p=p_origin)

    dest = rng.choice(node_ids, size=n_req)
    same = orig == dest
    while np.any(same):
        dest[same] = rng.choice(node_ids, size=int(np.sum(same)))
        same = orig == dest

    pop = rng.geometric(p=pop_p, size=n_req)

    sort_idx = np.argsort(times_min, kind="stable")
    times_min = times_min[sort_idx]
    orig = orig[sort_idx]
    dest = dest[sort_idx]
    pop = pop[sort_idx]

    rows = []
    for i in range(n_req):
        rows.append({
            "User_ID": i + 1,
            "Start_node": int(orig[i]),
            "End_node": int(dest[i]),
            "Request_time": int(times_min[i]),
            "Population": int(pop[i]),
        })
    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["User_ID", "Start_node", "End_node", "Request_time", "Population"]
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="S1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", dest="n_req", type=int, default=80, help="요청 수")
    p.add_argument("--horizon", type=int, default=60, help="시뮬레이션 시간창(분)")
    p.add_argument("--lambda-base", type=float, default=DEFAULT_LAMBDA_BASE)
    p.add_argument("--lambda-high", type=float, default=DEFAULT_LAMBDA_HIGH)
    p.add_argument("--pop-p", type=float, default=DEFAULT_POP_P, help="Geometric 인원 분포의 p")
    p.add_argument("--od", default="data/od_matrix.csv")
    p.add_argument("--out", default=None, help="결과 CSV 경로 (미지정 시 data/requests_<S>_seed<n>_n<N>.csv)")
    p.add_argument("--all", action="store_true", help="S1~S4 × seed 0~(--seeds-1) 일괄 생성")
    p.add_argument("--seeds", type=int, default=5, help="--all 모드에서 사용할 seed 개수")
    return p.parse_args()


def default_out(scenario, seed, n_req):
    return f"data/requests_{scenario}_seed{seed}_n{n_req}.csv"


def scenario_request_filename(scenario, seed, n_req, t_horizon=None):
    parts = [f"requests_{scenario}", f"seed{seed}", f"n{n_req}"]
    if t_horizon is not None:
        parts.append(f"h{t_horizon}")
    return "_".join(parts) + ".csv"


def generate_scenario_csv(
    data_dir,
    scenario="S1",
    seed=0,
    n_req=80,
    t_horizon=60,
    lambda_base=DEFAULT_LAMBDA_BASE,
    lambda_high=DEFAULT_LAMBDA_HIGH,
    pop_p=DEFAULT_POP_P,
    od_filename="od_matrix.csv",
    out_dir=None,
    out_path=None,
):
    """main.py 등에서 바로 호출할 수 있는 시나리오 CSV 생성 헬퍼."""
    od_path = os.path.join(data_dir, od_filename)
    od_dict = load_od_dict(od_path)
    rows = generate_scenario_rows(
        scenario=scenario,
        seed=seed,
        n_req=n_req,
        t_horizon=t_horizon,
        od_dict=od_dict,
        lambda_base=lambda_base,
        lambda_high=lambda_high,
        pop_p=pop_p,
    )
    if out_path is None:
        out_dir = out_dir or data_dir
        out_path = os.path.join(
            out_dir,
            scenario_request_filename(scenario, seed, n_req, t_horizon),
        )
    write_csv(rows, out_path)
    return out_path


def main():
    args = parse_args()
    od_dict = load_od_dict(args.od)

    if args.all:
        for sc in SCENARIOS.keys():
            for seed in range(args.seeds):
                rows = generate_scenario_rows(
                    scenario=sc, seed=seed, n_req=args.n_req, t_horizon=args.horizon,
                    od_dict=od_dict, lambda_base=args.lambda_base,
                    lambda_high=args.lambda_high, pop_p=args.pop_p,
                )
                out = default_out(sc, seed, args.n_req)
                write_csv(rows, out)
                print(f"  wrote {out}  (n={len(rows)})")
        return

    rows = generate_scenario_rows(
        scenario=args.scenario, seed=args.seed, n_req=args.n_req,
        t_horizon=args.horizon, od_dict=od_dict,
        lambda_base=args.lambda_base, lambda_high=args.lambda_high,
        pop_p=args.pop_p,
    )
    out = args.out or default_out(args.scenario, args.seed, args.n_req)
    write_csv(rows, out)

    pop_arr = np.array([r["Population"] for r in rows])
    print(f"wrote {out}  (n={len(rows)})")
    print(f"  scenario={args.scenario}  seed={args.seed}  horizon={args.horizon}min")
    print(f"  pop: mean={pop_arr.mean():.2f}  max={pop_arr.max()}  >{5}={int((pop_arr > 5).sum())}")


if __name__ == "__main__":
    main()
