import numpy as np
import pandas as pd

import app.config as cfg


class DRTNetwork:
    def __init__(self):
        self.num_nodes = 0
        self.od_dur_mat = None
        self.max_duration = -1

        # GAT용 인접행렬 (정적). NUM_NODES+1 크기 — index 0은 "no node" 센티넬.
        # edge_weight[i, j] = 1 / (1 + travel_time[i, j])  (가까울수록 높음)
        # 0번 노드와의 edge_weight = epsilon (정상 노드와 분리)
        self.edge_weight = None

    def set_od_matrix(self, path):
        df = pd.read_csv(path, index_col=0)
        df.columns = df.columns.astype(int)
        self.num_nodes = len(df.index)
        assert self.num_nodes == cfg.NUM_NODES, (
            f"OD matrix has {self.num_nodes} nodes but cfg.NUM_NODES={cfg.NUM_NODES}"
        )
        self.od_dur_mat = {
            int(o): {int(d): float(df.loc[o, d]) for d in df.columns}
            for o in df.index
        }
        self.max_duration = float(
            max(df.loc[o, d] for o in df.index for d in df.columns)
        )

        N = cfg.NUM_NODES
        ew = np.zeros((N + 1, N + 1), dtype=np.float32)
        # 노드 1..N (model 입장에선 index 1..N)
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                dur = float(df.loc[i, j])
                ew[i, j] = 1.0 / (1.0 + dur)
        # 0번(=no node) self-loop만 약하게
        ew[0, 0] = 1e-3
        self.edge_weight = ew

    def get_duration(self, from_node_id, to_node_id):
        return self.od_dur_mat[from_node_id][to_node_id]
