import pandas as pd


class DRTNetwork:
    def __init__(self):
        self.num_nodes = 0
        self.od_dur_mat = None
        self.max_duration = -1

    def set_od_matrix(self, path):
        df = pd.read_csv(path, index_col=0)
        self.num_nodes = len(df.index)
        self.od_dur_mat = {
            int(o): {int(d): df.loc[o, d] for d in df.columns}
            for o in df.index
        }
        self.max_duration = max(
            df.loc[o, d] for o in df.index for d in df.columns
        )

    def get_duration(self, from_node_id, to_node_id):
        return self.od_dur_mat[from_node_id][to_node_id]