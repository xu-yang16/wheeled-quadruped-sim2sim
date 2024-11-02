import os
import os.path as osp
import rerun as rr


class RRManager:
    def __init__(self):
        rr.init("wheeled_sim", spawn=True)

    def reset(self):
        pass

    def log(self, line_name: str, value: float):
        rr.log(line_name, rr.Scalar(value))

    def plot(self, save_fig=True, save_data=True, log_dir=None):
        if save_data:
            file_name = osp.join(log_dir, "rr_plot", "data.rrd")
            os.makedirs(osp.dirname(file_name), exist_ok=True)
            rr.save(file_name)

    def close(self):
        rr.disconnect()
